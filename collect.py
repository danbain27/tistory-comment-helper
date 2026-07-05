# -*- coding: utf-8 -*-
"""
티스토리 블로그 '발견 + 본문 수집' 도구 (댓글 자동 등록 없음)

내 키워드와 비슷한 티스토리 블로그 글을 찾아서, 본문을 읽기 좋게 정리해
reports/ 폴더에 마크다운으로 저장한다.
그 파일을 Claude에게 보여주면 글마다 맞춤 댓글 초안을 뽑아준다.
댓글 등록은 사람이 직접 한다.

사용법:
    python collect.py            # 기본 10개 수집
    python collect.py 5          # 5개만 수집
"""

import sys
import os
import re
import json
import time
import random
import datetime
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# 콘솔이 cp949여도 한글 출력이 깨지지 않게
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────
# 설정 (여기만 고치면 됨)
# ─────────────────────────────────────────────
KEYWORDS = [
    # 프라모델
    "건프라 리뷰",
    "프라모델 제작",
    "건프라 도색",
    # 게임 (슈로대 + 일반 리뷰/공략)
    "슈퍼로봇대전 공략",
    "게임 리뷰",
    "게임 공략",
    "스팀 게임 리뷰",
    "닌텐도 스위치 리뷰",
    "PS5 게임 공략",
    "콘솔 게임 리뷰",
    "RPG 공략",
]
TARGET_COUNT = 10          # 하루에 찾을 서로 다른 블로그 수
WITHIN_DAYS = 31           # 이 일수 이내에 발행된 글만 (발행일 확인된 경우)
ONE_PER_BLOG = True        # 같은 블로그(서브도메인)는 하루 1개까지만
MIN_DELAY, MAX_DELAY = 1.5, 4.0   # 요청 사이 예의상 대기(초)
EXCERPT_CHARS = 1800       # 리포트에 담을 본문 길이

# ── 광고/제휴글 자동 제외 ──────────────────────
FILTER_ADS = True
# 본문에 아래 문구가 있으면 광고/제휴글로 판단 (제휴 고지 문구)
AD_TEXT_MARKERS = [
    "쿠팡 파트너스", "쿠팡파트너스", "파트너스 활동", "활동의 일환", "일환으로",
    "수수료를 제공", "수수료를 지급", "수수료를 받", "네이버 쇼핑 커넥트", "쇼핑 커넥트",
    "제휴 마케팅", "제휴마케팅", "소정의 수수료", "원고료", "협찬", "체험단",
]
# 본문에 이런 쇼핑/제휴 링크가 있으면 광고글
AD_LINK_HOSTS = [
    "link.coupang.com", "coupa.ng", "coupang.com", "smartstore.naver.com",
    "shopping.naver.com", "aliexpress", "ali.ski", "iherb.com", "amzn.to",
]
# 제목에 이런 단어가 있으면 광고/판매성 글
AD_TITLE_MARKERS = [
    "최저가", "가격 비교", "가격비교", "핫딜", "특가", "구매 가이드", "구매가이드",
    "내돈내산", "추천 top", "top 10", "top 5", "top10", "top5", "할인 정보",
]

# ── 내 블로그(제외) ───────────────────────────
OWN_BLOGS = ["danbain.tistory.com"]

# ── 주제 관련성 필터 ─────────────────────────
# 제목/본문에 아래 단어가 하나도 없으면 딴 주제(오검색·정책페이지 등)로 보고 제외
REQUIRE_RELEVANCE = True
RELEVANCE_TERMS = [
    "건프라", "프라모델", "건담", "반다이", "도색", "먹선", "조립", "피규어", "디오라마", "프라모",
    "슈퍼로봇대전", "슈로대", "게임", "공략", "플레이", "리뷰", "클리어", "엔딩", "보스", "스테이지",
    "닌텐도", "스위치", "플스", "ps5", "ps4", "스팀", "steam", "rpg", "로그라이크", "콘솔",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
HISTORY_PATH = os.path.join(BASE_DIR, "history.json")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
}


# ─────────────────────────────────────────────
# 검색: ddgs (DuckDuckGo, 차단/챌린지 처리 포함)
# ─────────────────────────────────────────────
def search(query, limit=25):
    """키워드로 티스토리 글 URL 목록을 반환."""
    urls = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(
                f"{query} site:tistory.com",
                region="kr-kr",
                timelimit="m",   # 최근 1개월로 검색 단계에서 1차 제한
                max_results=limit,
            ):
                href = r.get("href", "")
                if "tistory.com" in href:
                    urls.append(href)
    except Exception as e:
        print(f"    [검색 오류] {e}")
    return urls


# ─────────────────────────────────────────────
# URL 필터: 실제 '글' URL만 (블로그 홈/태그/카테고리 제외)
# ─────────────────────────────────────────────
def is_post_url(url):
    try:
        p = urlparse(url)
    except Exception:
        return False
    host = p.netloc.lower()
    if not host.endswith("tistory.com"):
        return False
    if host in ("www.tistory.com", "tistory.com"):
        return False
    path = p.path.strip("/")
    if not path:
        return False  # 블로그 홈
    if path.startswith(("category", "tag", "guestbook", "search", "manage", "notice")):
        return False
    # /숫자  또는  /entry/제목  형태만 글로 인정
    if re.fullmatch(r"\d+", path):
        return True
    if path.startswith("entry/"):
        return True
    return False


def blog_host(url):
    return urlparse(url).netloc.lower()


# ─────────────────────────────────────────────
# 본문 추출
# ─────────────────────────────────────────────
BODY_SELECTORS = [
    "div.tt_article_useless_p_margin",
    "div.article_view",
    "div.contents_style",
    "div.entry-content",
    "div.article",
    "div#content",
    "article",
]


def clean_text(node):
    for bad in node.select("script, style, ins, iframe, .revenue_unit_wrap, .container_postbtn"):
        bad.decompose()
    text = node.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def parse_date(soup, body_text):
    """게시글 발행일을 datetime.date로 파싱. 못 찾으면 None."""
    # 1) 메타 태그 (ISO)
    for prop in ("article:published_time", "og:regDate"):
        m = soup.find("meta", property=prop)
        if m and m.get("content"):
            try:
                return datetime.date.fromisoformat(m["content"][:10])
            except ValueError:
                pass
    # 2) <time datetime="...">
    t = soup.find("time")
    if t and t.get("datetime"):
        try:
            return datetime.date.fromisoformat(t["datetime"][:10])
        except ValueError:
            pass
    # 3) 본문/헤더의 한국식 날짜: "2025. 8. 27." / "2025-08-27" / "2025.08.27"
    head = body_text[:600]
    mo = re.search(r"(20\d{2})[.\-]\s*(\d{1,2})[.\-]\s*(\d{1,2})", head)
    if mo:
        try:
            return datetime.date(int(mo.group(1)), int(mo.group(2)), int(mo.group(3)))
        except ValueError:
            pass
    return None


def detect_ad(title, body_text, soup):
    """광고/제휴글이면 사유 문자열을, 아니면 None을 반환."""
    low_title = (title or "").lower()
    for kw in AD_TITLE_MARKERS:
        if kw in low_title:
            return f"제목에 '{kw}'"
    for kw in AD_TEXT_MARKERS:
        if kw in body_text:
            return f"본문에 제휴 고지 '{kw}'"
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        for host in AD_LINK_HOSTS:
            if host in href:
                return f"쇼핑/제휴 링크({host})"
    return None


def fetch_post(url):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.encoding = resp.apparent_encoding or "utf-8"
    soup = BeautifulSoup(resp.text, "html.parser")

    # 제목
    title = None
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title:
        title = soup.title.get_text(strip=True)

    # 본문: 후보 셀렉터 중 텍스트가 가장 긴 것
    best = ""
    for sel in BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            txt = clean_text(node)
            if len(txt) > len(best):
                best = txt
    if not best:
        body = soup.find("body")
        best = clean_text(body) if body else ""

    date_obj = parse_date(soup, best)
    resolved_title = title or "(제목 없음)"
    ad_reason = detect_ad(resolved_title, best, soup)

    return {
        "title": resolved_title,
        "date_obj": date_obj,
        "date": date_obj.isoformat() if date_obj else "(날짜 미상)",
        "text": best,
        "ad_reason": ad_reason,
    }


# ─────────────────────────────────────────────
# 히스토리 (이미 리포트에 넣은 URL은 다시 안 뽑음)
# ─────────────────────────────────────────────
def load_history():
    if os.path.exists(HISTORY_PATH):
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"urls": []}


def save_history(history):
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def collect_posts(keywords, target, exclude_urls=None, log=None):
    """키워드로 조건에 맞는 글을 target개 모아 리스트로 반환. (파일 저장 안 함)

    CLI(main)와 웹앱(app.py)이 함께 쓰는 핵심 수집 함수.
    log: 진행 상황을 받을 콜백 (기본은 조용히).
    """
    if log is None:
        log = lambda *a: None
    exclude_urls = set(exclude_urls or [])
    seen_hosts = set()
    processed = set()   # 이번 수집에서 이미 확인한 URL — 재요청 방지
    cutoff = datetime.date.today() - datetime.timedelta(days=WITHIN_DAYS)

    keyword_results = {kw: search(kw) for kw in keywords}
    for kw in keywords:
        log(f"[검색] {kw} → {len(keyword_results[kw])}건")

    collected = []
    round_idx = 0
    exhausted = False
    while len(collected) < target and not exhausted:
        exhausted = True
        for kw in keywords:
            results = keyword_results[kw]
            if round_idx >= len(results):
                continue
            exhausted = False
            url = results[round_idx]

            if not is_post_url(url):
                continue
            if url in exclude_urls or url in processed:
                continue
            processed.add(url)
            host = blog_host(url)
            if host in OWN_BLOGS:
                continue  # 내 블로그 제외
            if ONE_PER_BLOG and host in seen_hosts:
                continue

            log(f"  ({len(collected)+1}/{target}) 수집중: {url}")
            try:
                post = fetch_post(url)
            except Exception as e:
                log(f"      [실패] {e}")
                continue

            if len(post["text"]) < 150:
                log("      [건너뜀] 본문이 너무 짧음")
                continue
            if post["date_obj"] is None:
                log("      [건너뜀] 발행일 확인 불가")
                continue
            if post["date_obj"] < cutoff:
                log(f"      [건너뜀] 1개월 초과 ({post['date']})")
                continue
            if FILTER_ADS and post["ad_reason"]:
                log(f"      [건너뜀] 광고/제휴글 — {post['ad_reason']}")
                continue
            if REQUIRE_RELEVANCE:
                haystack = (post["title"] + " " + post["text"][:1200]).lower()
                if not any(term.lower() in haystack for term in RELEVANCE_TERMS):
                    log("      [건너뜀] 주제 무관")
                    continue

            post["url"] = url
            post["host"] = host
            post["keyword"] = kw
            post["excerpt"] = post["text"][:EXCERPT_CHARS] + (
                " …(생략)" if len(post["text"]) > EXCERPT_CHARS else ""
            )
            collected.append(post)
            seen_hosts.add(host)

            time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
            if len(collected) >= target:
                break
        round_idx += 1
    return collected


def main():
    target = TARGET_COUNT
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        target = int(sys.argv[1])

    os.makedirs(REPORTS_DIR, exist_ok=True)
    history = load_history()
    print(f"키워드 {len(KEYWORDS)}개로 티스토리 글 {target}개 수집 시작 (발행 {WITHIN_DAYS}일 이내)\n")

    collected = collect_posts(KEYWORDS, target, exclude_urls=history["urls"], log=print)

    if not collected:
        print("\n수집된 글이 없습니다. 키워드를 바꾸거나 잠시 후 다시 시도하세요.")
        return

    # 리포트 저장
    today = datetime.date.today().isoformat()
    n = 1
    while True:
        out_path = os.path.join(REPORTS_DIR, f"{today}_{n}.md")
        if not os.path.exists(out_path):
            break
        n += 1

    write_report(out_path, collected, today)

    # 히스토리 갱신
    history["urls"] = list(set(history["urls"]) | {p["url"] for p in collected})
    save_history(history)

    print(f"\n완료: {len(collected)}개 수집 → {out_path}")
    print("이 파일을 Claude에게 보여주면 글마다 맞춤 댓글 초안을 만들어 줍니다.")


def write_report(path, posts, today):
    lines = []
    lines.append(f"# 티스토리 교류 후보 — {today}")
    lines.append("")
    lines.append(f"- 수집 {len(posts)}건 · 자동 등록 없음(수동 댓글용)")
    lines.append("- 사용법: 이 파일을 Claude에게 붙여넣고 \"글마다 댓글 초안 뽑아줘\"라고 요청")
    lines.append("- 매너: 글을 실제로 끝까지 읽고, 본문 내용에 맞는 진심 댓글만 남기세요.")
    lines.append("")
    for i, p in enumerate(posts, 1):
        excerpt = p["text"][:EXCERPT_CHARS]
        if len(p["text"]) > EXCERPT_CHARS:
            excerpt += " …(생략)"
        lines.append("---")
        lines.append("")
        lines.append(f"## {i}. {p['title']}")
        lines.append("")
        lines.append(f"- URL: {p['url']}")
        lines.append(f"- 블로그: {p['host']}")
        lines.append(f"- 날짜: {p['date']} · 검색 키워드: {p['keyword']}")
        lines.append("")
        lines.append("**본문 발췌**")
        lines.append("")
        lines.append("> " + excerpt.replace("\n", "\n> "))
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
