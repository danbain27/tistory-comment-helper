# -*- coding: utf-8 -*-
"""
최근 N일(기본 2일) 이내 티스토리 글 중, 내 블로그와 비슷한 게임 글 + 댓글 허용 글만 찾기
(댓글 자동 등록 없음)

티스토리 자체 검색 API(최신순)로 찾기 때문에 발행일이 정확하다.
각 글 페이지에 댓글 영역이 있는지 확인해 '댓글 가능' 글만 리포트에 담는다.

사용법:
    python find_recent.py            # 2일 이내, 최대 10개
    python find_recent.py 5          # 5개만
    python find_recent.py 10 7       # 7일 이내 10개
"""

import sys
import os
import re
import json
import datetime
import html
import pathlib
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote, urlparse

import requests
from bs4 import BeautifulSoup

from collect import (
    HEADERS, OWN_BLOGS, REPORTS_DIR, BODY_SELECTORS, EXCERPT_CHARS,
    clean_text, detect_ad, load_history, save_history,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
WITHIN_DAYS = 2
TARGET_COUNT = 10
MAX_PAGES = 3              # 키워드당 최신순 검색 페이지 수
WORKERS = 8                # 동시에 보내는 요청 수 (속도)

# 내 블로그(슈로대·파엠·레트로 SRPG·스위치 게임)와 비슷한 게임 키워드
KEYWORDS = [
    "만자천홍", "파이어 엠블렘", "슈퍼로봇대전", "슈로대", "마장기신",
    "SRPG", "시뮬레이션 RPG", "전략 RPG", "택틱스", "트라이앵글 스트래티지",
    "유니콘 오버로드", "레트로 게임", "고전 게임", "슈퍼패미컴", "한글패치",
    "발라트로", "슬레이 더 스파이어", "로그라이크", "마리오 RPG", "스위치2 게임",
]

# 관련도 점수: 비슷한 장르 단어는 3점, 일반 게임 단어는 1점 (제목에 있으면 2배)
# 검색 API가 느슨하게 매칭해서 딴 글이 섞이므로 MIN_SCORE 미만은 버린다
GENRE_TERMS = [
    "만자천홍", "파이어 엠블렘", "파이어엠블렘", "파엠", "슈퍼로봇대전", "슈로대", "마장기신",
    "srpg", "시뮬레이션 rpg", "전략 rpg", "택틱스", "트라이앵글 스트래티지", "유니콘 오버로드",
    "발라트로", "슬레이 더 스파이어", "로그라이크", "덱빌딩", "마리오 rpg", "슈퍼패미컴",
    "패미컴", "sfc", "gba", "게임보이", "ps2", "ps1", "레트로 게임", "고전 게임", "한글패치",
]
GAME_TERMS = [
    "공략", "게임", "rpg", "닌텐도", "스위치2", "스위치 게임", "스팀", "steam", "ps5", "ps4",
    "플레이스테이션", "엔딩", "회차", "보스전",
]
MIN_SCORE = 4

SEARCH_API = "https://www.tistory.com/api/v1/search/posts?keyword={kw}&page={page}&sort=RECENCY"


# ─────────────────────────────────────────────
# 검색 (티스토리 검색 API, 최신순)
# ─────────────────────────────────────────────
def search_recent(keyword, cutoff):
    """cutoff(date) 이후 발행된 글을 최신순으로 모은다."""
    found = []
    for page in range(1, MAX_PAGES + 1):
        try:
            r = requests.get(SEARCH_API.format(kw=quote(keyword), page=page),
                             headers=HEADERS, timeout=10)
            entries = r.json()["data"]["searchedEntries"]
        except Exception:
            break
        if not entries:
            break
        for e in entries:
            published = e.get("entryPublished", "")[:10]
            if published and published >= cutoff.isoformat():
                found.append(e)
        # 최신순이므로 마지막 글이 기준일보다 오래되면 다음 페이지는 볼 필요 없음
        if entries[-1].get("entryPublished", "")[:10] < cutoff.isoformat():
            break
    return found


def relevance(e):
    title = e.get("entryTitle", "").lower()
    summary = e.get("entrySummary", "").lower()
    # 제목에 게임 단어가 하나도 없으면 제외 (요약만 걸린 딴 주제 글)
    if not any(t in title for t in GENRE_TERMS + GAME_TERMS):
        return 0
    score = 0
    for terms, weight in ((GENRE_TERMS, 3), (GAME_TERMS, 1)):
        for t in terms:
            if t in title:
                score += weight * 2
            elif t in summary:
                score += weight
    return score


# ─────────────────────────────────────────────
# 글 확인: 본문 + 댓글 허용 여부
# ─────────────────────────────────────────────
def comments_open(html, entry_id):
    """댓글 작성 영역이 있으면 True. 글에서 댓글을 막으면 스킨이 이 영역을 아예 출력하지 않는다."""
    if 'data-tistory-react-app="Comment"' in html:
        return True
    return re.search(rf'id="entry{entry_id}Comment"', html) is not None


def inspect_post(url, entry_id):
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    login_only = '"NEED_COMMENT_LOGIN":true' in html
    allowed = comments_open(html, entry_id)

    best = ""
    for sel in BODY_SELECTORS:
        node = soup.select_one(sel)
        if node:
            txt = clean_text(node)
            if len(txt) > len(best):
                best = txt
    return {"comments": allowed, "login_only": login_only, "text": best, "soup": soup}


# ─────────────────────────────────────────────
# 댓글 초안 (AI 없이 템플릿으로 — 무료, 등록 전 꼭 다듬기)
# ─────────────────────────────────────────────
POST_TYPES = [
    # (판별 단어, 초안 템플릿들) — {game}은 게임명, {point}는 본문 핵심 문장
    (["한글패치", "한글화", "패치"], [
        "{game} 한글패치 작업하시느라 고생 많으셨어요. 덕분에 편하게 즐길 수 있겠네요, 감사합니다!",
        "{point} 이 부분 보니 정성이 느껴지네요. {game} 꼭 해보겠습니다.",
    ]),
    (["공략", "육성", "가이드", "총정리", "팁"], [
        "{game} 공략 깔끔하게 정리해 주셔서 잘 봤습니다. 저도 공략 글 쓰는 입장이라 이렇게 정리하는 게 얼마나 손이 많이 가는지 알아요.",
        "{point} 이 부분이 특히 도움이 됐어요. 진행할 때 참고하겠습니다!",
    ]),
    (["리뷰", "후기", "소감", "클리어", "엔딩"], [
        "{game} 후기 재밌게 읽었어요. 직접 플레이하신 느낌이 잘 전해지네요.",
        "{point} 저도 이 부분 공감해요. 좋은 글 감사합니다.",
    ]),
    (["출시", "업데이트", "추가", "발매", "공개", "소식"], [
        "{game} 소식 반갑네요! 정리해 주셔서 한눈에 들어왔어요.",
        "{point} 이건 몰랐던 내용이네요. 기대됩니다.",
    ]),
]
DEFAULT_TEMPLATES = [
    "{game} 이야기 재밌게 읽었어요. 좋은 글 감사합니다!",
    "{point} 이 부분 인상 깊었어요. 자주 놀러 올게요.",
]


def game_name(title):
    """제목에서 게임명 추출: 《》·「」·[] 안 → '공략/한글패치' 앞부분 → 제목 앞부분."""
    m = re.search(r"[《「『](.+?)[》」』]", title)
    if m:
        return m.group(1).strip()
    t = re.sub(r"^\[[^\]]*\]\s*", "", title)
    m = re.match(r"(.+?)\s*(?:공략|한글\s?패치|리뷰|후기|육성|초보자|업데이트|출시|총정리|[,|｜:\-—])", t)
    name = (m.group(1) if m else t).strip()
    return name[:25] or "이 게임"


def key_point(text, title):
    """본문에서 숫자·고유 정보가 들어간 짧은 문장 하나를 골라 구체성을 준다."""
    sentences = re.split(r"(?<=[.!?다요])\s+|\n", text)
    best, best_score = "", 0
    for s in sentences:
        s = s.strip()
        if not 15 <= len(s) <= 60 or s in title:
            continue
        # 링크·날짜 목록·해시값 같은 줄 제외: 한글이 절반 이상인 문장만
        if "http" in s or re.search(r"20\d\d\.\d\d\.\d\d|\[.*\]", s):
            continue
        if len(re.findall(r"[가-힣]", s)) < len(s.replace(" ", "")) / 2:
            continue
        score = len(re.findall(r"\d", s)) + 2 * any(w in s for w in ("추천", "핵심", "중요", "주의", "팁"))
        if score > best_score:
            best, best_score = s, score
    if not best:
        return "정리해 주신 내용 중"
    return f"'{best.strip(' :·-').rstrip('.')}'"


def make_drafts(title, text):
    game = game_name(title)
    point = key_point(text, title)
    templates = DEFAULT_TEMPLATES
    for words, tpl in POST_TYPES:
        if any(w in title for w in words):
            templates = tpl
            break
    return [t.format(game=game, point=point) for t in templates]


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────
def find_posts(target, days, exclude_urls=None, log=print):
    exclude_urls = exclude_urls or set()
    cutoff = datetime.date.today() - datetime.timedelta(days=days)

    candidates = {}
    # 키워드 검색을 동시에 보내고, 결과는 키워드 순서대로 합친다
    with ThreadPoolExecutor(WORKERS) as ex:
        results = list(ex.map(lambda k: search_recent(k, cutoff), KEYWORDS))
    for kw, found in zip(KEYWORDS, results):
        for e in found:
            url = e["entryUrl"]
            host = urlparse(e["blogUrl"]).netloc.lower()
            if url in candidates or url in exclude_urls:
                continue
            score = relevance(e)
            if any(own in host for own in OWN_BLOGS) or score < MIN_SCORE:
                continue
            candidates[url] = dict(e, keyword=kw, host=host, score=score)
        log(f"[검색] {kw} → {days}일 이내 관련 글 합계 {len(candidates)}개")

    posts, seen_blogs = [], set()
    # 관련도 높은 글 먼저, 같으면 최신 글 먼저
    ranked = sorted(candidates.items(), key=lambda x: (x[1]["score"], x[1]["entryPublished"]), reverse=True)
    # 블로그당 가장 관련 높은 글 하나만 후보로
    best_per_blog = []
    for url, e in ranked:
        if e["host"] not in seen_blogs:
            seen_blogs.add(e["host"])
            best_per_blog.append((url, e))

    def check(item):
        url, e = item
        try:
            return inspect_post(url, e["entryId"]), None
        except Exception as ex:
            return None, ex

    # 필요한 개수의 두 배씩 동시에 확인하고, 관련도 순서대로 채운다
    batch = max(target * 2, WORKERS)
    for start in range(0, len(best_per_blog), batch):
        if len(posts) >= target:
            break
        chunk = best_per_blog[start:start + batch]
        with ThreadPoolExecutor(WORKERS) as ex:
            checked = list(ex.map(check, chunk))
        for (url, e), (info, err) in zip(chunk, checked):
            if len(posts) >= target:
                break
            if err:
                log(f"[건너뜀] 접속 실패 {url} ({err.__class__.__name__})")
                continue
            if not info["comments"]:
                log(f"[제외] 댓글 막힘 · {e['entryTitle'][:40]}")
                continue
            ad = detect_ad(e["entryTitle"], info["text"], info["soup"])
            if ad:
                log(f"[제외] 광고글({ad}) · {e['entryTitle'][:40]}")
                continue
            if len(info["text"]) < 150:
                continue
            posts.append({
                "title": e["entryTitle"],
                "url": url,
                "entry_id": e["entryId"],
                "host": e["host"],
                "blog": e["blogTitle"],
                "date": e["entryPublished"][:16].replace("T", " "),
                "keyword": e["keyword"],
                "comment_count": e.get("commentCount", 0),
                "like_count": e.get("likeCount", 0),
                "login_only": info["login_only"],
                "text": info["text"],
                "drafts": make_drafts(e["entryTitle"], info["text"]),
            })
            log(f"[OK] {len(posts)}/{target} {e['blogTitle']} · {e['entryTitle'][:40]}")
    return posts


def write_report(path, posts, days):
    lines = [
        f"# 최근 {days}일 티스토리 교류 후보 — {datetime.date.today()}",
        "",
        f"- {len(posts)}건 · 모두 댓글 허용 글 · 자동 등록 없음(수동 댓글용)",
        "- 사용법: 이 파일을 Claude에게 붙여넣고 \"글마다 요약하고 댓글 초안 뽑아줘\"라고 요청",
        "",
    ]
    for i, p in enumerate(posts, 1):
        excerpt = p["text"][:EXCERPT_CHARS]
        if len(p["text"]) > EXCERPT_CHARS:
            excerpt += " …(생략)"
        login = " · 로그인 필요" if p["login_only"] else ""
        lines += [
            "---", "",
            f"## {i}. {p['title']}", "",
            f"- URL: {p['url']}",
            f"- 블로그: {p['blog']} ({p['host']})",
            f"- 발행: {p['date']} · 키워드: {p['keyword']}",
            f"- 댓글: 허용{login} · 현재 댓글 {p['comment_count']} · 좋아요 {p['like_count']}",
            "", "**댓글 초안** (템플릿 — 글을 읽고 다듬어서 직접 등록)", "",
            *[f"{n}. {d}" for n, d in enumerate(p["drafts"], 1)],
            "", "**본문 발췌**", "",
            "> " + excerpt.replace("\n", "\n> "), "",
        ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


HTML_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>티스토리 교류 후보 {today}</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --text:#1d2129; --sub:#667085; --line:#e4e7ec; --accent:#2563eb; --badge:#fff4e5; --badge-text:#b54708; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14161a; --card:#1e2127; --text:#e6e8eb; --sub:#9aa3af; --line:#2e333b; --accent:#60a5fa; --badge:#3a2a12; --badge-text:#fbbf68; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.6 "Malgun Gothic","Apple SD Gothic Neo",sans-serif; }}
main {{ max-width:860px; margin:0 auto; padding:24px 16px 60px; }}
h1 {{ font-size:22px; margin:0 0 4px; }}
.lead {{ color:var(--sub); margin:0 0 20px; font-size:14px; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:18px; margin-bottom:14px; }}
.card.done {{ opacity:.55; }}
.title {{ font-size:17px; font-weight:700; color:var(--accent); text-decoration:none; }}
.title:hover {{ text-decoration:underline; }}
.meta {{ color:var(--sub); font-size:13px; margin:4px 0 10px; }}
.badge {{ background:var(--badge); color:var(--badge-text); border-radius:6px; padding:1px 7px; font-size:12px; margin-left:6px; }}
.excerpt {{ color:var(--sub); font-size:14px; margin:0 0 12px; }}
.draft {{ display:flex; gap:8px; align-items:flex-start; border-top:1px dashed var(--line); padding:8px 0; }}
.draft p {{ margin:0; flex:1; }}
button {{ border:1px solid var(--line); background:var(--bg); color:var(--text); border-radius:8px; padding:4px 10px; cursor:pointer; font:inherit; font-size:13px; white-space:nowrap; }}
button:hover {{ border-color:var(--accent); }}
.actions {{ display:flex; gap:8px; margin-top:8px; }}
.open {{ background:var(--accent); color:#fff; border-color:var(--accent); text-decoration:none; border-radius:8px; padding:4px 12px; font-size:13px; }}
</style></head><body><main>
<h1>티스토리 교류 후보 · {today}</h1>
<p class="lead">최근 {days}일 · 비슷한 게임 · 댓글 허용 글 {count}개. 제목을 눌러 글을 읽고, 초안을 복사해 다듬어서 직접 댓글을 남기세요.</p>
{cards}
</main>
<script>
function copyText(btn) {{
  const text = btn.parentElement.querySelector('p').innerText;
  navigator.clipboard.writeText(text).then(() => {{ btn.innerText = '복사됨'; setTimeout(() => btn.innerText = '복사', 1500); }});
}}
function toggleDone(btn) {{
  const card = btn.closest('.card');
  card.classList.toggle('done');
  btn.innerText = card.classList.contains('done') ? '완료 취소' : '댓글 완료';
}}
</script></body></html>
"""


def write_html_report(path, posts, days):
    e = html.escape
    cards = []
    for i, p in enumerate(posts, 1):
        login = '<span class="badge">로그인 필요</span>' if p["login_only"] else ""
        excerpt = re.sub(r"\s+", " ", p["text"])[:220] + "…"
        drafts = "".join(
            f'<div class="draft"><p>{e(d)}</p><button onclick="copyText(this)">복사</button></div>'
            for d in p["drafts"]
        )
        cards.append(
            f'<div class="card"><a class="title" href="{e(p["url"])}" target="_blank" rel="noopener">{i}. {e(p["title"])}</a>'
            f'<div class="meta">{e(p["blog"])} · {e(p["date"])} · 댓글 {p["comment_count"]} · 좋아요 {p["like_count"]}{login}</div>'
            f'<p class="excerpt">{e(excerpt)}</p>{drafts}'
            f'<div class="actions"><a class="open" href="{e(p["url"])}" target="_blank" rel="noopener">글 열기</a>'
            f'<button onclick="toggleDone(this)">댓글 완료</button></div></div>'
        )
    page = HTML_PAGE.format(today=datetime.date.today(), days=days, count=len(posts), cards="\n".join(cards))
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = int(args[0]) if len(args) > 0 else TARGET_COUNT
    days = int(args[1]) if len(args) > 1 else WITHIN_DAYS

    history = load_history()
    posts = find_posts(target, days, exclude_urls=set(history.get("urls", [])))
    if not posts:
        print("조건에 맞는 글을 찾지 못했습니다.")
        sys.exit(1)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"recent_{datetime.date.today()}.md")
    write_report(path, posts, days)
    html_path = path[:-3] + ".html"
    write_html_report(html_path, posts, days)
    # comment_assist.py 가 읽는 목록
    with open(path[:-3] + ".json", "w", encoding="utf-8") as f:
        json.dump([{k: p[k] for k in ("title", "url", "entry_id", "blog", "drafts")} for p in posts],
                  f, ensure_ascii=False, indent=2)

    history.setdefault("urls", []).extend(p["url"] for p in posts)
    save_history(history)
    print(f"\n저장: {html_path}")
    if "--no-open" not in sys.argv:
        webbrowser.open(pathlib.Path(html_path).resolve().as_uri())


if __name__ == "__main__":
    main()
