# -*- coding: utf-8 -*-
"""
댓글 입력 도우미 — 글을 하나씩 열고 댓글 칸에 초안을 넣어 둔다.
**등록 버튼은 절대 누르지 않는다.** 사람이 읽고 고친 뒤 직접 [등록]을 누른다.

사용법:
    python find_recent.py       # 먼저 후보 찾기 (reports/recent_날짜.json 생성)
    python comment_assist.py    # 가장 최근 목록으로 시작

처음 한 번은 열린 브라우저에서 티스토리(카카오) 로그인을 해야 한다.
로그인 정보는 browser_profile/ 폴더에 남아 다음부터는 바로 시작된다.
"""

import os
import sys
import glob
import json
import queue
import threading
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROFILE_DIR = os.path.join(BASE_DIR, "browser_profile", "chrome")

READ_SECONDS = 90   # 글을 연 뒤 댓글 칸으로 가기 전까지 읽는 시간(초)
SCROLL_EVERY = 3    # 읽는 동안 몇 초마다 조금씩 내려갈지

# 스킨마다 댓글 칸 구조가 달라서 후보를 차례로 찾는다
COMMENT_BOX_SELECTORS = [
    '[data-tistory-react-app="Comment"] textarea',
    '[data-tistory-react-app="Comment"] [contenteditable="true"]',
    'div[id$="Comment"] textarea',
    'textarea[name="comment"]',
    'form[method="post"] textarea',
    'textarea',
]


def latest_list():
    files = sorted(glob.glob(os.path.join(REPORTS_DIR, "recent_*.json")))
    if not files:
        sys.exit("reports/recent_날짜.json 이 없습니다. 먼저 py find_recent.py 를 실행하세요.")
    with open(files[-1], encoding="utf-8") as f:
        return files[-1], json.load(f)


def launch(p):
    """설치된 크롬으로 연다. 크롬이 없으면 Playwright 내장 Chromium."""
    for channel in ("chrome", None):
        try:
            return p.chromium.launch_persistent_context(
                PROFILE_DIR, channel=channel, headless=False, no_viewport=True,
                args=["--start-maximized"],
            )
        except Exception:
            continue
    sys.exit("크롬을 열 수 없습니다. 크롬이 설치돼 있는지 확인하거나 py -m playwright install chromium 을 실행해 보세요.")


def find_comment_box(page):
    # 댓글 영역은 스크롤해야 불러오는 스킨이 있어 조금씩 맨 아래까지 내린다
    for _ in range(12):
        page.mouse.wheel(0, 2500)
        page.wait_for_timeout(250)
    app = page.locator('[data-tistory-react-app="Comment"]').first
    try:
        app.scroll_into_view_if_needed(timeout=3000)
        page.wait_for_timeout(1000)
    except Exception:
        pass
    for sel in COMMENT_BOX_SELECTORS:
        # 숨겨진 답글 칸 등은 건너뛰고 화면에 보이는 칸만
        box = page.locator(f"{sel} >> visible=true").first
        try:
            box.wait_for(state="visible", timeout=3000)
            return box
        except PWTimeout:
            continue
    return None


def put_draft(page, box, draft):
    """댓글 칸을 클릭해 초안을 넣는다. React 댓글 칸이 값을 못 받으면 키보드 입력으로 다시."""
    box.scroll_into_view_if_needed()
    box.click()
    try:
        box.fill(draft)
    except Exception:
        pass
    try:
        ok = draft[:10] in (box.input_value() if box.evaluate("e => 'value' in e") else box.inner_text())
    except Exception:
        ok = False
    if not ok:
        box.click()
        page.keyboard.press("Control+A")
        page.keyboard.insert_text(draft)


def article_bottom(page):
    """본문 끝(댓글 영역 시작) 위치. 못 찾으면 페이지 높이의 80%."""
    return page.evaluate("""() => {
        const c = document.querySelector('[data-tistory-react-app="Comment"], div[id$="Comment"], .comments, #comment');
        const y = c ? c.getBoundingClientRect().top + window.scrollY : document.body.scrollHeight * 0.8;
        return Math.max(0, y - window.innerHeight);
    }""")


def wait_reading(page, seconds):
    """글을 읽는 동안 본문 끝까지 천천히 스크롤하며 대기. 남은 시간을 명령창에 표시한다."""
    try:
        bottom = article_bottom(page)
    except Exception:
        bottom = 0
    steps = max(1, seconds // SCROLL_EVERY)
    for left in range(seconds, 0, -1):
        print(f"\r  글 읽는 중… 댓글 칸으로 가기까지 {left:2d}초 ", end="", flush=True)
        elapsed = seconds - left
        if elapsed % SCROLL_EVERY == 0 and bottom:
            # 지금까지 진행한 만큼의 위치로 부드럽게 이동 (직접 스크롤해도 따라잡기만 함)
            target = bottom * min(1, (elapsed // SCROLL_EVERY) / max(1, steps - 1))
            try:
                page.evaluate("y => { if (window.scrollY < y) window.scrollTo({top: y, behavior: 'smooth'}); }", target)
            except Exception:
                pass
        page.wait_for_timeout(1000)
    print("\r  읽기 시간 끝 — 댓글 칸으로 이동합니다.          ")


def start_keyboard():
    """명령창 입력을 따로 받아 둔다 — 기다리는 동안에도 등록 여부를 계속 확인하기 위해."""
    q = queue.Queue()

    def loop():
        while True:
            try:
                q.put(input().strip().lower())
            except EOFError:
                break
    threading.Thread(target=loop, daemon=True).start()
    return q


def logged_in(ctx):
    names = {c["name"] for c in ctx.cookies("https://www.tistory.com")}
    return bool(names & {"TSSESSION", "TSSESSION_KEEP"})


def comment_total(page, post):
    """글의 현재 댓글 수(티스토리 댓글 API). 알 수 없으면 None."""
    if not post.get("entry_id"):
        return None
    u = urlparse(post["url"])
    try:
        r = page.request.get(f"{u.scheme}://{u.netloc}/m/api/{post['entry_id']}/comment", timeout=8000)
        return r.json()["data"]["totalItems"]
    except Exception:
        return None


def wait_for_submit(page, post, keys):
    """사람이 [등록]을 누를 때까지 기다린다. 댓글 수가 늘면 다음 글로. Enter=건너뛰기, q=끝내기."""
    before = comment_total(page, post)
    if before is None:
        print("  → 고친 뒤 직접 [등록]을 누르고, 명령창에서 Enter (끝내기: q)")
    else:
        print("  → 고친 뒤 직접 [등록]을 누르면 자동으로 다음 글로 넘어갑니다. (건너뛰기: Enter · 끝내기: q)")
    while True:
        try:
            cmd = keys.get(timeout=3)
            return "quit" if cmd == "q" else "skip"
        except queue.Empty:
            pass
        if before is not None:
            now = comment_total(page, post)
            if now is not None and now > before:
                print("  등록 확인! 3초 뒤 다음 글로 넘어갑니다.")
                page.wait_for_timeout(3000)
                return "done"


def main():
    path, posts = latest_list()
    print(f"목록: {os.path.basename(path)} · {len(posts)}개\n")

    with sync_playwright() as p:
        ctx = launch(p)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()

        keys = start_keyboard()
        page.goto("https://www.tistory.com/")
        if logged_in(ctx):
            print("티스토리 로그인 확인됨 — 바로 시작합니다.")
        else:
            print("▶ 열린 크롬 창에서 티스토리에 로그인한 뒤, 명령창에서 Enter")
            keys.get()
        done = 0

        for i, post in enumerate(posts, 1):
            print(f"\n[{i}/{len(posts)}] {post['blog']} · {post['title']}")
            try:
                page.goto(post["url"], wait_until="domcontentloaded")
            except Exception as ex:
                print(f"  열기 실패 ({ex.__class__.__name__}) → 건너뜀")
                continue

            wait_reading(page, READ_SECONDS)
            draft = post["drafts"][0]
            box = find_comment_box(page)
            if box:
                put_draft(page, box, draft)
                print("  댓글 칸에 초안 1을 넣었습니다.")
            else:
                print("  댓글 칸을 못 찾았습니다. 아래 초안을 복사해 직접 붙여넣으세요.")
                print(f"  초안 1: {draft}")
            if len(post["drafts"]) > 1:
                print(f"  (다른 초안) {post['drafts'][1]}")

            result = wait_for_submit(page, post, keys)
            if result == "done":
                done += 1
            elif result == "quit":
                break

        print(f"\n끝났습니다. 댓글 {done}개 등록 확인. 브라우저를 닫습니다.")
        ctx.close()


if __name__ == "__main__":
    main()
