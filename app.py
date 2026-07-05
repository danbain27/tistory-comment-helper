# -*- coding: utf-8 -*-
"""
티스토리 교류 도우미 — 로컬 웹앱

브라우저에서 키워드를 검색하면, 조건에 맞는 서로 다른 블로그 글 10개를 추천한다.
(광고/제휴글·내 블로그·주제 무관·오래된 글 자동 제외)
댓글 자동 등록은 없음 — 추천 결과를 Claude에게 주면 댓글 초안을 만들어 준다.

실행:  python app.py   (또는 "앱 실행.bat" 더블클릭)
"""

import os
import html
import datetime
import threading
import webbrowser

from flask import Flask, request

import collect   # 수집 로직 재사용

app = Flask(__name__)

PAGE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>티스토리 교류 도우미</title>
<style>
  :root{--bg:#f5f6f8;--card:#fff;--line:#e6e8ec;--title:#2f6fed;--text:#23262b;
        --muted:#8a9099;--box:#f3f6ff;--btn:#2f6fed;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);line-height:1.6;
       font-family:"Pretendard","Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;
       padding:26px 16px 60px;}
  .wrap{max-width:780px;margin:0 auto}
  h1{font-size:21px;margin:0 0 4px}
  .sub{color:var(--muted);font-size:13px;margin:0 0 18px}
  form.search{background:var(--card);border:1px solid var(--line);border-radius:14px;
              padding:16px 18px;margin-bottom:20px}
  label{font-size:13px;color:#555;display:block;margin:0 0 5px}
  input[type=text]{width:100%;font:inherit;font-size:14px;padding:10px 12px;
                   border:1px solid var(--line);border-radius:9px}
  .rowline{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:12px}
  .rowline .cnt{width:90px}
  .chk{font-size:13px;color:#555;display:flex;align-items:center;gap:6px}
  button{font:inherit;font-size:14px;border:1px solid var(--line);background:#fff;
         color:var(--text);border-radius:9px;padding:9px 16px;cursor:pointer}
  button.primary{background:var(--btn);color:#fff;border-color:var(--btn);font-weight:600}
  button:active{transform:translateY(1px)}
  .note{font-size:12.5px;color:var(--muted);margin:6px 0 0}
  .summary{font-size:13.5px;color:#555;margin:4px 0 14px}
  .summary b{color:var(--text)}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;
        padding:16px 18px;margin-bottom:14px}
  .card h2{font-size:16px;margin:0 0 4px;line-height:1.45}
  .card h2 .num{color:var(--muted);margin-right:6px}
  .card h2 a{color:var(--title);text-decoration:none}
  .card h2 a:hover{text-decoration:underline}
  .meta{font-size:12px;color:var(--muted);margin-bottom:9px}
  .meta .tag{display:inline-block;background:#eef1f5;color:#5b6169;border-radius:6px;
             padding:1px 7px;margin-right:6px}
  .excerpt{font-size:13.5px;color:#3d434b;background:var(--box);border-radius:9px;
           padding:10px 13px;max-height:96px;overflow:hidden}
  .clabel{font-size:12px;color:var(--muted);margin:2px 0 5px}
  textarea.comment{width:100%;border:1px solid #d5e0fb;background:var(--box);border-radius:9px;
    padding:11px 13px;font:inherit;font-size:14px;color:var(--text);resize:vertical;
    min-height:70px;line-height:1.6}
  textarea.comment:focus{outline:2px solid #b9cdfb}
  .cardbtns{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}
  .warn{background:#fff4e2;border:1px solid #f2d59a;color:#8a5a00;border-radius:11px;
    padding:12px 15px;font-size:13px;margin-bottom:16px;line-height:1.6}
  .warn code{background:#fbe7c2;padding:1px 6px;border-radius:5px}
  .empty{background:#fff;border:1px dashed var(--line);border-radius:12px;
         padding:26px;text-align:center;color:var(--muted)}
  #overlay{position:fixed;inset:0;background:rgba(245,246,248,.92);display:none;
           align-items:center;justify-content:center;flex-direction:column;z-index:9}
  #overlay .spin{width:34px;height:34px;border:4px solid #d5e0fb;border-top-color:var(--btn);
                 border-radius:50%;animation:sp 1s linear infinite}
  @keyframes sp{to{transform:rotate(360deg)}}
  .bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:2px 0 16px}
  .barnote{font-size:12.5px;color:var(--muted)}
</style></head><body>
<div id="overlay"><div class="spin"></div>
  <p style="margin-top:14px;color:#555">글을 찾는 중… 30초~1분 걸려요</p></div>
<div class="wrap">
  <h1>티스토리 교류 도우미</h1>
  <p class="sub">키워드를 검색하면 비슷한 블로그 글 __TARGET__개를 추천합니다 · 광고·내 블로그·오래된 글 자동 제외 · 댓글 등록은 직접</p>

  <form class="search" method="post" action="/search" onsubmit="document.getElementById('overlay').style.display='flex'">
    <label>검색 키워드 (쉼표로 구분)</label>
    <input type="text" name="keywords" value="__KEYWORDS__" placeholder="예: 건프라 리뷰, 슈퍼로봇대전 공략, 게임 리뷰">
    <div class="rowline">
      <div>
        <label>추천 개수</label>
        <input class="cnt" type="text" name="count" value="__COUNT__">
      </div>
      <label class="chk"><input type="checkbox" name="fresh" __FRESH__> 매일 새 글 모드 (이전에 찾은 글 제외)</label>
      <button class="primary" type="submit">검색</button>
    </div>
    <p class="note">키워드는 collect.py 없이 여기서 바꿔도 됩니다. 여러 개 넣으면 골고루 섞어서 찾아요.</p>
  </form>

  __WARNING__
  __RESULTS__
</div>
</body></html>"""


def render(keywords_value, count, fresh, results_html, warning=""):
    return (
        PAGE.replace("__TARGET__", str(count))
        .replace("__KEYWORDS__", html.escape(keywords_value, quote=True))
        .replace("__COUNT__", str(count))
        .replace("__FRESH__", "checked" if fresh else "")
        .replace("__WARNING__", warning)
        .replace("__RESULTS__", results_html)
    )


def default_keywords_value():
    return ", ".join(collect.KEYWORDS)


@app.route("/")
def index():
    return render(default_keywords_value(), collect.TARGET_COUNT, True, "")


@app.route("/search", methods=["GET", "POST"])
def search():
    if request.method == "GET":
        # 주소창에 /search 를 직접 친 경우 → 홈으로
        return index()
    raw = request.form.get("keywords", "").strip()
    keywords = [k.strip() for k in raw.replace("\n", ",").split(",") if k.strip()]
    if not keywords:
        keywords = collect.KEYWORDS

    try:
        count = max(1, min(20, int(request.form.get("count", "10"))))
    except ValueError:
        count = 10
    fresh = request.form.get("fresh") is not None

    history = collect.load_history()
    exclude = history["urls"] if fresh else []

    posts = collect.collect_posts(keywords, count, exclude_urls=exclude, log=print)

    # 매일 새 글 모드면 이번 결과를 기록에 저장 → 다음엔 제외
    if fresh and posts:
        history["urls"] = list(set(history["urls"]) | {p["url"] for p in posts})
        collect.save_history(history)

    # 리포트 파일도 저장 (원하면 Claude에게 파일로 줄 수 있게)
    saved_path = ""
    if posts:
        os.makedirs(collect.REPORTS_DIR, exist_ok=True)
        today = datetime.date.today().isoformat()
        n = 1
        while True:
            saved_path = os.path.join(collect.REPORTS_DIR, f"{today}_{n}.md")
            if not os.path.exists(saved_path):
                break
            n += 1
        collect.write_report(saved_path, posts, today)

    return render(html.escape(", ".join(keywords), quote=True), count, fresh,
                  results_html(posts, saved_path))


def results_html(posts, saved_path):
    if not posts:
        return ('<div class="empty">조건에 맞는 글을 못 찾았어요.<br>'
                '키워드를 바꾸거나 “매일 새 글 모드”를 꺼보세요.</div>')

    parts = [f'<div class="summary">추천 <b>{len(posts)}</b>개 · 서로 다른 블로그 · '
             f'리포트: <code>{html.escape(os.path.basename(saved_path))}</code></div>']
    parts.append('<div class="bar">'
                 '<button class="primary" onclick="copyAll()">📋 결과 전체 복사 → Claude 채팅에 붙여넣기</button>'
                 '<span class="barnote">붙여넣으면 글마다 댓글 초안을 써드려요</span>'
                 '</div>')

    blob_lines = []
    for i, p in enumerate(posts, 1):
        title = html.escape(p["title"])
        url = html.escape(p["url"], quote=True)
        host = html.escape(p["host"])
        date = html.escape(p["date"])
        kw = html.escape(p["keyword"])
        short = html.escape(" ".join(p["text"][:300].split()))
        parts.append(f"""
  <div class="card">
    <h2><span class="num">{i}.</span> <a href="{url}" target="_blank" rel="noopener">{title}</a></h2>
    <div class="meta"><span class="tag">{host}</span> {date} · {kw}</div>
    <div class="excerpt">{short} …</div>
    <div class="cardbtns"><button onclick="window.open('{url}','_blank')">블로그 열기</button></div>
  </div>""")
        body = " ".join(p["text"][:700].split())
        blob_lines.append(f"{i}. {p['title']}\n   {p['url']}\n   ({p['date']} · {p['keyword']})\n   {body}\n")

    blob = html.escape("\n".join(blob_lines), quote=True)
    parts.append(f'<textarea id="blob" style="position:absolute;left:-9999px">{blob}</textarea>')
    parts.append("""<script>
      function copyAll(){
        const t=document.getElementById('blob');
        navigator.clipboard.writeText(t.value).then(()=>{
          const b=document.querySelector('.bar button');
          const o=b.textContent; b.textContent='복사됨 ✓ 이제 Claude 채팅에 붙여넣으세요';
          setTimeout(()=>b.textContent=o,2200);
        });
      }
    </script>""")
    return "\n".join(parts)


def open_browser(port):
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    # PORT 환경변수가 있으면 클라우드(공개 서버), 없으면 내 PC(로컬)
    port = int(os.environ.get("PORT", "5000"))
    is_cloud = "PORT" in os.environ
    if not is_cloud:
        threading.Timer(1.2, open_browser, args=(port,)).start()
        print(f"웹앱 실행 중 → http://127.0.0.1:{port}  (종료: Ctrl+C)")
    app.run(host="0.0.0.0", port=port, debug=False)
