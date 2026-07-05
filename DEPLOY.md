# 인터넷 공개 주소로 배포하기 (Render, 무료)

이 앱은 서버가 실시간 검색을 하므로 파이썬을 돌려주는 호스팅이 필요합니다.
가장 쉬운 무료 방법인 **Render**로 올려 `https://...onrender.com` 공개 주소를 만듭니다.

> ⚠️ 검색엔진이 클라우드 서버 IP를 막을 수 있어, 올린 뒤 결과가 잘 나오는지 테스트가 필요합니다.
> 안 나오면 "플랜 B"(내 PC를 공개주소로 여는 Cloudflare 터널)로 전환합니다.

## 1단계 — 코드를 GitHub에 올리기 (GitHub Desktop 추천)

1. https://desktop.github.com 에서 **GitHub Desktop** 설치 → GitHub 계정으로 로그인
   (계정 없으면 https://github.com 에서 무료 가입)
2. GitHub Desktop → **File → Add local repository** → 이 폴더 선택
   → "create a repository" 뜨면 **Create a repository** 클릭
3. **Publish repository** 클릭 (Private 체크해도 됨) → 업로드 완료

> `.gitignore`가 `api_key.txt`, `reports/`, `history.json` 등을 자동 제외하므로
> 개인 정보나 키는 올라가지 않습니다.

## 2단계 — Render에 배포

1. https://render.com 접속 → **Get Started** → **GitHub로 로그인**
2. **New +** → **Blueprint** 선택
3. 방금 올린 저장소(tistory-comment-helper) 선택 → **Connect**
   → 폴더의 `render.yaml`을 자동으로 읽어 설정이 채워집니다
4. **Apply / Deploy** 클릭 → 몇 분 기다리면 배포 완료
5. 상단에 생기는 **`https://tistory-helper-xxxx.onrender.com`** 주소가 공개 주소입니다
   → 폰, 다른 컴퓨터 어디서든 이 주소로 접속

## 3단계 — 테스트

- 공개 주소 접속 → 키워드 검색 → 글 10개 나오면 성공 🎉
- 무료 플랜 특성:
  - 15분간 아무도 안 쓰면 잠듦 → 다음 접속 시 깨는 데 ~50초
  - 검색 자체도 30~60초 → 첫 결과까지 최대 2분 정도 걸릴 수 있음
- 결과가 안 나오면(클라우드 IP 차단) → Claude에게 알려주면 플랜 B로 전환

## 플랜 B — Cloudflare 터널 (검색이 클라우드에서 막힐 때)

내 PC에서 서버를 돌리고, 그걸 공개 주소로 연결하는 방식.
검색이 확실히 되지만 **내 PC가 켜져 있어야** 접속됩니다. 필요 시 안내해 드립니다.
