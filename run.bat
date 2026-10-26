@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo [1/2] 최근 글 찾는 중...
py find_recent.py --no-open
if errorlevel 1 goto end
echo.
echo [2/2] 댓글 도우미 시작
py comment_assist.py
:end
pause
