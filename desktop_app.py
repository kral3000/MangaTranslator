from __future__ import annotations

import base64
import ctypes
import html
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import webview

from runtime_logging import RUNTIME_LOG_PATH_ENV, redact_sensitive


APP_NAME = "MangaTranslatorKo"
WINDOW_TITLE = "만화 번역기"
PREFERRED_PORT = 7676
STARTUP_TIMEOUT_SECONDS = 180
CREATE_NO_WINDOW = 0x08000000
ERROR_ALREADY_EXISTS = 183

_MUTEX_HANDLE: int | None = None


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _message_box(message: str, title: str = WINDOW_TITLE) -> None:
    ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)


def _acquire_single_instance() -> bool:
    global _MUTEX_HANDLE
    handle = ctypes.windll.kernel32.CreateMutexW(
        None, False, "Local\\MangaTranslatorKoDesktop"
    )
    _MUTEX_HANDLE = handle
    return bool(handle) and ctypes.windll.kernel32.GetLastError() != ERROR_ALREADY_EXISTS


def _find_available_port(preferred: int = PREFERRED_PORT) -> int:
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("사용 가능한 로컬 포트를 찾지 못했습니다.")


def _loading_html(icon_path: Path) -> str:
    try:
        icon = base64.b64encode(icon_path.read_bytes()).decode("ascii")
    except OSError:
        wordmark_icon = ""
    else:
        wordmark_icon = f'<img src="data:image/x-icon;base64,{icon}" width="52" height="52" alt="">'
    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>만화 번역기</title>
<style>
html,body { height: 100%; margin: 0; }
body {
    display: grid; place-items: center;
    background: linear-gradient(125deg, #20202a, #191922 46%, #15161c);
    color: #f2eff9; font-family: "Malgun Gothic", "Segoe UI", sans-serif;
}
main { display: grid; justify-items: center; gap: 20px; padding: 12px; }
.wordmark, .startup-status { display: flex; align-items: center; justify-content: center; }
.wordmark { gap: 18px; }
.wordmark img { width: clamp(34px, 5vw, 52px); height: auto; }
h1 { font-size: clamp(24px, 4vw, 38px); margin: 0; letter-spacing: -.025em; }
.startup-status { gap: 10px; margin: 0; color: #bfb9d0; font-size: 17px; }
.glyph {
    position: relative; flex: none; width: 22px; height: 22px; box-sizing: border-box;
    border: 1px solid #8077aa; border-radius: 4px; overflow: hidden;
    color: #cbbfff; font: 650 12px/1 "Malgun Gothic", "Yu Gothic UI", sans-serif;
}
.glyph b { display: grid; place-items: center; position: absolute; inset: 0; }
.glyph b:first-child { animation: glyph-source 2.35s ease-in-out infinite; }
.glyph b:last-child { opacity: 0; animation: glyph-target 2.35s ease-in-out infinite; }
@keyframes glyph-source {
    0%, 22%, 100% { opacity: 1; transform: translateY(0); }
    42%, 83% { opacity: 0; transform: translateY(-4px); }
    84% { opacity: 0; transform: translateY(4px); }
}
@keyframes glyph-target {
    0%, 24% { opacity: 0; transform: translateY(4px); }
    44%, 72% { opacity: 1; transform: translateY(0); }
    92%, 100% { opacity: 0; transform: translateY(-4px); }
}
@media (prefers-reduced-motion: reduce) { .glyph b { animation: none !important; } }
</style></head><body><main>
<div class="wordmark">__APP_ICON__<h1>만화 번역기</h1></div>
<p class="startup-status" role="status">
<span class="glyph" aria-hidden="true"><b lang="ja">あ</b><b>가</b></span>번역기를 시작하는 중…
</p></main></body></html>""".replace("__APP_ICON__", wordmark_icon)


def _error_html(message: str, log_path: Path) -> str:
    safe_message = html.escape(message)
    safe_path = html.escape(str(log_path))
    return f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<style>html,body{{height:100%;margin:0;background:#111113;color:#f4f4f5;font-family:"Segoe UI","Malgun Gothic",sans-serif}}
body{{display:grid;place-items:center}}main{{width:min(620px,calc(100% - 64px))}}h1{{font-size:22px;margin:0 0 12px}}
p{{color:#d4d4d8;line-height:1.65}}code{{display:block;margin-top:20px;padding:12px;border:1px solid #3f3f46;border-radius:6px;color:#fbbf24;word-break:break-all}}</style>
</head><body><main><h1>앱을 시작하지 못했습니다</h1><p>{safe_message}</p><code>{safe_path}</code></main></body></html>"""


class BackendProcess:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.process: subprocess.Popen[str] | None = None
        self.log_path = (
            root
            / "output"
            / "logs"
            / f"MangaTranslatorKo_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        self._write_lock = threading.Lock()

    def write_log(self, text: str) -> None:
        safe_text = redact_sensitive(text.rstrip("\r\n"))
        with self._write_lock:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(safe_text + "\n")

    def start(self, port: int) -> None:
        runtime_python = self.root / "runtime" / "python.exe"
        app_script = self.root / "app.py"
        if not runtime_python.is_file() or not app_script.is_file():
            raise FileNotFoundError(
                "runtime\\python.exe 또는 app.py를 찾지 못했습니다. "
                "MangaTranslatorKo.exe를 번역기 폴더 안에 두세요."
            )

        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONUNBUFFERED": "1",
                "SDNQ_USE_TORCH_COMPILE": "0",
                RUNTIME_LOG_PATH_ENV: str(self.log_path),
            }
        )
        self.write_log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {APP_NAME} 시작")
        self.process = subprocess.Popen(
            [str(runtime_python), "-u", str(app_script), "--port", str(port)],
            cwd=self.root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._drain_output, daemon=True).start()

    def _drain_output(self) -> None:
        process = self.process
        if not process or not process.stdout:
            return
        for line in process.stdout:
            self.write_log(line)

    def wait_until_ready(self, url: str) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise RuntimeError(
                    f"백엔드가 시작 중 종료되었습니다. 종료 코드: {self.process.returncode}"
                )
            try:
                with urllib.request.urlopen(url, timeout=1.0) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.25)
        raise TimeoutError("백엔드 시작 시간이 3분을 초과했습니다.")

    def stop(self) -> None:
        process = self.process
        if not process or process.poll() is not None:
            return
        self.write_log(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] 앱 종료")
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)


def main() -> int:
    if not _acquire_single_instance():
        _message_box("만화 번역기가 이미 실행 중입니다.")
        return 1

    root = _app_root()
    icon_path = root / "assets" / "alya-kujo-app-icon-v3.ico"
    backend = BackendProcess(root)
    window_closed = threading.Event()
    port = _find_available_port()
    url = f"http://127.0.0.1:{port}/"

    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = True
    window = webview.create_window(
        WINDOW_TITLE,
        html=_loading_html(icon_path),
        width=1440,
        height=900,
        min_size=(1100, 680),
        resizable=True,
        background_color="#191922",
        text_select=True,
        zoomable=True,
    )

    def handle_window_closed() -> None:
        window_closed.set()
        backend.stop()

    window.events.closed += handle_window_closed

    def boot() -> None:
        try:
            window.show()
            backend.start(port)
            backend.wait_until_ready(url)
            if window_closed.is_set():
                return
            window.load_url(url)
        except Exception as exc:
            backend.write_log(f"시작 오류: {exc}")
            if not window_closed.is_set():
                window.load_html(_error_html(str(exc), backend.log_path))

    try:
        webview.start(
            boot,
            gui="edgechromium",
            debug=False,
            private_mode=False,
            storage_path=str(root / "output" / "webview2-desktop"),
            icon=str(icon_path) if icon_path.is_file() else None,
        )
    except Exception as exc:
        backend.write_log(f"WebView2 오류: {exc}")
        _message_box(
            "앱 창을 열 수 없습니다. Microsoft Edge WebView2 Runtime을 확인하세요.\n\n"
            f"{exc}\n\n로그: {backend.log_path}"
        )
        return 1
    finally:
        backend.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
