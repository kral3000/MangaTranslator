from __future__ import annotations

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


def _loading_html() -> str:
    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>만화 번역기</title>
<style>
html,body{height:100%;margin:0;background:#111113;color:#f4f4f5;font-family:"Segoe UI","Malgun Gothic",sans-serif}
body{display:grid;place-items:center}.shell{width:min(460px,calc(100% - 64px))}
h1{font-size:24px;margin:0 0 10px;font-weight:650}p{margin:0 0 24px;color:#a1a1aa;line-height:1.6}
.track{height:4px;overflow:hidden;border-radius:4px;background:#27272a}.bar{width:38%;height:100%;background:#f97316;animation:load 1.25s ease-in-out infinite}
@keyframes load{0%{transform:translateX(-110%)}100%{transform:translateX(365%)}}
</style></head><body><main class="shell"><h1>만화 번역기</h1>
<p>모델과 설정을 준비하고 있습니다. 잠시만 기다려 주세요.</p><div class="track"><div class="bar"></div></div>
</main></body></html>"""


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
        html=_loading_html(),
        width=1440,
        height=900,
        min_size=(1100, 680),
        resizable=True,
        background_color="#111113",
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
