"""Start Streamlit locally, check HTTP health, and always stop the subprocess."""

from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    """Probe a real headless server without opening a browser."""
    root = Path(__file__).resolve().parents[1]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = [sys.executable, "-m", "streamlit", "run", str(root / "app.py"),
               "--server.headless=true", "--server.address=127.0.0.1",
               f"--server.port={port}", "--browser.gatherUsageStats=false"]
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError("Streamlit exited before becoming healthy.")
                try:
                    with urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as response:
                        if response.status == 200 and response.read().strip() == b"ok":
                            break
                except (URLError, TimeoutError):
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("Streamlit did not become healthy within 30 seconds.")
            with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                assert response.status == 200
                assert b"streamlit" in response.read().lower()
            print("PASS: Streamlit health and root page returned HTTP 200.")
            print("App execution and widget interactions are checked separately by pytest AppTest.")
        except Exception:
            log.seek(0)
            print(log.read(), file=sys.stderr)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
