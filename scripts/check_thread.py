"""Periksa panjang tiap post di draf thread. Jalankan: uv run python scripts/check_thread.py"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Link dihitung 23 karakter di X; placeholder di sini dihitung apa adanya (lebih ketat).
for name, limit in (("thread_x.md", 280), ("thread_threads.md", 500)):
    text = (ROOT / "post" / name).read_text(encoding="utf-8")
    for block in text.split("\n---\n")[1:]:
        lines = block.strip().split("\n")
        body = "\n".join(lines[1:]).strip()
        flag = "  <-- TERLALU PANJANG" if len(body) > limit else ""
        print(f"{name} {lines[0].split()[0]:>5} {len(body):4d}/{limit}{flag}")
