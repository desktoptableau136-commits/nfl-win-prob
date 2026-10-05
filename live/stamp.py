"""Stamps every script import in docs/ with a version, so browsers never mix new and old files.

    .venv\\Scripts\\python -m live.stamp

Run before each push that changes docs/js. GitHub Pages lets browsers cache files for 10 minutes;
without a stamp, a fresh index.html could load a cached older module that lacks a function it
imports, and the page stops at "Loading…". With ?v=<hash of all the scripts>, any change to any
script gives every import a new address, so the whole set is fetched fresh together.
"""
import hashlib
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
IMPORT = re.compile(r'''(from\s+["'](?:\./|\./js/)[\w-]+\.js)(?:\?v=\w+)?(["'])''')


def main():
    files = [DOCS / "index.html", *sorted((DOCS / "js").glob("*.js"))]
    strip = lambda text: IMPORT.sub(r"\1\2", text)
    digest = hashlib.sha1()
    for f in sorted((DOCS / "js").glob("*.js")):
        digest.update(strip(f.read_text(encoding="utf-8")).encode())
    version = digest.hexdigest()[:8]
    for f in files:
        text = f.read_text(encoding="utf-8")
        new = IMPORT.sub(rf"\1?v={version}\2", strip(text))
        if new != text:
            f.write_text(new, encoding="utf-8", newline="\n")
    print(f"Scripts stamped v={version}")


if __name__ == "__main__":
    main()
