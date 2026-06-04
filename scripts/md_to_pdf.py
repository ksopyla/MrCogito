"""Convert a markdown file to a styled PDF using a pure-Python pipeline.

Run via uv with isolated dependencies (does not pollute pyproject.toml):

    uv run --with "markdown-pdf>=1.5" --with "pygments>=2.18" --with "requests>=2.31" \\
        python scripts/md_to_pdf.py docs/sprind_frontier_ai/team_brief.md

The `markdown-pdf` package uses `fpdf2` under the hood — pure Python, no system
dependencies (no cairo, no Chromium, no LaTeX). Mermaid blocks (```mermaid ... ```)
are pre-rendered to PNG via the public kroki.io HTTP API and embedded as images;
this requires network access during conversion. If kroki is unreachable the
mermaid block is left as-is so the PDF still builds.
"""

from __future__ import annotations

import argparse
import re
import sys
from hashlib import sha1
from pathlib import Path

import requests  # type: ignore[import-not-found]
from markdown_pdf import MarkdownPdf, Section  # type: ignore[import-not-found]


CSS = """
body {
    font-family: Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #1f2937;
}

h1 {
    font-size: 18pt;
    color: #0f172a;
    margin-top: 14pt;
    margin-bottom: 8pt;
    page-break-after: avoid;
}

h2 {
    font-size: 13pt;
    color: #334155;
    margin-top: 14pt;
    margin-bottom: 5pt;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    color: #334155;
    margin-top: 10pt;
    margin-bottom: 4pt;
    page-break-after: avoid;
}

h4 {
    font-size: 10pt;
    color: #475569;
    margin-top: 8pt;
    margin-bottom: 3pt;
    page-break-after: avoid;
}

p {
    margin-top: 4pt;
    margin-bottom: 4pt;
    text-align: left;
}

a {
    color: #1d4ed8;
    text-decoration: none;
}

ul, ol {
    margin-top: 4pt;
    margin-bottom: 6pt;
    padding-left: 18pt;
}

li {
    margin-bottom: 1.5pt;
}

blockquote {
    padding: 4pt 9pt;
    margin: 6pt 0;
    background: #f3f4f6;
    color: #1f2937;
    font-style: italic;
}

code {
    font-family: "Courier New", Courier, monospace;
    font-size: 9pt;
    background: #f4f4f5;
    padding: 1px 3px;
    color: #334155;
}

pre {
    background: #f4f4f5;
    color: #1f2937;
    font-family: "Courier New", Courier, monospace;
    font-size: 8.4pt;
    line-height: 1.3;
    padding: 6pt 8pt;
    page-break-inside: avoid;
}

pre code {
    background: transparent;
    color: inherit;
    padding: 0;
    font-size: inherit;
}

img {
    max-width: 100%;
    page-break-inside: avoid;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 6pt 0;
    page-break-inside: avoid;
    font-size: 9pt;
}

th, td {
    border: 1px solid #e5e7eb;
    padding: 4pt 6pt;
    vertical-align: top;
    text-align: left;
}

th {
    background: #f3f4f6;
    color: #1f2937;
    font-weight: bold;
}

hr {
    border: 0;
    border-top: 1px solid #e5e7eb;
    margin: 10pt 0;
}
"""


MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)\n```", re.DOTALL)


def render_mermaid_to_png(source: str, out_path: Path, theme: str = "default") -> bool:
    """POST mermaid source to kroki.io and write PNG to out_path. Returns True on success."""
    try:
        # kroki.io accepts raw POST body for the diagram source.
        response = requests.post(
            f"https://kroki.io/mermaid/png",
            data=source.encode("utf-8"),
            headers={"Content-Type": "text/plain"},
            timeout=30,
        )
        response.raise_for_status()
        out_path.write_bytes(response.content)
        return True
    except Exception as exc:  # noqa: BLE001 - we want any failure to fall through
        print(f"  ! mermaid render failed via kroki.io: {exc}", file=sys.stderr)
        return False


def preprocess_mermaid(md_text: str, md_path: Path) -> str:
    """Replace ```mermaid``` blocks with PNG image references rendered via kroki.io.

    Generated images go to ``<md_dir>/_assets/`` next to the source markdown.
    Block-to-image mapping is keyed by SHA1 of the source so re-renders are
    cached and stable across runs.
    """
    blocks = list(MERMAID_RE.finditer(md_text))
    if not blocks:
        return md_text

    assets_dir = md_path.parent / "_assets"
    assets_dir.mkdir(exist_ok=True)

    print(f"Found {len(blocks)} mermaid block(s); rendering via kroki.io ...")

    # Replace from the end so spans stay valid as we rewrite.
    out = md_text
    for match in reversed(blocks):
        source = match.group(1).strip()
        digest = sha1(source.encode("utf-8")).hexdigest()[:10]
        png_path = assets_dir / f"mermaid_{digest}.png"

        if not png_path.exists():
            ok = render_mermaid_to_png(source, png_path)
            if not ok:
                # Leave the block as-is so the PDF still builds.
                continue
            print(f"  ✓ {png_path.relative_to(md_path.parent)}")
        else:
            print(f"  ↺ cached {png_path.relative_to(md_path.parent)}")

        # Reference the image with a relative path so markdown-pdf can find it.
        rel = png_path.relative_to(md_path.parent).as_posix()
        replacement = f"\n![diagram]({rel})\n"
        out = out[: match.start()] + replacement + out[match.end():]

    return out


def convert(md_path: Path, pdf_path: Path, title: str | None = None) -> None:
    md_text = md_path.read_text(encoding="utf-8")
    md_text = preprocess_mermaid(md_text, md_path)

    pdf = MarkdownPdf(toc_level=0, optimize=True)
    pdf.meta["title"] = title or md_path.stem.replace("_", " ").title()
    pdf.meta["author"] = "Krzysztof Sopyla"
    pdf.meta["creator"] = "scripts/md_to_pdf.py"
    pdf.add_section(Section(md_text, toc=False, root=str(md_path.parent)), user_css=CSS)

    pdf.save(str(pdf_path))
    size_kb = pdf_path.stat().st_size / 1024
    print(f"Wrote {pdf_path} ({size_kb:.1f} KB)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input markdown file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PDF path (defaults to input with .pdf extension)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Document title metadata (defaults to input filename)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    pdf_path = args.output or args.input.with_suffix(".pdf")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    convert(args.input, pdf_path, title=args.title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
