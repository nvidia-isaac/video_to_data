"""Build the static exercise download page from the ZIP files beside it."""

import argparse
import html
import re
import zipfile
from pathlib import Path
from urllib.parse import quote


def build_index(directory: Path) -> str:
    archives = sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".zip"),
        key=lambda path: [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)],
    )
    links = []
    for archive in archives:
        if not zipfile.is_zipfile(archive):
            raise ValueError(f"{archive.name} is not a ZIP archive; fetch Git LFS files before staging the site")
        filename = html.escape(archive.name, quote=True)
        href = html.escape(quote(archive.name, safe=""), quote=True)
        size = archive.stat().st_size / 1_000_000
        links.append(
            f'      <li><a href="{href}" download="{filename}">{filename}</a>'
            f'<span>{size:.1f} MB · ZIP</span></li>'
        )
    downloads = "\n".join(links) if links else '      <li>Exercises will be added soon.</li>'
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tutorial Exercises — V2D CoRL Tutorial</title>
  <link rel="icon" type="image/svg+xml" href="../assets/favicon.svg">
  <link rel="stylesheet" href="../_ds/kaizen-ui-foundations-5919e993-0c25-47fe-9d3b-b24c0049b4d8/fonts/fonts.css">
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; background: #faf9f5; color: #1a1a18; font-family: "NVIDIA Sans", Arial, sans-serif; }
    main { max-width: 900px; margin: 0 auto; padding: 64px 24px; }
    a { color: inherit; text-underline-offset: 4px; }
    a:hover { color: #527f00; }
    h1 { font-size: clamp(32px, 6vw, 56px); font-weight: 500; letter-spacing: -0.03em; margin: 40px 0 20px; }
    p { color: #575750; font-size: 18px; line-height: 1.6; }
    ul { list-style: none; padding: 0; margin: 36px 0 0; }
    li { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px 24px; padding: 24px 0; border-top: 1px solid #d9d9ce; }
    li a { font-size: 20px; font-weight: 600; overflow-wrap: anywhere; }
    li span { color: #575750; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <a href="../">← Back to tutorial</a>
    <h1>Tutorial Exercises</h1>
    <p>Download and extract an exercise ZIP to get started.</p>
    <ul aria-label="Exercise downloads">
""" + downloads + """
    </ul>
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing exercise ZIPs")
    args = parser.parse_args()
    (args.directory / "index.html").write_text(build_index(args.directory), encoding="utf-8")


if __name__ == "__main__":
    main()
