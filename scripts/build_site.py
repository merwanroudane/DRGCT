#!/usr/bin/env python
"""Build the static documentation site served by GitHub Pages from ``/docs``.

Sources
-------
* ``site/pages/*.md``  -- hand-written pages (overview, the three result
  chapters).  They may embed live results with two directives:

      {{table: results/macro/tables/x.csv | caption text }}
      {{figure: macro_fig2_pvalue_heatmap.png | caption text }}

  so every number and every plot on the site comes from the committed run
  output rather than being retyped.
* ``docs/*.md``        -- the repository documentation, converted as-is.

Output
------
``docs/`` gets ``index.html`` and one page per source, plus
``docs/assets/`` (stylesheet and every result figure as PNG) and a
``.nojekyll`` marker so GitHub Pages serves the files verbatim.

The existing ``docs/*.md`` files are left in place: GitHub renders them for
people browsing the repository, the HTML serves the website.

Usage
-----
    python scripts/build_site.py
    python scripts/build_site.py --serve      # build then preview on :8000
"""

from __future__ import annotations

import argparse
import html
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
OUT = ROOT / "docs"
ASSETS = OUT / "assets"

AUTHOR = "Dr Merwan Roudane"
AUTHOR_EMAIL = "merwanroudane920@gmail.com"
GITHUB = "https://github.com/merwanroudane/DRGCT"
GITHUB_USER = "https://github.com/merwanroudane"
PYPI = "https://pypi.org/project/drgct/"
PAPER = "https://arxiv.org/abs/2509.15798"
SITE_URL = "https://merwanroudane.github.io/DRGCT/"

#: (output file, nav title, source, nav group)
PAGES = [
    ("index.html",      "Overview",                 SITE / "pages/index.md",      "Start"),
    ("guide.html",      "Applied guide",            ROOT / "docs/GUIDE.md",       "Start"),
    ("theory.html",     "Theory",                   ROOT / "docs/THEORY.md",      "Method"),
    ("api.html",        "API reference",            ROOT / "docs/SYNTAX.md",      "Method"),
    ("faq.html",        "FAQ",                      ROOT / "docs/FAQ.md",         "Method"),
    ("macro.html",      "US macro application",     SITE / "pages/macro.md",      "Results"),
    ("finance.html",    "Price–volume application", SITE / "pages/finance.md",    "Results"),
    ("simulation.html", "Simulation evidence",      SITE / "pages/simulation.md", "Results"),
]

#: Rewrite in-repository markdown links to their site equivalents.
LINK_MAP = {
    "GUIDE.md": "guide.html",
    "SYNTAX.md": "api.html",
    "THEORY.md": "theory.html",
    "FAQ.md": "faq.html",
    "docs/GUIDE.md": "guide.html",
    "docs/SYNTAX.md": "api.html",
    "docs/THEORY.md": "theory.html",
    "docs/FAQ.md": "faq.html",
    "../results/README.md": "simulation.html",
    "results/README.md": "simulation.html",
    "../README.md": "index.html",
    "README.md": "index.html",
    "../examples": f"{GITHUB}/tree/main/examples",
    "examples": f"{GITHUB}/tree/main/examples",
    "LICENSE": f"{GITHUB}/blob/main/LICENSE",
}

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — drgct</title>
<meta name="description" content="{description}">
<meta name="author" content="{author}">
<link rel="canonical" href="{site_url}{slug}">
<meta property="og:type" content="website">
<meta property="og:title" content="{title} — drgct">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{site_url}{slug}">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='18' fill='%231F4E79'/><text x='50' y='70' font-size='58' font-family='Georgia,serif' fill='white' text-anchor='middle'>D</text></svg>">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="layout">

<aside class="sidebar" id="sidebar">
  <a class="brand" href="index.html">
    <span class="mark">drgct</span>
    <span class="tagline">Deep-learning based doubly robust test<br>for Granger causality</span>
  </a>
  <button class="navtoggle" onclick="document.getElementById('sidebar').classList.toggle('open')">Menu</button>
  <nav>
{nav}
  </nav>
  <div class="ext">
    <a href="{github}">GitHub repository</a>
    <a href="{pypi}">PyPI package</a>
    <a href="{paper}">The paper (arXiv)</a>
  </div>
  <div class="byline">
    Built and maintained by<br>
    <strong>{author}</strong><br>
    <a href="mailto:{email}">{email}</a><br>
    <a href="{github_user}">github.com/merwanroudane</a>
  </div>
</aside>

<main class="main">
  <div class="content md-body">
{body}
    <div class="pagefoot">
      <div>
        <strong>drgct {version}</strong> · MIT licence · © 2026 {author}
      </div>
      <div>
        <a href="{github}">Source</a> ·
        <a href="{pypi}">PyPI</a> ·
        <a href="{paper}">Paper</a> ·
        <a href="{github}/issues">Report an issue</a>
      </div>
    </div>
  </div>
</main>

{toc}

</div>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Directives
# --------------------------------------------------------------------------- #
def inline_md(text: str) -> str:
    """Convert the inline markdown of a caption (bold, italic, code, links).

    Captions are emitted as raw HTML by the directives below, and the markdown
    converter does not descend into raw HTML blocks -- so they have to be
    converted here or ``**Figure 1.**`` reaches the page verbatim.
    """
    import markdown as _md

    if not text:
        return ""
    out = _md.markdown(text, extensions=["attr_list"]).strip()
    if out.startswith("<p>") and out.endswith("</p>"):
        out = out[3:-4]
    return rewrite_links(out)


def render_table(spec: str) -> str:
    """``{{table: path.csv | caption}}`` -> an HTML table with a caption."""
    import pandas as pd

    path, _, caption = (s.strip() for s in spec.partition("|"))
    f = ROOT / path
    if not f.exists():
        return f'<div class="warn"><strong>missing</strong><p>{html.escape(path)} '
        f'has not been generated yet.</p></div>'
    df = pd.read_csv(f)
    df.columns = [str(c) for c in df.columns]
    body = df.to_html(index=False, escape=False, border=0, na_rep="")
    body = body.replace(' class="dataframe"', "")
    body = body.replace("✓", '<span class="tick">✓</span>')
    body = body.replace("✗", '<span class="cross">✗</span>')
    cap = f"<figcaption>{inline_md(caption)}</figcaption>" if caption else ""
    return f'<figure><div class="table-wrap">{body}</div>{cap}</figure>'


def render_figure(spec: str) -> str:
    """``{{figure: name.png | caption}}`` -> a captioned figure."""
    name, _, caption = (s.strip() for s in spec.partition("|"))
    cap = f"<figcaption>{inline_md(caption)}</figcaption>" if caption else ""
    alt = html.escape(re.sub(r"[*_`<>]|\[|\]\([^)]*\)", "",
                             caption.split(".")[0] if caption else name))
    return (f'<figure><img loading="lazy" src="assets/figures/{name}" alt="{alt}">'
            f"{cap}</figure>")


DIRECTIVES = {"table": render_table, "figure": render_figure}


def apply_directives(text: str) -> str:
    def sub(m):
        kind, spec = m.group(1).lower(), m.group(2)
        return DIRECTIVES[kind](spec) if kind in DIRECTIVES else m.group(0)

    return re.sub(r"\{\{\s*(table|figure)\s*:\s*(.+?)\s*\}\}", sub, text, flags=re.S)


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #
def rewrite_links(html_text: str) -> str:
    def sub(m):
        target = m.group(1)
        if target in LINK_MAP:
            return f'href="{LINK_MAP[target]}"'
        path, _, anchor = target.partition("#")
        if path in LINK_MAP:
            return f'href="{LINK_MAP[path]}{"#" + anchor if anchor else ""}"'
        # Anything else that stays inside the repository points at GitHub.
        if not target.startswith(("http", "mailto:", "#", "assets/")) and not target.endswith(".html"):
            return f'href="{GITHUB}/blob/main/{target}"'
        return m.group(0)

    return re.sub(r'href="([^"]+)"', sub, html_text)


def convert(md_text: str):
    """Markdown -> (html, [(level, id, title)]) for the on-page TOC."""
    import markdown

    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "codehilite", "toc", "attr_list",
                    "sane_lists", "md_in_html"],
        extension_configs={"codehilite": {"guess_lang": False},
                           "toc": {"permalink": False}},
    )
    body = md.convert(md_text)
    body = rewrite_links(body)
    body = re.sub(r"(<table>)", r'<div class="table-wrap">\1', body)
    body = re.sub(r"(</table>)", r"\1</div>", body)
    # The directive renderer already wraps its own tables.
    body = body.replace('<div class="table-wrap"><div class="table-wrap">',
                        '<div class="table-wrap">')
    toc = [(int(t["level"]), t["id"], t["name"])
           for t in getattr(md, "toc_tokens", []) for t in [t, *t.get("children", [])]]
    return body, toc


def build_nav(current: str) -> str:
    out, group = [], None
    for slug, title, _src, grp in PAGES:
        if grp != group:
            out.append(f'    <div class="group">{grp}</div>')
            group = grp
        cls = ' class="active"' if slug == current else ""
        out.append(f'    <a href="{slug}"{cls}>{title}</a>')
    return "\n".join(out)


def build_toc(body_html: str) -> str:
    heads = re.findall(r'<h([23]) id="([^"]+)">(.*?)</h[23]>', body_html, flags=re.S)
    if len(heads) < 3:
        return ""
    items = []
    for level, hid, text in heads[:40]:
        label = re.sub(r"<[^>]+>", "", text).strip()
        cls = ' class="h3"' if level == "3" else ""
        items.append(f'  <a href="#{hid}"{cls}>{html.escape(label)}</a>')
    return ('<aside class="toc"><div class="group">On this page</div>\n'
            + "\n".join(items) + "\n</aside>")


def first_paragraph(body_html: str) -> str:
    m = re.search(r"<p[^>]*>(.*?)</p>", body_html, flags=re.S)
    if not m:
        return "Deep-learning based doubly robust test for Granger causality."
    txt = re.sub(r"<[^>]+>", "", m.group(1))
    txt = " ".join(txt.split())
    return html.escape(txt[:180])


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serve", action="store_true", help="preview on http://localhost:8000")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    sys.path.insert(0, str(ROOT / "src"))
    from drgct import __version__

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "figures").mkdir(parents=True, exist_ok=True)

    # ---- assets ---- #
    shutil.copy2(SITE / "assets" / "style.css", ASSETS / "style.css")
    copied = 0
    for src in sorted(ROOT.glob("results/**/figures/*.png")):
        shutil.copy2(src, ASSETS / "figures" / src.name)
        copied += 1
    print(f"[site] copied {copied} figures and the stylesheet into docs/assets")

    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    # ---- pages ---- #
    for slug, title, src, _grp in PAGES:
        if not pathlib.Path(src).exists():
            print(f"[site] SKIP {slug}: {src} not found")
            continue
        raw = pathlib.Path(src).read_text(encoding="utf-8")
        raw = apply_directives(raw)
        body, _ = convert(raw)
        page = TEMPLATE.format(
            title=title,
            slug=slug,
            description=first_paragraph(body),
            nav=build_nav(slug),
            body=body,
            toc=build_toc(body),
            version=__version__,
            author=AUTHOR,
            email=AUTHOR_EMAIL,
            github=GITHUB,
            github_user=GITHUB_USER,
            pypi=PYPI,
            paper=PAPER,
            site_url=SITE_URL,
        )
        (OUT / slug).write_text(page, encoding="utf-8")
        print(f"[site] wrote docs/{slug}  ({len(page) // 1024} KB)")

    print(f"\n[site] done — open {OUT / 'index.html'}")
    print("[site] GitHub Pages: Settings -> Pages -> Source: 'Deploy from a branch',")
    print(f"[site]               branch 'main', folder '/docs'.  URL: {SITE_URL}")

    if a.serve:
        import functools
        import http.server
        import socketserver

        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
        with socketserver.TCPServer(("", a.port), handler) as httpd:
            print(f"[site] serving http://localhost:{a.port}/  (Ctrl-C to stop)")
            httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
