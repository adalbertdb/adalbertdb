#!/usr/bin/env python3
"""Generate the neofetch-style profile card (ASCII avatar + info panel) as SVG.

Usage:
    python scripts/generate_profile.py                 # uses https://github.com/<user>.png
    python scripts/generate_profile.py --avatar me.jpg # uses a local image

Set GITHUB_TOKEN (or ACCESS_TOKEN) to refresh the "GitHub Stats" block.
Without a token the last known stats in stats_cache.json are reused.
"""

import argparse
import calendar
import datetime as dt
import html
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "stats_cache.json"

USER = "adalbertdb"
# Used for "Uptime". Leave as None to count from the GitHub account creation date.
BIRTHDAY = dt.date(2005, 5, 12)

# Same fields as the original card. Edit freely; empty values are shown as "-".
INFO = [
    ("OS", "Linux (Omarchy)"),
    ("Uptime", "{uptime}"),
    ("Host", "Comunidad Valenciana, Spain"),
    ("Kernel", "Founder @ Aparcaloo (pre-founding ofc)"),
    ("IDE", "Claude Code, nvim"),
    None,
    ("Languages.Programming", "Java, Dart, Python, JavaScript"),
    ("Languages.Computer", "HTML, CSS, JSON, SQL"),
    ("Languages.Real", "Spanish, English, Valencian, Romanian"),
    None,
    ("Hobbies.General", "Videogames, cars, gym, airsoft"),
    ("Hobbies.Software", "Local LLMs, agentic workflows, MCP"),
    ("Hobbies.Hardware", "Squeezing Gemma 3 into 4GB of VRAM"),
    "GitHub Stats",
    "{stats}",
]

ASCII_COLS = 56
INFO_WIDTH = 62  # characters per info line, dots fill the gap
RAMP = " .:-=+*#%@"


# ---------------------------------------------------------------- ASCII avatar

def load_avatar(path):
    if path:
        return Image.open(path)
    url = f"https://avatars.githubusercontent.com/{USER}?s=460"
    with urllib.request.urlopen(url, timeout=30) as r:
        return Image.open(io.BytesIO(r.read()))


def background_mask(rgb):
    """True where the pixel looks like the flat backdrop (sampled from the corners)."""
    h, w, _ = rgb.shape
    k = max(4, h // 20)
    corners = np.concatenate([
        rgb[:k, :k].reshape(-1, 3), rgb[:k, -k:].reshape(-1, 3),
        rgb[-k:, :k].reshape(-1, 3), rgb[-k:, -k:].reshape(-1, 3),
    ])
    # Bottom corners are often clothes, so only trust the top ones if they differ.
    top = np.concatenate([rgb[:k, :k].reshape(-1, 3), rgb[:k, -k:].reshape(-1, 3)])
    bg = np.median(top if np.abs(np.median(corners, 0) - np.median(top, 0)).sum() > 60 else corners, 0)
    return np.linalg.norm(rgb - bg, axis=2) < 70


def image_to_ascii(img, cols=ASCII_COLS):
    img = ImageOps.exif_transpose(img).convert("RGB")
    fg = ~background_mask(np.asarray(img, dtype=float))

    # Crop to the head: its width comes from the top 60% of the picture so the
    # shoulders don't shrink the face, and the crop keeps a portrait ratio.
    ys, _ = np.nonzero(fg)
    _, hx = np.nonzero(fg[: int(img.height * 0.6)])
    hx0, hx1 = np.percentile(hx, [1, 99])
    pad = (hx1 - hx0) * 0.15
    x0, x1 = int(max(0, hx0 - pad)), int(min(img.width, hx1 + pad))
    y0 = int(np.percentile(ys, 0.5))
    y1 = int(min(img.height, y0 + (x1 - x0) * 1.25))
    img, fg = img.crop((x0, y0, x1, y1)), fg[y0:y1, x0:x1]
    rows = int(cols * img.height / img.width * 0.5)  # glyphs are ~2x taller than wide

    # Local contrast: sharpen, then subtract a blurred copy so uneven lighting
    # doesn't flatten one side of the face.
    gray = ImageOps.grayscale(img).filter(ImageFilter.UnsharpMask(3, 160, 2))
    local = np.asarray(gray, dtype=float) - 0.8 * np.asarray(gray.filter(ImageFilter.GaussianBlur(15)), dtype=float)

    # Equalize over the subject only (not the backdrop) so the whole ramp is used
    # for skin, eyes, hair... Bright -> dense glyphs (light text on a dark card).
    values = np.sort(local[fg])
    equalized = np.searchsorted(values, local) / len(values)
    density = np.asarray(Image.fromarray((equalized * 255).astype(np.uint8)).resize((cols, rows), Image.BOX), dtype=float) / 255
    mask = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((cols, rows), Image.BOX), dtype=float) < 128

    lines = []
    for y in range(rows):
        line = "".join(
            " " if mask[y, x] else RAMP[min(len(RAMP) - 1, int(density[y, x] * len(RAMP)))]
            for x in range(cols)
        )
        lines.append(line.rstrip())
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


# ---------------------------------------------------------------- GitHub stats

def gh(query, variables, token):
    req = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": variables}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.load(r)
    if "errors" in body:
        raise RuntimeError(body["errors"])
    return body["data"]


def rest(path, token):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={"Authorization": f"bearer {token}", "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, (json.load(r) if r.status == 200 else None)


def fetch_stats(token):
    d = gh("""
      query($login: String!) {
        user(login: $login) {
          createdAt
          followers { totalCount }
          repositoriesContributedTo(contributionTypes: [COMMIT, PULL_REQUEST, REPOSITORY]) { totalCount }
          repositories(ownerAffiliations: OWNER, first: 100) {
            totalCount
            nodes { name isFork stargazerCount }
          }
          contributionsCollection { contributionYears }
        }
      }""", {"login": USER}, token)["user"]

    commits = 0
    for year in d["contributionsCollection"]["contributionYears"]:
        c = gh("""
          query($login: String!, $from: DateTime!, $to: DateTime!) {
            user(login: $login) {
              contributionsCollection(from: $from, to: $to) {
                totalCommitContributions restrictedContributionsCount
              }
            }
          }""", {"login": USER, "from": f"{year}-01-01T00:00:00Z", "to": f"{year}-12-31T23:59:59Z"},
            token)["user"]["contributionsCollection"]
        commits += c["totalCommitContributions"] + c["restrictedContributionsCount"]

    added = deleted = 0
    for repo in d["repositories"]["nodes"]:
        if repo["isFork"]:
            continue
        # GitHub computes this lazily and answers 202 until it is ready.
        for _ in range(6):
            status, data = rest(f"/repos/{USER}/{repo['name']}/stats/contributors", token)
            if status == 200:
                break
            time.sleep(5)
        for contributor in data or []:
            if (contributor.get("author") or {}).get("login", "").lower() == USER.lower():
                added += sum(w["a"] for w in contributor["weeks"])
                deleted += sum(w["d"] for w in contributor["weeks"])

    return {
        "created_at": d["createdAt"],
        "repos": d["repositories"]["totalCount"],
        "contributed": d["repositoriesContributedTo"]["totalCount"],
        "stars": sum(r["stargazerCount"] for r in d["repositories"]["nodes"]),
        "commits": commits,
        "followers": d["followers"]["totalCount"],
        "loc_add": added,
        "loc_del": deleted,
    }


def load_stats():
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    token = os.environ.get("ACCESS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("No GITHUB_TOKEN/ACCESS_TOKEN: using cached stats", file=sys.stderr)
        return cached
    try:
        stats = fetch_stats(token)
    except Exception as e:  # keep the card renderable even if the API hiccups
        print(f"Could not fetch stats ({e}): using cached stats", file=sys.stderr)
        return cached
    CACHE.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


# ---------------------------------------------------------------- rendering

def uptime(stats):
    if BIRTHDAY:
        start = BIRTHDAY
    elif stats.get("created_at"):
        start = dt.date.fromisoformat(stats["created_at"][:10])
    else:
        return "-"
    today = dt.date.today()
    years = today.year - start.year - ((today.month, today.day) < (start.month, start.day))
    months = (today.month - start.month - (today.day < start.day)) % 12
    # Days since the last monthly anniversary.
    y, m = (today.year, today.month) if today.day >= start.day else divmod(today.year * 12 + today.month - 2, 12)
    if today.day < start.day:
        m += 1
    last = dt.date(y, m, min(start.day, calendar.monthrange(y, m)[1]))
    days = (today - last).days
    return f"{years} years, {months} months, {days} days"


def fmt(n):
    return "-" if n is None else f"{n:,}"


def build_info(stats):
    """Returns a list of lines; each line is a list of (text, css_class) segments."""
    out = [[(f"{USER}@github ", "key"), ("-" * (INFO_WIDTH - len(USER) - 8), "cc")]]

    def row(key, value):
        value = value or "-"
        dots = " " + "." * max(1, INFO_WIDTH - len(key) - len(value) - 5) + " "
        return [(". ", "cc"), (key, "key"), (":", "cc"), (dots, "cc"), (value, "value")]

    for item in INFO:
        if item is None:
            out.append([])
        elif item == "{stats}":
            s = stats
            out.append(row("Repos", f"{fmt(s.get('repos'))} {{Contributed: {fmt(s.get('contributed'))}}}"
                                    f" | Stars: {fmt(s.get('stars'))}"))
            out.append(row("Commits", f"{fmt(s.get('commits'))} | Followers: {fmt(s.get('followers'))}"))
            add, dele = s.get("loc_add"), s.get("loc_del")
            total = None if add is None else add - dele
            line = row("Lines of Code on GitHub", fmt(total))
            line[-1] = (fmt(total) + " ( ", "value")
            line += [(f"{fmt(add)}++", "add"), (", ", "value"), (f"{fmt(dele)}--", "del"), (" )", "value")]
            # recompute dots now that the value is longer
            visible = sum(len(t) for t, c in line) - len(line[3][0])
            line[3] = (" " + "." * max(1, INFO_WIDTH - visible - 2) + " ", "cc")
            out.append(line)
        elif isinstance(item, str):
            out.append([("- ", "cc"), (item, "key"), (" " + "-" * (INFO_WIDTH - len(item) - 3), "cc")])
        else:
            key, value = item
            out.append(row(key, value.replace("{uptime}", uptime(stats))))
    return out


THEMES = {
    "dark": {"bg": "#161b22", "text": "#c9d1d9", "key": "#ffa657", "value": "#a5d6ff",
             "cc": "#616e7f", "add": "#3fb950", "del": "#f85149"},
    "light": {"bg": "#f6f8fa", "text": "#24292f", "key": "#953800", "value": "#0a3069",
              "cc": "#c2cfde", "add": "#1a7f37", "del": "#cf222e"},
}


def render_svg(ascii_lines, info_lines, theme):
    c = THEMES[theme]
    font, line_h, char_w = 16, 20, 9.6
    # The portrait uses a smaller font so it gets more detail at about the same
    # height as the info panel (but never so small it turns into noise).
    scale = max(0.65, min(1, len(info_lines) / len(ascii_lines)))
    a_font, a_line = round(font * scale, 2), round(line_h * scale, 2)
    ascii_w = max(len(l) for l in ascii_lines) * char_w * scale
    info_x = round(15 + ascii_w + 25)
    height = round(max(len(ascii_lines) * a_line, len(info_lines) * line_h) + 30)
    width = int(info_x + INFO_WIDTH * char_w + 15)
    info_y = round(30 + max(0, len(ascii_lines) * a_line - len(info_lines) * line_h) / 2)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" font-family="ConsolasFallback,Consolas,monospace" '
        f'width="{width}px" height="{height}px" font-size="{font}px">',
        "<style>",
        "@font-face { font-family: ConsolasFallback; src: local('Consolas'); size-adjust: 109%; }",
        f".key {{fill: {c['key']};}} .value {{fill: {c['value']};}} .cc {{fill: {c['cc']};}}",
        f".add {{fill: {c['add']};}} .del {{fill: {c['del']};}}",
        "text, tspan {white-space: pre;}",
        "</style>",
        f'<rect width="{width}px" height="{height}px" fill="{c["bg"]}" rx="15"/>',
        f'<text x="15" y="30" fill="{c["text"]}" class="ascii" font-size="{a_font}px">',
    ]
    for i, line in enumerate(ascii_lines):
        parts.append(f'<tspan x="15" y="{round(30 + i * a_line, 2)}">{html.escape(line)}</tspan>')
    parts.append("</text>")

    parts.append(f'<text x="{info_x}" y="{info_y}" fill="{c["text"]}">')
    for i, segments in enumerate(info_lines):
        if not segments:
            continue
        spans = "".join(f'<tspan class="{cls}">{html.escape(t)}</tspan>' for t, cls in segments)
        parts.append(f'<tspan x="{info_x}" y="{info_y + i * line_h}">{spans}</tspan>')
    parts.append("</text>")
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--avatar", help="local image instead of the GitHub profile picture")
    ap.add_argument("--cols", type=int, default=ASCII_COLS, help="width of the ASCII portrait")
    ap.add_argument("--print", action="store_true", help="also print the ASCII portrait to stdout")
    args = ap.parse_args()

    info_lines = build_info(load_stats())
    ascii_lines = image_to_ascii(load_avatar(args.avatar), args.cols)
    if args.print:
        print("\n".join(ascii_lines))
    (ROOT / "ascii_avatar.txt").write_text("\n".join(ascii_lines) + "\n")

    for theme in THEMES:
        (ROOT / f"{theme}_mode.svg").write_text(render_svg(ascii_lines, info_lines, theme))
    print("Wrote dark_mode.svg, light_mode.svg and ascii_avatar.txt")


if __name__ == "__main__":
    main()
