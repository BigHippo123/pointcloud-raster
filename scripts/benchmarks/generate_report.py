#!/usr/bin/env python3
"""
Generate Markdown + HTML benchmark reports from PCR benchmark CSV outputs.

Usage:
    python3 generate_report.py --results-dir /path/to/results [--title "My Run"]

Expected inputs in results-dir:
    system_info.txt             key=value system metadata
    cpu_vs_gpu.csv              from benchmark_cpu_gpu.py
    multithread.csv             from benchmark_multithread.py
    hybrid.csv                  from benchmark_hybrid.py
    glyphs_full.csv             from benchmark_glyph_full.py
    bench_glyphs_chart.png      quick glyph chart (optional)
    glyphs_full_chart.png       full glyph chart (optional)
    glyph_01_*.png … _08_*.png  pattern gallery images (optional)
    dc_lidar.csv                from test_dc_lidar.py (optional)
    dc_comparison.png           rendered DC raster comparison (optional)

Outputs:
    benchmark_report.md         GitHub-flavored Markdown
    benchmark_report.html       Standalone HTML (PNGs embedded as base64)
"""

import argparse
import base64
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── Data reading ──────────────────────────────────────────────────────────────

def read_system_info(results_dir: Path) -> dict:
    info = {}
    p = results_dir / "system_info.txt"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip()
    return info


def read_csv_file(path: Path):
    if not path.exists():
        return [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return [], []
        rows = [r for r in reader if r]
    return headers, rows


def embed_png(path: Path):
    if path and path.exists():
        data = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{data}"
    return None


def glob_pngs(results_dir: Path, prefix: str):
    """Return sorted list of PNGs in results_dir matching prefix."""
    return sorted(results_dir.glob(f"{prefix}*.png"))


# ── Formatting helpers ─────────────────────────────────────────────────────────

def fmt_points(s: str) -> str:
    try:
        n = int(float(s))
        if n >= 1_000_000_000:
            return f"{n/1e9:.1f}B"
        if n >= 1_000_000:
            v = n / 1_000_000
            return f"{v:.0f}M" if v == int(v) else f"{v:.1f}M"
        if n >= 1_000:
            v = n / 1_000
            return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
        return str(n)
    except (ValueError, TypeError):
        return s


def fmt_float(s: str, decimals: int = 3) -> str:
    try:
        return f"{float(s):.{decimals}f}"
    except (ValueError, TypeError):
        return str(s)


def fmt_speedup(s: str, baseline_label: str = "baseline"):
    try:
        v = float(s)
        if abs(v - 1.0) < 0.02:
            return baseline_label, "base"
        if v >= 5.0:
            return f"{v:.1f}×", "high"
        if v >= 2.0:
            return f"{v:.1f}×", "med"
        return f"{v:.1f}×", "low"
    except (ValueError, TypeError):
        if str(s).upper() in ("SKIPPED", "", "N/A", "—"):
            return "—", "skip"
        return s, ""


def is_skipped(s: str) -> bool:
    return str(s).upper() in ("SKIPPED", "", "N/A", "—")


# ── Shared CSS ────────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg: #f6f8fa; --card: #fff; --border: #d0d7de; --muted: #656d76;
  --header-bg: #0d1117; --header-fg: #e6edf3; --header-sub: #8b949e;
  --green-bg: #dafbe1; --green-fg: #116329;
  --amber-bg: #fff8c5; --amber-fg: #7d4e00;
  --blue-bg:  #ddf4ff; --blue-fg:  #0550ae;
  --purple-bg:#f3e8ff; --purple-fg:#6f42c1;
  --gray-bg:  #e6edf3; --gray-fg:  #57606a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       background: var(--bg); color: #1f2328; line-height: 1.5; }
a { color: var(--blue-fg); }

.hdr { background: var(--header-bg); color: var(--header-fg); padding: 24px 32px; }
.hdr h1 { font-size: 1.5rem; margin-bottom: 4px; }
.hdr p  { font-size: 0.875rem; color: var(--header-sub); }

.wrap { max-width: 1120px; margin: 0 auto; padding: 24px 16px; }

.card { background: var(--card); border: 1px solid var(--border);
        border-radius: 6px; margin-bottom: 16px; overflow: hidden; }
.card-hdr { padding: 10px 16px; border-bottom: 1px solid var(--border);
            background: var(--bg); display: flex; align-items: center; gap: 8px; }
.card-hdr h2 { font-size: 1rem; flex: 1; }
.card-body { padding: 16px; }

.sys-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 16px; }
.sys-item .lbl { font-size: 0.72rem; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); }
.sys-item .val { font-size: 0.9375rem; font-weight: 600; margin-top: 2px; }

.tbl-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
thead th { text-align: right; padding: 8px 12px; border-bottom: 2px solid var(--border);
           color: var(--muted); font-size: 0.72rem; text-transform: uppercase;
           letter-spacing: .05em; white-space: nowrap; }
thead th:first-child, thead th:nth-child(2) { text-align: left; }
tbody td { padding: 8px 12px; border-bottom: 1px solid var(--border); text-align: right; }
tbody td:first-child, tbody td:nth-child(2) { text-align: left; font-weight: 500; }
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: var(--bg); }
tbody tr.sep td { border-top: 2px solid var(--border); }

.badge { display: inline-block; padding: 1px 8px; border-radius: 12px;
         font-weight: 600; font-size: 0.8rem; }
.badge-high   { background: var(--green-bg);  color: var(--green-fg); }
.badge-med    { background: var(--amber-bg);  color: var(--amber-fg); }
.badge-base   { background: var(--gray-bg);   color: var(--gray-fg); font-style: italic; }
.badge-skip   { color: var(--muted); font-style: italic; }

.tag { font-size: 0.72rem; padding: 2px 8px; border-radius: 10px; font-weight: 600; }
.tag-cpu    { background: var(--blue-bg);   color: var(--blue-fg); }
.tag-gpu    { background: var(--green-bg);  color: var(--green-fg); }
.tag-hybrid { background: var(--amber-bg);  color: var(--amber-fg); }
.tag-glyph  { background: var(--gray-bg);   color: var(--gray-fg); }
.tag-dc     { background: var(--purple-bg); color: var(--purple-fg); }

.chart-wrap { margin-top: 16px; text-align: center; }
.chart-wrap img { max-width: 100%; border-radius: 4px; border: 1px solid var(--border); }

/* Glyph gallery grid */
.gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; margin-top: 4px; }
.gallery-item { border: 1px solid var(--border); border-radius: 4px; overflow: hidden; }
.gallery-item img { width: 100%; display: block; }
.gallery-caption { padding: 6px 10px; font-size: 0.8rem; color: var(--muted);
                   background: var(--bg); border-top: 1px solid var(--border); }

.notice { background: var(--blue-bg); border: 1px solid #54aeff; border-radius: 6px;
          padding: 10px 14px; font-size: 0.875rem; color: var(--blue-fg); }

.footer { text-align: center; padding: 24px; color: var(--muted); font-size: 0.8125rem; }
"""


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _badge(sp_text: str, hint: str) -> str:
    cls = {"high": "badge-high", "med": "badge-med",
           "base": "badge-base", "skip": "badge-skip"}.get(hint, "")
    return f'<span class="badge {cls}">{sp_text}</span>'


def html_system_card(info: dict) -> str:
    pairs = []
    if "cpu" in info:
        pairs.append(("CPU", info["cpu"]))
    if "cpu_cores" in info:
        pairs.append(("Cores", info["cpu_cores"]))
    if "ram_gb" in info:
        pairs.append(("RAM", f"{info['ram_gb']} GB"))
    gpu = info.get("gpu", "none")
    if gpu and gpu != "none":
        pairs.append(("GPU", gpu))
        if "gpu_memory" in info:
            pairs.append(("GPU Memory", info["gpu_memory"]))
        if "cuda_version" in info:
            pairs.append(("CUDA", info["cuda_version"]))
    else:
        pairs.append(("GPU", '<span style="color:var(--muted)">Not available</span>'))
    if "pcr_version" in info:
        pairs.append(("PCR", f"v{info['pcr_version']}"))

    items = "".join(
        f'<div class="sys-item"><div class="lbl">{k}</div>'
        f'<div class="val">{v}</div></div>'
        for k, v in pairs
    )
    return (
        '<div class="card">'
        '<div class="card-hdr"><h2>System Information</h2></div>'
        f'<div class="card-body"><div class="sys-grid">{items}</div></div>'
        '</div>'
    )


def html_throughput_card(title: str, tag: str, headers, rows,
                         baseline_label: str = "baseline") -> str:
    if not rows:
        body = '<div class="notice">No data — benchmark was skipped or GPU not available.</div>'
    else:
        hdr_html = (
            "<thead><tr><th>Points</th><th>Mode</th>"
            "<th>Time (s)</th><th>Mpt/s</th><th>Speedup</th></tr></thead>"
        )
        tbody_rows = []
        prev_n = None
        for row in rows:
            if len(row) < 4:
                continue
            n_str = fmt_points(row[0])
            sep = ' class="sep"' if prev_n and prev_n != n_str else ""
            prev_n = n_str
            time_s, tput = row[2], row[3]
            sp_raw = row[4] if len(row) > 4 else ""
            if is_skipped(time_s):
                td_t = td_r = '<td class="badge-skip">—</td>'
                td_sp = '<td class="badge-skip">—</td>'
            else:
                td_t = f"<td>{fmt_float(time_s, 3)}</td>"
                td_r = f"<td>{fmt_float(tput, 2)}</td>"
                sp_text, hint = fmt_speedup(sp_raw, baseline_label)
                td_sp = f"<td>{_badge(sp_text, hint)}</td>"
            tbody_rows.append(
                f"<tr{sep}><td>{n_str}</td><td>{row[1]}</td>{td_t}{td_r}{td_sp}</tr>"
            )
        body = (
            f'<div class="tbl-wrap"><table>{hdr_html}'
            f"<tbody>{''.join(tbody_rows)}</tbody></table></div>"
        )

    return (
        '<div class="card">'
        f'<div class="card-hdr"><h2>{title}</h2>'
        f'<span class="tag tag-{tag}">{tag.upper()}</span></div>'
        f'<div class="card-body">{body}</div>'
        '</div>'
    )


def html_glyph_bench_card(title: str, headers, rows,
                          chart_url=None) -> str:
    if rows:
        hdr_html = (
            "<thead><tr><th>Glyph</th><th>Points</th><th>Mode</th>"
            "<th>Time (s)</th><th>Mpt/s</th></tr></thead>"
        )
        tbody_rows = [
            f"<tr><td>{r[0]}</td><td>{fmt_points(r[1])}</td><td>{r[2]}</td>"
            f"<td>{fmt_float(r[3], 3)}</td><td>{fmt_float(r[4], 2)}</td></tr>"
            for r in rows if len(r) >= 5
        ]
        table_html = (
            f'<div class="tbl-wrap"><table>{hdr_html}'
            f"<tbody>{''.join(tbody_rows)}</tbody></table></div>"
        )
    else:
        table_html = '<div class="notice">No data — run benchmark_glyph_full.py to generate.</div>'

    chart_html = (
        f'<div class="chart-wrap"><img src="{chart_url}" alt="Glyph chart"></div>'
        if chart_url else ""
    )
    return (
        '<div class="card">'
        f'<div class="card-hdr"><h2>{title}</h2>'
        '<span class="tag tag-glyph">GLYPH</span></div>'
        f'<div class="card-body">{table_html}{chart_html}</div>'
        '</div>'
    )


# Gallery captions for the 8 pattern images
_GLYPH_CAPTIONS = {
    "01_gap_fill_comparison": "Gap-Fill: Point vs Gaussian σ=2 vs σ=6 on a sparse cloud",
    "02_sigma_progression":   "Gaussian Sigma Progression: σ = 0.5 → 16 cells",
    "03_anisotropic_gaussian":"Anisotropic & Rotated Gaussian (σx ≠ σy, rotation)",
    "04_line_directions":     "Line Glyph — Direction Sweep & Half-Length Progression",
    "05_flow_field":          "Flow Field — Line Glyph vs Adaptive Gaussian",
    "06_sparse_vs_dense":     "Density Comparison: Point vs Gaussian at 50 / 500 / 5,000 pts",
    "07_per_point_sigma":     "Adaptive Per-Point Sigma (σ ∝ distance from centre)",
    "08_glyph_showcase":      "Bullseye Showcase — Point, Line, Gaussian σ=2, Gaussian σ=5",
}


def html_glyph_gallery_card(gallery_pngs: list) -> str:
    if not gallery_pngs:
        return (
            '<div class="card">'
            '<div class="card-hdr"><h2>Glyph Visual Gallery</h2>'
            '<span class="tag tag-glyph">GALLERY</span></div>'
            '<div class="card-body"><div class="notice">'
            'No gallery images — run <code>scripts/patterns/generate_glyph_patterns.py</code>.</div>'
            '</div></div>'
        )

    items = []
    for png_path in gallery_pngs:
        url = embed_png(png_path)
        if not url:
            continue
        stem = png_path.stem  # e.g. "01_gap_fill_comparison"
        caption = _GLYPH_CAPTIONS.get(stem, stem.replace("_", " ").title())
        items.append(
            f'<div class="gallery-item">'
            f'<img src="{url}" alt="{caption}" loading="lazy">'
            f'<div class="gallery-caption">{caption}</div>'
            f'</div>'
        )

    return (
        '<div class="card">'
        '<div class="card-hdr"><h2>Glyph Visual Gallery</h2>'
        '<span class="tag tag-glyph">GALLERY</span></div>'
        f'<div class="card-body"><div class="gallery">{"".join(items)}</div></div>'
        '</div>'
    )


def html_dc_lidar_card(headers, rows, comparison_url=None) -> str:
    if rows:
        hdr_html = (
            "<thead><tr><th>Mode</th><th>Glyph</th><th>Points</th>"
            "<th>Wall (s)</th><th>I/O (s)</th><th>Ingest (s)</th>"
            "<th>Mpt/s</th><th>Speedup</th></tr></thead>"
        )
        # Map column names to indices
        col = {h: i for i, h in enumerate(headers)} if headers else {}
        tbody_rows = []
        prev_mode = None
        for row in rows:
            def get(name, default=""):
                i = col.get(name)
                return row[i] if i is not None and i < len(row) else default
            mode = get("mode", "")
            sep = ' class="sep"' if prev_mode and prev_mode != mode else ""
            prev_mode = mode
            pts = fmt_points(get("points"))
            wall = fmt_float(get("wall_total_s"), 1)
            io_s = fmt_float(get("io_read_s"), 1)
            ing  = fmt_float(get("ingest_s"), 1)
            mpts = fmt_float(get("throughput_mpts"), 2)
            sp_raw = get("speedup")
            sp_text, hint = fmt_speedup(sp_raw)
            td_sp = _badge(sp_text, hint)
            glyph = get("glyph_label") or get("glyph")
            tbody_rows.append(
                f"<tr{sep}><td>{mode}</td><td>{glyph}</td><td>{pts}</td>"
                f"<td>{wall}</td><td>{io_s}</td><td>{ing}</td>"
                f"<td>{mpts}</td><td>{td_sp}</td></tr>"
            )
        table_html = (
            f'<div class="tbl-wrap"><table>{hdr_html}'
            f"<tbody>{''.join(tbody_rows)}</tbody></table></div>"
        )
    else:
        table_html = (
            '<div class="notice">No data — run '
            '<code>scripts/data/test_dc_lidar.py --las-dir /path/to/las --glyph all</code> '
            'to generate real-world LiDAR results.</div>'
        )

    chart_html = (
        f'<div class="chart-wrap"><img src="{comparison_url}" '
        f'alt="DC LiDAR glyph comparison" loading="lazy"></div>'
        if comparison_url else ""
    )

    return (
        '<div class="card">'
        '<div class="card-hdr"><h2>DC LiDAR — Real-World Example</h2>'
        '<span class="tag tag-dc">REAL DATA</span></div>'
        f'<div class="card-body">{table_html}{chart_html}</div>'
        '</div>'
    )


# ── Markdown generation ────────────────────────────────────────────────────────

def md_system_section(info: dict) -> str:
    rows = []
    if "cpu" in info:      rows.append(("CPU", info["cpu"]))
    if "cpu_cores" in info: rows.append(("CPU Cores", info["cpu_cores"]))
    if "ram_gb" in info:   rows.append(("RAM", f"{info['ram_gb']} GB"))
    gpu = info.get("gpu", "none")
    if gpu and gpu != "none":
        rows.append(("GPU", gpu))
        if "gpu_memory" in info: rows.append(("GPU Memory", info["gpu_memory"]))
        if "cuda_version" in info: rows.append(("CUDA", info["cuda_version"]))
    else:
        rows.append(("GPU", "Not available (CPU-only run)"))
    if "pcr_version" in info: rows.append(("PCR Version", info["pcr_version"]))
    if "timestamp" in info:   rows.append(("Generated", info["timestamp"]))

    lines = ["## System Information", "", "| | |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| **{k}** | {v} |")
    lines.append("")
    return "\n".join(lines)


def md_throughput_section(title: str, headers, rows, baseline_label: str) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        lines += ["> _No data — benchmark skipped or GPU not available._", ""]
        return "\n".join(lines)
    lines += [
        "| Points | Mode | Time (s) | Mpt/s | Speedup |",
        "|-------:|:-----|----------:|------:|--------:|",
    ]
    prev_n = None
    for row in rows:
        if len(row) < 4:
            continue
        n_str = fmt_points(row[0])
        if prev_n and prev_n != n_str:
            lines.append("| | | | | |")
        prev_n = n_str
        time_s, tput = row[2], row[3]
        sp_raw = row[4] if len(row) > 4 else ""
        if is_skipped(time_s):
            lines.append(f"| {n_str} | {row[1]} | — | — | — |")
        else:
            sp_text, hint = fmt_speedup(sp_raw, baseline_label)
            sp_fmt = f"**{sp_text}**" if hint in ("high", "med") else sp_text
            lines.append(
                f"| {n_str} | {row[1]} | {fmt_float(time_s, 3)} "
                f"| {fmt_float(tput, 2)} | {sp_fmt} |"
            )
    lines.append("")
    return "\n".join(lines)


def md_glyph_bench_section(title: str, headers, rows, chart_path=None) -> str:
    lines = [f"## {title}", ""]
    if rows:
        lines += [
            "| Glyph | Points | Mode | Time (s) | Mpt/s |",
            "|:------|-------:|:-----|----------:|------:|",
        ]
        for row in rows:
            if len(row) >= 5:
                lines.append(
                    f"| {row[0]} | {fmt_points(row[1])} | {row[2]} "
                    f"| {fmt_float(row[3], 3)} | {fmt_float(row[4], 2)} |"
                )
        lines.append("")
    else:
        lines += ["> _No data — run benchmark_glyph_full.py to generate._", ""]
    if chart_path and chart_path.exists():
        lines += [f"![Glyph benchmark chart]({chart_path.name})", ""]
    return "\n".join(lines)


def md_glyph_gallery_section(gallery_pngs: list) -> str:
    lines = ["## Glyph Visual Gallery", ""]
    if not gallery_pngs:
        lines += [
            "> _No gallery images — run `scripts/patterns/generate_glyph_patterns.py`._", ""
        ]
        return "\n".join(lines)
    lines.append("Visual examples of each glyph type rendered on synthetic point clouds.")
    lines.append("")
    for p in gallery_pngs:
        caption = _GLYPH_CAPTIONS.get(p.stem, p.stem.replace("_", " ").title())
        lines += [f"### {caption}", f"![]({p.name})", ""]
    return "\n".join(lines)


def md_dc_lidar_section(headers, rows, comparison_path=None) -> str:
    lines = ["## DC LiDAR — Real-World Example", ""]
    if rows:
        col = {h: i for i, h in enumerate(headers)} if headers else {}
        lines += [
            "| Mode | Glyph | Points | Wall (s) | Mpt/s | Speedup |",
            "|:-----|:------|-------:|---------:|------:|--------:|",
        ]
        for row in rows:
            def get(name, default=""):
                i = col.get(name)
                return row[i] if i is not None and i < len(row) else default
            glyph = get("glyph_label") or get("glyph")
            pts = fmt_points(get("points"))
            wall = fmt_float(get("wall_total_s"), 1)
            mpts = fmt_float(get("throughput_mpts"), 2)
            sp_raw = get("speedup")
            sp_text, hint = fmt_speedup(sp_raw)
            sp_fmt = f"**{sp_text}**" if hint in ("high", "med") else sp_text
            lines.append(
                f"| {get('mode')} | {glyph} | {pts} | {wall} | {mpts} | {sp_fmt} |"
            )
        lines.append("")
    else:
        lines += [
            "> _No data — run `scripts/data/test_dc_lidar.py --las-dir /path/to/las --glyph all`_",
            ""
        ]
    if comparison_path and comparison_path.exists():
        lines += [f"![DC LiDAR comparison]({comparison_path.name})", ""]
    return "\n".join(lines)


# ── Full report assembly ───────────────────────────────────────────────────────

def generate_html(results_dir: Path, info: dict, title: str) -> str:
    ts = info.get("timestamp",
                  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    gpu = info.get("gpu", "none")
    gpu_note = f"GPU: {gpu}" if gpu and gpu != "none" else "CPU-only run"

    sections = [html_system_card(info)]

    # ── Throughput benchmarks ──
    h, rows = read_csv_file(results_dir / "cpu_vs_gpu.csv")
    sections.append(html_throughput_card(
        "CPU vs GPU Throughput", "gpu", h, rows, "baseline (1T)"))

    h, rows = read_csv_file(results_dir / "multithread.csv")
    sections.append(html_throughput_card(
        "CPU Multi-Thread Scaling", "cpu", h, rows, "baseline (1T)"))

    h, rows = read_csv_file(results_dir / "hybrid.csv")
    sections.append(html_throughput_card(
        "Hybrid Mode (CPU routing + GPU accumulation)", "hybrid", h, rows, "baseline (MT)"))

    # ── Glyph throughput table + chart ──
    h, rows = read_csv_file(results_dir / "glyphs_full.csv")
    chart_url = (embed_png(results_dir / "glyphs_full_chart.png") or
                 embed_png(results_dir / "bench_glyphs_chart.png"))
    sections.append(html_glyph_bench_card(
        "Glyph Rendering Throughput", h, rows, chart_url))

    # ── Glyph visual gallery ──
    gallery_pngs = glob_pngs(results_dir, "glyph_")
    sections.append(html_glyph_gallery_card(gallery_pngs))

    # ── DC LiDAR real-world example ──
    h, rows = read_csv_file(results_dir / "dc_lidar.csv")
    dc_cmp_url = embed_png(results_dir / "dc_comparison.png")
    sections.append(html_dc_lidar_card(h, rows, dc_cmp_url))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{_CSS}</style>
</head>
<body>
  <div class="hdr">
    <h1>{title}</h1>
    <p>{ts} &nbsp;·&nbsp; {info.get('hostname', 'unknown')} &nbsp;·&nbsp; {gpu_note}</p>
  </div>
  <div class="wrap">
{"".join(sections)}
  </div>
  <div class="footer">
    Generated by <code>scripts/benchmarks/generate_report.py</code>
    &nbsp;·&nbsp; <a href="https://github.com/anthropics/pcr">PCR Library</a>
  </div>
</body>
</html>
"""


def generate_markdown(results_dir: Path, info: dict, title: str) -> str:
    ts = info.get("timestamp",
                  datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    gpu = info.get("gpu", "none")
    gpu_tag = f" · GPU: {gpu}" if gpu and gpu != "none" else " · CPU-only"

    parts = [
        f"# {title}", "",
        f"**Generated:** {ts} · **Host:** {info.get('hostname', 'unknown')}{gpu_tag}",
        "",
        md_system_section(info),
        "---", "",
    ]

    h, rows = read_csv_file(results_dir / "cpu_vs_gpu.csv")
    parts.append(md_throughput_section("CPU vs GPU Throughput", h, rows, "baseline (1T)"))

    h, rows = read_csv_file(results_dir / "multithread.csv")
    parts.append(md_throughput_section("CPU Multi-Thread Scaling", h, rows, "baseline (1T)"))

    h, rows = read_csv_file(results_dir / "hybrid.csv")
    parts.append(md_throughput_section(
        "Hybrid Mode (CPU routing + GPU accumulation)", h, rows, "baseline (MT)"))

    h, rows = read_csv_file(results_dir / "glyphs_full.csv")
    chart = results_dir / "glyphs_full_chart.png"
    if not chart.exists():
        chart = results_dir / "bench_glyphs_chart.png"
    parts.append(md_glyph_bench_section("Glyph Rendering Throughput", h, rows,
                                        chart if chart.exists() else None))

    gallery_pngs = glob_pngs(results_dir, "glyph_")
    parts.append(md_glyph_gallery_section(gallery_pngs))

    h, rows = read_csv_file(results_dir / "dc_lidar.csv")
    dc_cmp = results_dir / "dc_comparison.png"
    parts.append(md_dc_lidar_section(h, rows, dc_cmp if dc_cmp.exists() else None))

    parts += ["---", "", "*Generated by `scripts/benchmarks/generate_report.py`*", ""]
    return "\n".join(parts)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate PCR benchmark report")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--title", default="PCR Benchmark Report")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        print(f"Error: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    info = read_system_info(results_dir)
    if not info:
        info["timestamp"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    md_path = results_dir / "benchmark_report.md"
    html_path = results_dir / "benchmark_report.html"

    md_path.write_text(generate_markdown(results_dir, info, args.title))
    html_path.write_text(generate_html(results_dir, info, args.title))

    print(f"  Markdown : {md_path}")
    print(f"  HTML     : {html_path}")


if __name__ == "__main__":
    main()
