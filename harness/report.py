"""Standalone HTML reports (matrix and compare). Inline CSS and inline SVG only: no JS, no fonts, no CDN.

Colour follows the dataviz rules: one categorical slot per model in fixed order (blue, orange, aqua),
pass/fail as status colours with a shape and a word beside them (never colour alone), text always in
ink tokens, light and dark themes defined as variables from the same ramps.
"""

from __future__ import annotations

import html
import math
from typing import TYPE_CHECKING

from harness.compare import VERDICT_TEXT, Comparison, explain, fmt_p, noise_floor_lines, pct, points

if TYPE_CHECKING:
    from harness.matrix import MatrixData

_THEME_LIGHT = """
  color-scheme: light;
  --bg: #f4f4f1; --surface: #fcfcfb; --ink: #0b0b0b; --ink2: #52514e; --ink3: #6b6a65;
  --line: #e2e1dc; --grid: #ebeae6; --wash: #f0efec;
  --s1: #2a78d6; --s2: #eb6834; --s3: #1baf7a; --s4: #eda100; --s5: #e87ba4; --s6: #008300;
  --good: #0ca30c; --bad: #d03b3b; --warn: #b87a00; --pos: #2a78d6; --neg: #e34948;
"""
_THEME_DARK = """
  color-scheme: dark;
  --bg: #121211; --surface: #1a1a19; --ink: #f0efec; --ink2: #c3c2b7; --ink3: #9a998f;
  --line: #2e2e2b; --grid: #272725; --wash: #232321;
  --s1: #3987e5; --s2: #d95926; --s3: #199e70; --s4: #c98500; --s5: #d55181; --s6: #008300;
  --good: #0ca30c; --bad: #d03b3b; --warn: #fab219; --pos: #3987e5; --neg: #e66767;
"""

CSS = f"""
:root {{{_THEME_LIGHT}}}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) {{{_THEME_DARK}}} }}
:root[data-theme="dark"] {{{_THEME_DARK}}}
* {{ box-sizing: border-box; }}
html {{ -webkit-text-size-adjust: 100%; }}
body {{ margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 960px; margin: 0 auto; padding: 40px 16px 64px; }}
h1 {{ font-size: 26px; line-height: 1.2; margin: 0 0 6px; font-weight: 650; letter-spacing: -0.01em; }}
h2 {{ font-size: 17px; margin: 0 0 4px; font-weight: 650; }}
.sub {{ color: var(--ink2); margin: 0 0 28px; font-size: 14px; }}
.card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 20px 22px; margin: 0 0 18px; }}
.card > .hint {{ color: var(--ink2); font-size: 13px; margin: 0 0 14px; }}
.scroll {{ overflow-x: auto; }}
svg {{ display: block; width: 100%; height: auto; min-width: 480px; }}
svg text {{ font-family: inherit; fill: var(--ink); }}
svg .t2 {{ fill: var(--ink2); }} svg .t3 {{ fill: var(--ink3); }}
svg .axis {{ stroke: var(--line); stroke-width: 1; }} svg .grid {{ stroke: var(--grid); stroke-width: 1; }}
svg .zero {{ stroke: var(--ink2); stroke-width: 1; }}
.m0 {{ fill: var(--s1); stroke: var(--s1); }} .m1 {{ fill: var(--s2); stroke: var(--s2); }}
.m2 {{ fill: var(--s3); stroke: var(--s3); }} .m3 {{ fill: var(--s4); stroke: var(--s4); }}
.m4 {{ fill: var(--s5); stroke: var(--s5); }} .m5 {{ fill: var(--s6); stroke: var(--s6); }}
svg .nostroke {{ stroke: none; }} svg .nofill {{ fill: none; }}
svg .ring {{ stroke: var(--surface); stroke-width: 2; }}
svg .cell-pass {{ fill: var(--good); }} svg .cell-fail {{ fill: var(--bad); }}
svg .cell-x {{ stroke: var(--surface); stroke-width: 1.6; fill: none; }}
svg .cell:hover {{ stroke: var(--ink); stroke-width: 1.5; }}
svg .diff {{ fill: var(--ink2); }}
.legend {{ display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 0 0 12px; font-size: 13px; color: var(--ink2); }}
.legend i {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
.legend i.m0 {{ background: var(--s1); }} .legend i.m1 {{ background: var(--s2); }} .legend i.m2 {{ background: var(--s3); }}
.legend i.m3 {{ background: var(--s4); }} .legend i.m4 {{ background: var(--s5); }} .legend i.m5 {{ background: var(--s6); }}
.legend i {{ stroke: none; }}
.chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px; font-weight: 600; font-size: 14px; border: 1px solid var(--line); background: var(--wash); }}
.chip svg {{ width: 14px; height: 14px; min-width: 0; }}
.chip.good {{ border-color: var(--good); }} .chip.bad {{ border-color: var(--bad); }} .chip.warn {{ border-color: var(--warn); }}
.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 0 0 18px; }}
.tile {{ background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
.tile .l {{ color: var(--ink2); font-size: 13px; }} .tile .v {{ font-size: 26px; font-weight: 650; line-height: 1.2; margin-top: 2px; }}
.tile .d {{ color: var(--ink2); font-size: 12.5px; margin-top: 2px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ color: var(--ink2); font-weight: 600; font-size: 12.5px; white-space: nowrap; }}
td.n, th.n {{ text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }}
tr:last-child td {{ border-bottom: 0; }}
.pill {{ display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 12px; border: 1px solid var(--line); background: var(--wash); white-space: nowrap; }}
.pill.good {{ border-color: var(--good); }} .pill.bad {{ border-color: var(--bad); }}
.banner {{ border: 1px solid var(--warn); background: var(--wash); border-radius: 10px; padding: 10px 14px; margin: 0 0 18px; font-size: 14px; }}
ul.plain {{ margin: 0; padding-left: 20px; }} ul.plain li {{ margin: 0 0 8px; }}
.mono {{ font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12.5px; }}
.foot {{ color: var(--ink3); font-size: 12.5px; margin-top: 28px; }}
@media (max-width: 560px) {{ main {{ padding-top: 24px; }} .card {{ padding: 16px; }} h1 {{ font-size: 22px; }} }}
"""


class Raw(str):
    """Markup that must not be escaped again."""


def esc(x: object) -> str:
    return x if isinstance(x, Raw) else html.escape(str(x), quote=True)


def page(title: str, subtitle: str, body: str, synthetic: bool = False) -> str:
    banner = (
        '<div class="banner" role="note"><b>SYNTHETIC.</b> This report comes from the fake adapter. '
        "The numbers are not a real model's.</div>"
        if synthetic
        else ""
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>"
        f'<h1>{esc(title)}</h1><p class="sub">{esc(subtitle)}</p>{banner}{body}'
        '<p class="foot">Generated by the FDE eval harness. Self-contained page: no scripts, no external assets.</p>'
        "</main></body></html>"
    )


def card(title: str, body: str, hint: str = "") -> str:
    h = f'<p class="hint">{esc(hint)}</p>' if hint else ""
    return f'<section class="card"><h2>{esc(title)}</h2>{h}{body}</section>'


def table(headers: list[str], rows: list[list[object]], num_cols: set[int] | None = None) -> str:
    num_cols = num_cols or set()
    head = "".join(f'<th class="{"n" if i in num_cols else ""}">{esc(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{"n" if i in num_cols else ""}">{esc(c)}</td>' for i, c in enumerate(r))
        + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _f(x: float) -> str:
    return f"{x:.1f}"


def _rounded_bar(x0: float, y: float, w: float, h: float, cls: str, r: float = 4) -> str:
    """Horizontal bar from x0, square at the baseline, rounded at the data end."""
    if w <= 0:
        return ""
    if w < r * 2:
        return f'<rect x="{_f(x0)}" y="{_f(y)}" width="{_f(w)}" height="{h}" class="{cls} nostroke"/>'
    x1 = x0 + w
    d = (
        f"M{_f(x0)},{_f(y)} H{_f(x1 - r)} Q{_f(x1)},{_f(y)} {_f(x1)},{_f(y + r)} V{_f(y + h - r)} "
        f"Q{_f(x1)},{_f(y + h)} {_f(x1 - r)},{_f(y + h)} H{_f(x0)} Z"
    )
    return f'<path d="{d}" class="{cls} nostroke"/>'


def _legend(models: list[str]) -> str:
    items = "".join(f'<span><i class="m{i % 6}"></i>{esc(m)}</span>' for i, m in enumerate(models))
    return f'<div class="legend">{items}</div>'


def _icon(kind: str) -> str:
    """Small status shapes: check, cross, dash. Shape plus word, never colour alone."""
    paths = {
        "good": '<path d="M2 7.5 5.5 11 12 3.5" fill="none" stroke="var(--good)" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/>',
        "bad": '<path d="M3 3 11 11M11 3 3 11" fill="none" stroke="var(--bad)" stroke-width="2.2" stroke-linecap="round"/>',
        "warn": '<path d="M7 2.5 12.5 12H1.5Z" fill="none" stroke="var(--warn)" stroke-width="2" stroke-linejoin="round"/>',
        "flat": '<path d="M2.5 7h9" fill="none" stroke="var(--ink2)" stroke-width="2.2" stroke-linecap="round"/>',
    }
    return f'<svg viewBox="0 0 14 14" aria-hidden="true">{paths[kind]}</svg>'


# ---- charts: matrix ----

_AXIS_PCT = (0, 20, 40, 60, 80, 100)


def accuracy_chart(d: MatrixData) -> str:
    """Overall accuracy per model with its 95% Wilson interval (dot and whisker)."""
    w, left, right, row_h, top = 680, 190, 24, 46, 8
    plot_w = w - left - right
    h = top + row_h * len(d.models) + 30
    out = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Overall accuracy by model with 95 percent intervals">'
    ]
    for t in _AXIS_PCT:
        x = left + plot_w * t / 100
        out.append(f'<line x1="{_f(x)}" x2="{_f(x)}" y1="{top}" y2="{h - 24}" class="grid"/>')
        out.append(
            f'<text x="{_f(x)}" y="{h - 8}" text-anchor="middle" font-size="11" class="t3">{t}%</text>'
        )
    for i, s in enumerate(d.models):
        cy = top + row_h * i + row_h / 2
        lo, hi = left + plot_w * s.ci[0], left + plot_w * s.ci[1]
        px = left + plot_w * s.accuracy
        out.append(f'<text x="0" y="{_f(cy - 3)}" font-size="13" font-weight="600">{esc(s.model)}</text>')
        out.append(
            f'<text x="0" y="{_f(cy + 13)}" font-size="12" class="t2">{s.passed}/{s.n} passed, '
            f"{100 * s.accuracy:.0f}%</text>"
        )
        title = (
            f"{s.model}: {s.passed}/{s.n} passed ({100 * s.accuracy:.1f}%), "
            f"95% interval {100 * s.ci[0]:.0f}% to {100 * s.ci[1]:.0f}%"
        )
        out.append(
            f'<g class="cell"><title>{esc(title)}</title>'
            f'<line x1="{_f(lo)}" x2="{_f(hi)}" y1="{_f(cy)}" y2="{_f(cy)}" class="m{i % 6}" stroke-width="2" stroke-linecap="round"/>'
            f'<line x1="{_f(lo)}" x2="{_f(lo)}" y1="{_f(cy - 5)}" y2="{_f(cy + 5)}" class="m{i % 6}" stroke-width="2"/>'
            f'<line x1="{_f(hi)}" x2="{_f(hi)}" y1="{_f(cy - 5)}" y2="{_f(cy + 5)}" class="m{i % 6}" stroke-width="2"/>'
            f'<circle cx="{_f(px)}" cy="{_f(cy)}" r="5.5" class="m{i % 6} ring"/></g>'
        )
    out.append("</svg>")
    return "".join(out)


def category_chart(d: MatrixData) -> str:
    """Accuracy by category: one thin bar per model, grouped by category."""
    bar_h, gap, group_gap, left, right = 14, 2, 20, 150, 56
    w = 680
    plot_w = w - left - right
    group_h = len(d.models) * (bar_h + gap) - gap
    h = 8 + len(d.categories) * (group_h + group_gap) + 22
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Accuracy by category and model">']
    axis_bottom = h - 22
    for t in _AXIS_PCT:
        x = left + plot_w * t / 100
        out.append(f'<line x1="{_f(x)}" x2="{_f(x)}" y1="4" y2="{axis_bottom}" class="grid"/>')
        out.append(
            f'<text x="{_f(x)}" y="{h - 6}" text-anchor="middle" font-size="11" class="t3">{t}%</text>'
        )
    for gi, cat in enumerate(d.categories):
        y0 = 8 + gi * (group_h + group_gap)
        n = max((s.by_cat.get(cat, (0, 0))[1] for s in d.models), default=0)
        out.append(
            f'<text x="0" y="{_f(y0 + group_h / 2 + 4)}" font-size="13" font-weight="600">{esc(cat)}</text>'
        )
        out.append(
            f'<text x="0" y="{_f(y0 + group_h / 2 + 19)}" font-size="11.5" class="t2">{n} cases</text>'
        )
        for mi, s in enumerate(d.models):
            k, tot = s.by_cat.get(cat, (0, 0))
            acc = k / tot if tot else 0
            y = y0 + mi * (bar_h + gap)
            bw = plot_w * acc
            out.append(
                f'<g class="cell"><title>{esc(f"{s.model}, {cat}: {k}/{tot} ({100 * acc:.0f}%)")}</title>'
            )
            out.append(
                f'<rect x="{left}" y="{_f(y)}" width="{plot_w}" height="{bar_h}" fill="none" stroke="none"/>'
            )
            out.append(_rounded_bar(left, y, bw, bar_h, f"m{mi % 6}"))
            out.append(
                f'<text x="{_f(left + bw + 8)}" y="{_f(y + bar_h - 3)}" font-size="12">{100 * acc:.0f}%</text></g>'
            )
    out.append("</svg>")
    return "".join(out)


def latency_chart(d: MatrixData) -> str:
    """p50 (solid) inside p95 (faint) per model, in seconds."""
    w, left, right, row_h = 680, 190, 40, 44
    plot_w = w - left - right
    top_val = max((s.perf.p95_ms for s in d.models), default=0) / 1000 or 1.0
    step = 10 ** math.floor(math.log10(top_val))
    for mult in (1, 2, 2.5, 5, 10):
        if top_val / (step * mult) <= 6:
            step *= mult
            break
    axis_max = math.ceil(top_val / step) * step
    h = 8 + row_h * len(d.models) + 28
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Latency p50 and p95 by model">']
    t = 0.0
    while t <= axis_max + 1e-9:
        x = left + plot_w * t / axis_max
        out.append(f'<line x1="{_f(x)}" x2="{_f(x)}" y1="4" y2="{h - 24}" class="grid"/>')
        out.append(
            f'<text x="{_f(x)}" y="{h - 8}" text-anchor="middle" font-size="11" class="t3">{t:g}s</text>'
        )
        t += step
    for i, s in enumerate(d.models):
        y = 8 + i * row_h + 8
        p50, p95 = s.perf.p50_ms / 1000, s.perf.p95_ms / 1000
        out.append(f'<text x="0" y="{_f(y + 11)}" font-size="13" font-weight="600">{esc(s.model)}</text>')
        out.append(
            f'<text x="0" y="{_f(y + 27)}" font-size="12" class="t2">p50 {p50:.1f}s, p95 {p95:.1f}s</text>'
        )
        out.append(f'<g class="cell"><title>{esc(f"{s.model}: p50 {p50:.1f}s, p95 {p95:.1f}s")}</title>')
        out.append(f'<g opacity="0.3">{_rounded_bar(left, y, plot_w * p95 / axis_max, 14, f"m{i % 6}")}</g>')
        out.append(_rounded_bar(left, y, plot_w * p50 / axis_max, 14, f"m{i % 6}"))
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def case_grid(d: MatrixData) -> str:
    """One row per model, one cell per case, grouped by category. Fail cells carry a cross."""
    cell, gap, left, group_gap, top = 13, 2, 190, 12, 34
    by_cat: dict[str, list[str]] = {}
    for cid in d.case_ids:
        by_cat.setdefault(d.case_category[cid], []).append(cid)
    cats = [c for c in d.categories if c in by_cat]
    width = left + sum(len(by_cat[c]) * (cell + gap) for c in cats) + group_gap * (len(cats) - 1) + 8
    row_h = cell + 7
    h = top + row_h * len(d.models) + 18
    out = [
        f'<svg viewBox="0 0 {width} {h}" style="min-width:{min(width, 900)}px" role="img" '
        f'aria-label="Pass or fail of every case for every model">'
    ]
    x = left
    positions: dict[str, float] = {}
    for cat in cats:
        ids = by_cat[cat]
        span = len(ids) * (cell + gap) - gap
        out.append(f'<text x="{_f(x)}" y="14" font-size="12" font-weight="600">{esc(cat)}</text>')
        out.append(f'<line x1="{_f(x)}" x2="{_f(x + span)}" y1="22" y2="22" class="axis"/>')
        for cid in ids:
            positions[cid] = x
            x += cell + gap
        x += group_gap - gap
    by_model = [{r.case_id: r for r in s.results} for s in d.models]
    for mi, s in enumerate(d.models):
        y = top + mi * row_h
        out.append(f'<text x="0" y="{_f(y + cell - 2)}" font-size="12.5">{esc(s.model)}</text>')
        for cid, cx in positions.items():
            r = by_model[mi].get(cid)
            if r is None:
                continue
            kind = "pass" if r.passed else "fail"
            why = "" if r.passed else f" ({r.reasons[0][:80]})" if r.reasons else ""
            status = "" if r.status == "ok" else f" [{r.status}]"
            out.append(
                f'<g class="cell"><title>{esc(f"{cid} {s.model}: {kind}{status}{why}")}</title>'
                f'<rect x="{_f(cx)}" y="{_f(y)}" width="{cell}" height="{cell}" rx="2" class="cell-{kind}"/>'
            )
            if not r.passed:
                out.append(
                    f'<path d="M{_f(cx + 3.5)},{_f(y + 3.5)} l{cell - 7},{cell - 7} M{_f(cx + cell - 3.5)},{_f(y + 3.5)} '
                    f'l-{cell - 7},{cell - 7}" class="cell-x"/>'
                )
            out.append("</g>")
    ybase = top + row_h * len(d.models) + 4
    for cid, cx in positions.items():  # a dot under columns where the models disagree
        outcomes = {bm[cid].passed for bm in by_model if cid in bm}
        if len(outcomes) > 1:
            out.append(
                f'<circle cx="{_f(cx + cell / 2)}" cy="{_f(ybase + 5)}" r="2.5" class="diff"><title>'
                f"{esc(cid)}: models disagree</title></circle>"
            )
    out.append("</svg>")
    return "".join(out)


def matrix_html(d: MatrixData) -> str:
    n = d.models[0].n if d.models else 0
    names = [s.model for s in d.models]
    body: list[str] = []
    best = max(d.models, key=lambda s: s.accuracy)
    tiles = [
        ("Cases per model", str(n), "one run each"),
        ("Highest accuracy", f"{100 * best.accuracy:.0f}%", best.model),
        (
            "Fastest p50",
            f"{min(s.perf.p50_ms for s in d.models) / 1000:.1f}s",
            min(d.models, key=lambda s: s.perf.p50_ms).model,
        ),
        (
            "Cost per case",
            "INR " + f"{max(s.perf.mean_cost_inr for s in d.models):.2f}",
            "highest across models; local = 0",
        ),
    ]
    body.append(
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="l">{esc(a)}</div><div class="v">{esc(b)}</div><div class="d">{esc(c)}</div></div>'
            for a, b, c in tiles
        )
        + "</div>"
    )
    body.append(
        card(
            "Overall accuracy",
            _legend(names) + '<div class="scroll">' + accuracy_chart(d) + "</div>",
            "Dot = share of cases passed. Line = 95% interval (Wilson): where the true rate plausibly lies given this many cases.",
        )
    )
    body.append(
        card(
            "Accuracy by category",
            _legend(names) + '<div class="scroll">' + category_chart(d) + "</div>",
            "Share of cases passed in each category.",
        )
    )
    body.append(
        card(
            "Latency",
            _legend(names) + '<div class="scroll">' + latency_chart(d) + "</div>",
            "Solid bar = median (p50). Faint bar = 95th percentile (p95). Wall time per case.",
        )
    )
    perf_rows = [
        [
            s.model,
            f"{s.perf.p50_ms / 1000:.1f}s",
            f"{s.perf.p95_ms / 1000:.1f}s",
            f"{s.perf.mean_prompt_tokens:.0f}",
            f"{s.perf.mean_completion_tokens:.0f}",
            f"INR {s.perf.mean_cost_inr:.2f}",
            f"INR {s.perf.total_cost_inr:.2f}",
            s.errors,
        ]
        for s in d.models
    ]
    body.append(
        card(
            "Tokens and cost",
            table(
                [
                    "model",
                    "p50",
                    "p95",
                    "prompt tok / case",
                    "completion tok / case",
                    "cost / case",
                    "total cost",
                    "errored",
                ],
                perf_rows,
                num_cols={1, 2, 3, 4, 5, 6, 7},
            ),
            "Local models cost INR 0; the field is ready for paid models.",
        )
    )
    body.append(
        card(
            "Every case, every model",
            '<div class="scroll">' + case_grid(d) + "</div>",
            "Green = passed. Red with a cross = failed. A dot underneath marks a case where the models disagree. "
            "Hover a cell for the case and the first failure reason.",
        )
    )
    if d.pairs:
        rows = []
        for c in d.pairs:
            m = c.mcnemar
            kind, word = {
                "identical": ("flat", "identical"),
                "better": ("good", "B better"),
                "worse": ("bad", "B worse"),
                "borderline": ("warn", "borderline"),
                "noise": ("flat", "within noise"),
            }[c.verdict]
            rows.append(
                [
                    c.a.model,
                    c.b.model,
                    f"{c.a_pass} / {c.b_pass}",
                    f"{m.only_b} / {m.only_a}",
                    fmt_p(m.p_value),
                    f"{points(c.boot.low)} to {points(c.boot.high)}",
                    Raw(
                        f'<span class="chip" style="padding:2px 10px;font-size:12.5px">{_icon(kind)}{esc(word)}</span>'
                    ),
                ]
            )
        body.append(
            card(
                "Are the differences real?",
                table(
                    [
                        "A",
                        "B",
                        "passed A / B",
                        "flips: B better / B worse",
                        "McNemar p",
                        "bootstrap 95% (points)",
                        "reading",
                    ],
                    rows,
                    num_cols={2, 3, 4, 5},
                ),
                "Same cases, paired. McNemar is an exact test on the flips only; a p above 0.05 means the gap could be luck. "
                "With few cases, several points of accuracy is often noise.",
            )
        )
    run_rows = [
        [
            s.model,
            s.run.run_id,
            s.run.status,
            s.run.git_hash or "-",
            s.run.prompt_hash or "-",
            s.run.code_hash or "-",
        ]
        for s in d.models
    ]
    body.append(card("Runs", table(["model", "run", "status", "git", "prompt", "code"], run_rows)))
    if d.notes:
        body.append(
            card("Notes", '<ul class="plain">' + "".join(f"<li>{esc(x)}</li>" for x in d.notes) + "</ul>")
        )
    return page(
        f"Model matrix: {d.suite}",
        f"{n} cases per model. Generated {d.generated}. Pass means the feature's own exact scoring passed.",
        "".join(body),
        d.synthetic,
    )


# ---- charts: compare ----


def _nice_extent(x: float) -> float:
    for cand in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0):
        if x <= cand:
            return cand
    return 1.0


def interval_chart(c: Comparison) -> str:
    """Bootstrap interval of accuracy(B) - accuracy(A), in points, around a zero line."""
    w, h, pad = 680, 110, 30
    ext = _nice_extent(max(abs(c.boot.low), abs(c.boot.high), abs(c.boot.diff)) * 1.15)
    zero = w / 2

    def px(v: float) -> float:
        return zero + (w / 2 - pad) * v / ext

    out = [
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Bootstrap interval of the accuracy difference">'
    ]
    for k in (-1, -0.5, 0, 0.5, 1):
        v = ext * k
        x = px(v)
        out.append(
            f'<line x1="{_f(x)}" x2="{_f(x)}" y1="18" y2="70" class="{"zero" if k == 0 else "grid"}"/>'
        )
        out.append(
            f'<text x="{_f(x)}" y="88" text-anchor="middle" font-size="11" class="t3">{(f"{v * 100:+.0f}" if v else "0")}</text>'
        )
    out.append(
        f'<text x="{_f(zero)}" y="104" text-anchor="middle" font-size="11.5" class="t2">'
        "accuracy of B minus accuracy of A, in percentage points</text>"
    )
    out.append(f'<text x="{_f(px(-ext * 0.98))}" y="12" font-size="12" class="t2">B worse</text>')
    out.append(
        f'<text x="{_f(px(ext * 0.98))}" y="12" font-size="12" text-anchor="end" class="t2">B better</text>'
    )
    lo, hi, pt = px(c.boot.low), px(c.boot.high), px(c.boot.diff)
    cy = 44
    title = (
        f"B minus A: {points(c.boot.diff)} points, 95% interval {points(c.boot.low)} to {points(c.boot.high)}"
    )
    out.append(
        f'<g class="cell"><title>{esc(title)}</title>'
        f'<line x1="{_f(lo)}" x2="{_f(hi)}" y1="{cy}" y2="{cy}" class="m0" stroke-width="2" stroke-linecap="round"/>'
        f'<line x1="{_f(lo)}" x2="{_f(lo)}" y1="{cy - 6}" y2="{cy + 6}" class="m0" stroke-width="2"/>'
        f'<line x1="{_f(hi)}" x2="{_f(hi)}" y1="{cy - 6}" y2="{cy + 6}" class="m0" stroke-width="2"/>'
        f'<circle cx="{_f(pt)}" cy="{cy}" r="6" class="m0 ring"/></g>'
    )
    out.append("</svg>")
    return "".join(out)


def flip_bar(c: Comparison) -> str:
    """Segmented bar: cases that got worse vs cases that got better, out of all flips."""
    m = c.mcnemar
    if m.flips == 0:
        return '<p class="hint" style="margin:0">No case flipped.</p>'
    w, h, left = 680, 62, 0
    worse_w = (w - 2) * m.only_a / m.flips
    better_w = (w - 2) * m.only_b / m.flips
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Flips: {m.only_a} worse, {m.only_b} better">']
    if m.only_a:
        out.append(
            f'<g class="cell"><title>{m.only_a} case(s) passed in A and failed in B</title>'
            f'<rect x="{left}" y="6" width="{_f(worse_w)}" height="12" rx="6" style="fill:var(--bad)"/></g>'
        )
    if m.only_b:
        out.append(
            f'<g class="cell"><title>{m.only_b} case(s) failed in A and passed in B</title>'
            f'<rect x="{_f(left + worse_w + 2)}" y="6" width="{_f(better_w)}" height="12" rx="6" style="fill:var(--good)"/></g>'
        )
    out.append(f'<text x="0" y="40" font-size="13" font-weight="600">{m.only_a} got worse</text>')
    out.append('<text x="0" y="56" font-size="12" class="t2">pass in A, fail in B</text>')
    out.append(
        f'<text x="{w}" y="40" font-size="13" font-weight="600" text-anchor="end">{m.only_b} got better</text>'
    )
    out.append(
        f'<text x="{w}" y="56" font-size="12" text-anchor="end" class="t2">fail in A, pass in B</text>'
    )
    out.append("</svg>")
    return "".join(out)


def category_delta_chart(c: Comparison) -> str:
    """Per category: change in passed cases (B minus A) as a diverging bar."""
    if not c.categories:
        return ""
    w, left, row_h, right = 680, 150, 34, 70
    ext = max(1, max(abs(d.delta) for d in c.categories))
    mid = left + (w - left - right) / 2
    half = (w - left - right) / 2 - 8
    h = 8 + row_h * len(c.categories) + 14
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Change in passed cases by category">']
    out.append(f'<line x1="{_f(mid)}" x2="{_f(mid)}" y1="2" y2="{h - 10}" class="zero"/>')
    for i, d in enumerate(c.categories):
        y = 8 + i * row_h
        bw = half * abs(d.delta) / ext
        out.append(f'<text x="0" y="{_f(y + 11)}" font-size="13" font-weight="600">{esc(d.category)}</text>')
        out.append(f'<text x="0" y="{_f(y + 26)}" font-size="11.5" class="t2">{d.n} cases</text>')
        title = f"{d.category}: A {d.a_pass}/{d.n}, B {d.b_pass}/{d.n}, change {d.delta:+d}, p = {fmt_p(d.mcnemar.p_value)}"
        out.append(f'<g class="cell"><title>{esc(title)}</title>')
        out.append(
            f'<rect x="{left}" y="{y}" width="{w - left - right}" height="18" fill="none" stroke="none"/>'
        )
        if d.delta > 0:
            out.append(
                f'<rect x="{_f(mid)}" y="{y + 2}" width="{_f(bw)}" height="14" rx="3" class="nostroke" style="fill:var(--pos)"/>'
            )
            out.append(f'<text x="{_f(mid + bw + 8)}" y="{y + 14}" font-size="12">{d.delta:+d}</text>')
        elif d.delta < 0:
            out.append(
                f'<rect x="{_f(mid - bw)}" y="{y + 2}" width="{_f(bw)}" height="14" rx="3" class="nostroke" style="fill:var(--neg)"/>'
            )
            out.append(
                f'<text x="{_f(mid - bw - 8)}" y="{y + 14}" font-size="12" text-anchor="end">{d.delta:+d}</text>'
            )
        else:
            out.append(f'<text x="{_f(mid + 8)}" y="{y + 14}" font-size="12" class="t2">no change</text>')
        out.append("</g>")
    out.append("</svg>")
    return "".join(out)


def compare_html(c: Comparison, noise_floor: bool = False) -> str:
    m = c.mcnemar
    kind = {"identical": "flat", "better": "good", "worse": "bad", "borderline": "warn", "noise": "flat"}[
        c.verdict
    ]
    chip_cls = {"flat": "", "good": "good", "bad": "bad", "warn": "warn"}[kind]
    body = [f'<p><span class="chip {chip_cls}">{_icon(kind)}{esc(VERDICT_TEXT[c.verdict])}</span></p>']
    tiles = [
        ("A", pct(c.a_pass, c.n), f"{c.a.model}"),
        ("B", pct(c.b_pass, c.n), f"{c.b.model}"),
        (
            "Change (B minus A)",
            (f"{c.net_cases:+d}" if c.net_cases else "0") + " case(s)",
            f"{points(c.boot.diff)} points",
        ),
        ("McNemar p", fmt_p(m.p_value), f"{m.flips} flips"),
    ]
    body.append(
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="l">{esc(a)}</div><div class="v">{esc(b)}</div><div class="d">{esc(x)}</div></div>'
            for a, b, x in tiles
        )
        + "</div>"
    )
    body.append(
        card(
            "Is the difference real?",
            '<div class="scroll">' + interval_chart(c) + "</div>",
            f"Bootstrap 95% interval: resample the {c.n} cases {c.boot.n_boot:,} times. "
            "If the interval crosses the zero line, the difference could be nothing.",
        )
    )
    body.append(
        card(
            "Flips",
            '<div class="scroll">' + flip_bar(c) + "</div>",
            "Only cases that changed result carry information. Equal numbers each way is what noise looks like.",
        )
    )
    if c.categories:
        rows = []
        for d in c.categories:
            mm = d.mcnemar
            word = (
                "no flips" if mm.flips == 0 else ("significant" if mm.significant else "weak: within noise")
            )
            rows.append(
                [
                    d.category,
                    d.n,
                    pct(d.a_pass, d.n),
                    pct(d.b_pass, d.n),
                    (f"{d.delta:+d}" if d.delta else "0"),
                    f"{mm.only_b} / {mm.only_a}",
                    fmt_p(mm.p_value),
                    word,
                ]
            )
        body.append(
            card(
                "By category",
                '<div class="scroll">'
                + category_delta_chart(c)
                + "</div>"
                + table(
                    ["category", "cases", "A", "B", "change", "better / worse", "p", "reading"],
                    rows,
                    num_cols={1, 4, 5, 6},
                ),
                "Small groups: read the p-values as hints.",
            )
        )
    if c.flips:
        rows = []
        for f in sorted(c.flips, key=lambda f: (f.direction != "regressed", f.case_id)):
            word = Raw(
                f'<span class="pill {"bad" if f.direction == "regressed" else "good"}">'
                f"{'pass to fail' if f.direction == 'regressed' else 'fail to pass'}</span>"
            )
            why = "; ".join(f.b_reasons if f.direction == "regressed" else f.a_reasons) or "-"
            rows.append([f.case_id, f.category, word, why[:160]])
        body.append(
            card(
                "Flipped cases",
                table(["case", "category", "direction", "why it failed"], rows),
                "Worse first.",
            )
        )
    body.append(
        card(
            "In plain words",
            '<ul class="plain">' + "".join(f"<li>{esc(x)}</li>" for x in explain(c)) + "</ul>",
        )
    )
    if noise_floor:
        body.append(
            card(
                "Noise floor",
                '<ul class="plain">' + "".join(f"<li>{esc(x)}</li>" for x in noise_floor_lines(c)) + "</ul>",
            )
        )
    pa, pb = c.a_perf, c.b_perf

    def rel(a: float, b: float) -> str:
        return f"{(b - a) / a * 100:+.0f}%" if a else "-"

    perf = [
        ["p50 latency", f"{pa.p50_ms / 1000:.1f}s", f"{pb.p50_ms / 1000:.1f}s", rel(pa.p50_ms, pb.p50_ms)],
        ["p95 latency", f"{pa.p95_ms / 1000:.1f}s", f"{pb.p95_ms / 1000:.1f}s", rel(pa.p95_ms, pb.p95_ms)],
        [
            "prompt tokens / case",
            f"{pa.mean_prompt_tokens:.0f}",
            f"{pb.mean_prompt_tokens:.0f}",
            rel(pa.mean_prompt_tokens, pb.mean_prompt_tokens),
        ],
        [
            "completion tokens / case",
            f"{pa.mean_completion_tokens:.0f}",
            f"{pb.mean_completion_tokens:.0f}",
            rel(pa.mean_completion_tokens, pb.mean_completion_tokens),
        ],
        [
            "cost / case",
            f"INR {pa.mean_cost_inr:.2f}",
            f"INR {pb.mean_cost_inr:.2f}",
            f"{pb.mean_cost_inr - pa.mean_cost_inr:+.2f}",
        ],
    ]
    body.append(
        card(
            "Latency, tokens, cost",
            table(["", "A", "B", "change"], perf, num_cols={1, 2, 3}),
            "Wall time per case measured by the harness, over cases that completed.",
        )
    )
    runs = [
        [t, r.run_id, r.suite, r.model, r.git_hash or "-", r.prompt_hash or "-", r.code_hash or "-"]
        for t, r in (("A", c.a), ("B", c.b))
    ]
    body.append(card("Runs", table(["", "run", "suite", "model", "git", "prompt", "code"], runs)))
    if c.notes:
        body.append(
            card("Notes", '<ul class="plain">' + "".join(f"<li>{esc(x)}</li>" for x in c.notes) + "</ul>")
        )
    return page(
        "Run comparison",
        f"{c.a.run_id} (A) against {c.b.run_id} (B). B is the newer or candidate run.",
        "".join(body),
        c.synthetic,
    )
