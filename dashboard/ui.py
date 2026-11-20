"""Shared look for the dashboard: page setup, header, KPI cards, empty states, footer.

Author: Sharbesh S (https://github.com/SharbeshS8199). MIT License.
"""

from __future__ import annotations

import streamlit as st

AUTHOR, AUTHOR_URL = "Sharbesh S", "https://github.com/SharbeshS8199"
COLORS = ["#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
GOOD, WARN, BAD, MUTED = "#16a34a", "#f59e0b", "#dc2626", "#64748b"

_CSS = """
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1280px;}
  .hdr {display:flex; justify-content:space-between; align-items:flex-end; border-bottom:1px solid rgba(100,116,139,.25);
        padding-bottom:.6rem; margin-bottom:1.1rem;}
  .hdr h1 {font-size:1.75rem; margin:0; line-height:1.2;}
  .hdr p {margin:.25rem 0 0; color:#64748b; font-size:.95rem;}
  .hdr .by {font-size:.8rem; color:#64748b; text-align:right; white-space:nowrap;}
  .kpi {border:1px solid rgba(100,116,139,.22); border-radius:12px; padding:.75rem .9rem; height:100%;
        background:rgba(148,163,184,.06);}
  .kpi .l {font-size:.78rem; color:#64748b; text-transform:uppercase; letter-spacing:.04em;}
  .kpi .v {font-size:1.55rem; font-weight:650; margin-top:.15rem;}
  .kpi .s {font-size:.8rem; color:#64748b; margin-top:.1rem;}
  .pill {display:inline-block; padding:.1rem .55rem; border-radius:99px; font-size:.78rem; font-weight:600; color:#fff;}
  .empty {border:1px dashed rgba(100,116,139,.45); border-radius:12px; padding:1.2rem 1.3rem; margin:.4rem 0 1rem;}
  .empty h4 {margin:.1rem 0 .4rem;}
  .foot {margin-top:2.2rem; padding-top:.7rem; border-top:1px solid rgba(100,116,139,.2); color:#64748b; font-size:.8rem;}
</style>
"""


def page(title: str, subtitle: str, icon: str = "📊", wide: bool = True) -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide" if wide else "centered")
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="hdr"><div><h1>{icon} {title}</h1><p>{subtitle}</p></div>'
                f'<div class="by">by <a href="{AUTHOR_URL}">{AUTHOR}</a></div></div>', unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str]]) -> None:
    """items: (label, value, small subtext)."""
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                     f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def pill(text: str, color: str = GOOD) -> str:
    return f'<span class="pill" style="background:{color}">{text}</span>'


def empty_state(title: str, body: str, command: str | None = None) -> None:
    st.markdown(f'<div class="empty"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)
    if command:
        st.code(command, language="bash")


def style_fig(fig, height: int = 360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), template="plotly_white",
                      colorway=COLORS, legend=dict(orientation="h", y=-0.18), font=dict(size=12))
    return fig


def footer(extra: str = "") -> None:
    st.markdown(f'<div class="foot">Built by <a href="{AUTHOR_URL}">{AUTHOR}</a> · MIT License'
                f'{" · " + extra if extra else ""}</div>', unsafe_allow_html=True)
