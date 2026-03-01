from __future__ import annotations

import streamlit as st


def apply_page_style(max_width_px: int = 1680) -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --pywp-accent: #0D6E6E;
            --pywp-accent-soft: #E8F6F6;
            --pywp-text: #1A2B3C;
            --pywp-muted: #4D6175;
            --pywp-border: #D8E2EE;
        }}
        .stApp {{
            background: radial-gradient(1200px 500px at 12% -12%, #E8F6F6 0%, #F8FAFD 48%, #FAFCFF 100%);
            color: var(--pywp-text);
            font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }}
        .block-container {{
            max-width: {int(max_width_px)}px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
        h1, h2, h3, h4 {{
            letter-spacing: 0.01em;
            color: #17324D;
        }}
        .pywp-hero {{
            border: 1px solid #CCE7E7;
            background: linear-gradient(135deg, #EAFBFB 0%, #FFFFFF 62%);
            border-radius: 18px;
            padding: 1.05rem 1.25rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 26px rgba(22, 45, 68, 0.08);
        }}
        .pywp-hero h2 {{
            margin: 0 0 0.25rem 0;
            font-size: 1.55rem;
            color: #0E3D52;
        }}
        .pywp-hero p {{
            margin: 0;
            color: #2F556A;
            font-size: 0.97rem;
        }}
        .pywp-small-note {{
            color: var(--pywp-muted);
            font-size: 0.87rem;
        }}
        div[data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid var(--pywp-border);
            border-radius: 12px;
            padding: 0.5rem 0.7rem;
        }}
        div[data-testid="stMetricLabel"] {{
            color: #3B566F;
        }}
        div[data-testid="stMetricValue"] {{
            color: #10324A;
        }}
        div[data-testid="stForm"] {{
            border-radius: 14px;
        }}
        div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stStatusWidget"]) {{
            border-radius: 12px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str = "") -> None:
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
        <div class="pywp-hero">
          <h2>{title}</h2>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_small_note(text: str) -> None:
    st.markdown(f"<div class='pywp-small-note'>{text}</div>", unsafe_allow_html=True)
