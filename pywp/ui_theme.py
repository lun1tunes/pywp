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
            --pywp-page-bg: radial-gradient(1200px 500px at 12% -12%, #E8F6F6 0%, #F8FAFD 48%, #FAFCFF 100%);
            --pywp-header-bg: #FAFCFF;
        }}
        .stApp {{
            background: var(--pywp-page-bg);
            color: var(--pywp-text);
            font-family: "IBM Plex Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif;
        }}
        header[data-testid="stHeader"] {{
            background: var(--pywp-header-bg);
        }}
        div[data-testid="stToolbar"] {{
            background: transparent;
        }}
        div[data-testid="stDecoration"] {{
            background: transparent;
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


def render_small_note(text: str) -> None:
    st.markdown(f"<div class='pywp-small-note'>{text}</div>", unsafe_allow_html=True)
