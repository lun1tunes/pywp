from __future__ import annotations

from html import escape

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
            --pywp-hero-content-max-width: 100%;
            border: 1px solid #D7E2F1;
            background: linear-gradient(135deg, #FFFFFF 0%, #F9FBFF 100%);
            border-radius: 18px;
            padding: 1.05rem 1.25rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 24px rgba(18, 48, 84, 0.08);
        }}
        .pywp-hero__inner {{
            max-width: var(--pywp-hero-content-max-width);
        }}
        .pywp-hero--centered {{
            padding: 1.1rem 1.4rem 1rem;
            text-align: center;
        }}
        .pywp-hero--centered .pywp-hero__inner {{
            max-width: min(var(--pywp-hero-content-max-width), 100%);
            margin-inline: auto;
        }}
        .pywp-hero h2 {{
            margin: 0 0 0.25rem 0;
            font-size: 1.55rem;
            color: #111111;
        }}
        .pywp-hero--centered h2 {{
            font-size: 1.75rem;
            letter-spacing: 0.018em;
        }}
        .pywp-hero p {{
            margin: 0;
            color: #111111;
            font-size: 0.97rem;
        }}
        .pywp-hero--centered p {{
            max-width: 30rem;
            margin: 0 auto;
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


def render_hero(
    title: str,
    subtitle: str = "",
    *,
    centered: bool = False,
    max_content_width_px: int | None = None,
) -> None:
    subtitle_html = f"<p>{escape(subtitle)}</p>" if subtitle else ""
    hero_classes = ["pywp-hero"]
    if centered:
        hero_classes.append("pywp-hero--centered")
    style_attr = ""
    if max_content_width_px is not None:
        style_attr = (
            f' style="--pywp-hero-content-max-width: {int(max_content_width_px)}px;"'
        )
    st.markdown(
        f"""
        <div class="{' '.join(hero_classes)}"{style_attr}>
          <div class="pywp-hero__inner">
            <h2>{escape(title)}</h2>
            {subtitle_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_small_note(text: str) -> None:
    st.markdown(f"<div class='pywp-small-note'>{text}</div>", unsafe_allow_html=True)
