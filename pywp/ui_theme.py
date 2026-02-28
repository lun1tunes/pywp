from __future__ import annotations

import streamlit as st


def apply_page_style(max_width_px: int = 1680) -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: radial-gradient(1200px 500px at 20% -20%, #E8F1FF 0%, #F8FAFD 45%, #FAFCFF 100%);
        }}
        .block-container {{
            max-width: {int(max_width_px)}px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }}
        .pywp-hero {{
            border: 1px solid #D6E4FF;
            background: linear-gradient(135deg, #F1F7FF 0%, #FFFFFF 60%);
            border-radius: 16px;
            padding: 1.0rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 8px 24px rgba(15, 33, 66, 0.06);
        }}
        .pywp-hero h2 {{
            margin: 0 0 0.25rem 0;
            font-size: 1.55rem;
            letter-spacing: 0.01em;
            color: #12355B;
        }}
        .pywp-hero p {{
            margin: 0;
            color: #355070;
            font-size: 0.97rem;
        }}
        .pywp-small-note {{
            color: #54657D;
            font-size: 0.87rem;
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
