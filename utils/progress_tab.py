"""
utils/progress_tab.py
──────────────────────────────────────────────────────────────────────────────
Renders the full '📈 Progress' tab in StudyPal.
Call render_progress_tab() inside `with tab_progress:` in app.py.

Sections:
  1. Overview cards  — streak, total quizzes, avg score, subjects
  2. Score trend     — line chart of quiz scores over time
  3. Subject radar   — radar chart of avg score per subject
  4. Weak topics     — flagged subjects needing review
  5. Recent activity — quiz history + question history tables
"""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from utils.progress_tracker import (
    get_summary_stats,
    get_subject_stats,
    get_score_trend,
    get_quiz_history,
    get_question_history,
    get_weak_topics,
    clear_all_progress,
)

# ── Colour palette (matches StudyPal theme) ───────────────────────────────────
GOLD   = "#f0c040"
TEAL   = "#4ecca3"
RED    = "#e05c5c"
MUTED  = "#7a8099"
BG     = "#0d0f14"
SURF   = "#161922"
SURF2  = "#1e2230"
BORDER = "#2a2f3e"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#e8eaf0", size=12),
    margin=dict(l=20, r=20, t=30, b=20),
)


# ── Small helpers ─────────────────────────────────────────────────────────────
def _stat_card(icon: str, label: str, value: str, color: str = GOLD, note: str = "") -> str:
    return f"""
<div style="background:{SURF};border:1px solid {BORDER};border-radius:14px;
            padding:1.2rem 1.4rem;text-align:center">
  <div style="font-size:1.6rem;margin-bottom:.3rem">{icon}</div>
  <div style="font-size:2rem;font-weight:800;font-family:'Syne',sans-serif;color:{color}">{value}</div>
  <div style="font-size:.8rem;color:{MUTED};margin-top:.2rem">{label}</div>
  {"" if not note else f'<div style="font-size:.72rem;color:{TEAL};margin-top:.2rem">{note}</div>'}
</div>"""


def _score_color(pct: float) -> str:
    if pct >= 80: return TEAL
    if pct >= 50: return GOLD
    return RED


# ── Main renderer ─────────────────────────────────────────────────────────────
def render_progress_tab() -> None:

    stats        = get_summary_stats()
    subject_data = get_subject_stats()
    trend_data   = get_score_trend(limit=30)
    quiz_hist    = get_quiz_history(limit=20)
    q_hist       = get_question_history(limit=20)
    weak         = get_weak_topics()

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
<div style="background:linear-gradient(135deg,#161922,#1a1e2e);
            border:1px solid rgba(240,192,64,.3);border-radius:16px;
            padding:1.4rem 1.8rem;margin-bottom:1.5rem;position:relative;overflow:hidden">
  <div style="position:absolute;inset:0;background:radial-gradient(ellipse 50% 60% at 90% 50%,
       rgba(240,192,64,.07),transparent);pointer-events:none"></div>
  <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.2rem;margin-bottom:.3rem">
    📈 Student Progress Dashboard
  </div>
  <div style="font-size:.85rem;color:#9aa0b4">
    Track your learning journey — quiz scores, study habits, weak spots, and streaks.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── No data state ─────────────────────────────────────────────────────────
    if stats["total_quizzes"] == 0 and stats["total_questions"] == 0:
        st.markdown("""
<div style="text-align:center;padding:3rem;color:#7a8099">
  <div style="font-size:3rem;margin-bottom:.8rem">📊</div>
  <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:600">No data yet</div>
  <div style="font-size:.88rem;margin-top:.4rem">
    Ask questions in <strong>Ask & Explain</strong> and complete quizzes in
    <strong>Quiz Mode</strong> — your progress will appear here automatically.
  </div>
</div>""", unsafe_allow_html=True)
        return

    # ══════════════════════════════════════════════════════════════════════════
    # 1. Overview cards
    # ══════════════════════════════════════════════════════════════════════════
    c1, c2, c3, c4, c5 = st.columns(5)
    cards = [
        ("🔥", "Day Streak",       str(stats["streak"]),          GOLD,  "Keep it up!"),
        ("🧠", "Quizzes Taken",    str(stats["total_quizzes"]),   TEAL,  ""),
        ("💬", "Questions Asked",  str(stats["total_questions"]), TEAL,  ""),
        ("🎯", "Avg Quiz Score",   f"{stats['avg_score']}%",
         _score_color(stats["avg_score"]), ""),
        ("📚", "Subjects Studied", str(stats["subjects_studied"]), GOLD, ""),
    ]
    for col, (icon, label, val, color, note) in zip([c1,c2,c3,c4,c5], cards):
        col.markdown(_stat_card(icon, label, val, color, note), unsafe_allow_html=True)

    # ── Weak topic alert ──────────────────────────────────────────────────────
    if weak:
        st.markdown(
            f'<div style="background:rgba(224,92,92,.08);border:1px solid rgba(224,92,92,.3);'
            f'border-radius:10px;padding:.8rem 1.2rem;margin-top:1rem;font-size:.88rem;">'
            f'⚠️ <strong>Needs review:</strong> '
            + ", ".join(f'<span style="color:{RED}">{w}</span>' for w in weak)
            + ' — avg score below 60%</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Score trend line chart
    # ══════════════════════════════════════════════════════════════════════════
    if trend_data:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:.95rem;'
            'margin-bottom:.6rem">📉 Quiz Score Trend</div>',
            unsafe_allow_html=True,
        )
        df_trend = pd.DataFrame(trend_data)
        df_trend["ts_fmt"] = pd.to_datetime(df_trend["ts"]).dt.strftime("%d %b %H:%M")

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=df_trend["ts_fmt"],
            y=df_trend["pct"],
            mode="lines+markers",
            line=dict(color=GOLD, width=2.5),
            marker=dict(size=7, color=[_score_color(p) for p in df_trend["pct"]],
                        line=dict(width=1.5, color=SURF2)),
            text=df_trend["subject"],
            hovertemplate="<b>%{text}</b><br>Score: %{y:.1f}%<br>%{x}<extra></extra>",
            fill="tozeroy",
            fillcolor="rgba(240,192,64,0.06)",
        ))
        fig_trend.add_hline(y=60, line_dash="dash", line_color=RED,
                            annotation_text="60% threshold",
                            annotation_font_color=RED, annotation_font_size=11)
        fig_trend.add_hline(y=80, line_dash="dot", line_color=TEAL,
                            annotation_text="80% target",
                            annotation_font_color=TEAL, annotation_font_size=11)
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            height=280,
            yaxis=dict(range=[0, 105], gridcolor=BORDER, zerolinecolor=BORDER,
                       ticksuffix="%"),
            showlegend=False,
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Subject breakdown — bar + radar side by side
    # ══════════════════════════════════════════════════════════════════════════
    if subject_data:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:.95rem;'
            'margin-bottom:.6rem">📚 Performance by Subject</div>',
            unsafe_allow_html=True,
        )

        col_bar, col_radar = st.columns(2)

        subjects   = list(subject_data.keys())
        avg_scores = [subject_data[s]["avg_score"] for s in subjects]
        bar_colors = [_score_color(v) for v in avg_scores]

        # Bar chart
        with col_bar:
            fig_bar = go.Figure(go.Bar(
                x=subjects,
                y=avg_scores,
                marker=dict(color=bar_colors,
                            line=dict(color=SURF2, width=1.5)),
                text=[f"{v}%" for v in avg_scores],
                textposition="outside",
                textfont=dict(color="#e8eaf0", size=11),
                hovertemplate="<b>%{x}</b><br>Avg: %{y:.1f}%<extra></extra>",
            ))
            fig_bar.update_layout(
                **PLOTLY_LAYOUT,
                height=280,
                yaxis=dict(range=[0, 115], ticksuffix="%",
                           gridcolor=BORDER, zerolinecolor=BORDER),
                xaxis=dict(gridcolor=BORDER),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart (needs ≥ 3 subjects)
        with col_radar:
            if len(subjects) >= 3:
                fig_radar = go.Figure(go.Scatterpolar(
                    r=avg_scores + [avg_scores[0]],
                    theta=subjects + [subjects[0]],
                    fill="toself",
                    fillcolor="rgba(240,192,64,0.1)",
                    line=dict(color=GOLD, width=2),
                    marker=dict(size=6, color=GOLD),
                ))
                fig_radar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="DM Sans", color="#e8eaf0", size=11),
                    margin=dict(l=30, r=30, t=30, b=30),
                    height=280,
                    polar=dict(
                        bgcolor=SURF2,
                        radialaxis=dict(
                            visible=True, range=[0, 100],
                            gridcolor=BORDER, linecolor=BORDER,
                            ticksuffix="%", tickfont=dict(size=9),
                        ),
                        angularaxis=dict(gridcolor=BORDER, linecolor=BORDER),
                    ),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.markdown(
                    '<div style="height:280px;display:flex;align-items:center;'
                    'justify-content:center;color:#7a8099;font-size:.85rem;'
                    'background:#161922;border:1px solid #2a2f3e;border-radius:12px">'
                    'Study 3+ subjects to unlock the radar chart</div>',
                    unsafe_allow_html=True,
                )

        # Subject detail table
        with st.expander("📋 Subject Detail Table"):
            rows = []
            for s, v in subject_data.items():
                rows.append({
                    "Subject":   s,
                    "Attempts":  v["attempts"],
                    "Avg Score": f"{v['avg_score']}%",
                    "Best":      f"{v['best']}%",
                    "Worst":     f"{v['worst']}%",
                    "Last Quiz": v["last_ts"][:10] if v["last_ts"] else "—",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Recent activity
    # ══════════════════════════════════════════════════════════════════════════
    act_col1, act_col2 = st.columns(2)

    with act_col1:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:.9rem;'
            'margin-bottom:.6rem">🧠 Recent Quizzes</div>',
            unsafe_allow_html=True,
        )
        if quiz_hist:
            for q in quiz_hist[:8]:
                pct   = q["pct"]
                color = _score_color(pct)
                ts    = q["ts"][:16].replace("T", " ")
                st.markdown(
                    f'<div style="background:{SURF};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:.7rem 1rem;margin-bottom:.4rem;'
                    f'display:flex;justify-content:space-between;align-items:center">'
                    f'<div>'
                    f'  <div style="font-size:.88rem;font-weight:600">{q["subject"]}</div>'
                    f'  <div style="font-size:.75rem;color:{MUTED}">{ts}</div>'
                    f'</div>'
                    f'<div style="font-family:Syne,sans-serif;font-weight:800;'
                    f'font-size:1.1rem;color:{color}">{q["score"]}/{q["total"]}'
                    f'<span style="font-size:.75rem;color:{MUTED};margin-left:4px">'
                    f'({pct:.0f}%)</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No quizzes completed yet.")

    with act_col2:
        st.markdown(
            '<div style="font-family:Syne,sans-serif;font-weight:700;font-size:.9rem;'
            'margin-bottom:.6rem">💬 Recent Questions</div>',
            unsafe_allow_html=True,
        )
        if q_hist:
            for q in q_hist[:8]:
                ts = q["ts"][:16].replace("T", " ")
                st.markdown(
                    f'<div style="background:{SURF};border:1px solid {BORDER};'
                    f'border-radius:10px;padding:.7rem 1rem;margin-bottom:.4rem">'
                    f'<div style="font-size:.82rem;color:{MUTED};margin-bottom:.2rem">'
                    f'{q["subject"]} · {ts}</div>'
                    f'<div style="font-size:.85rem">{q["text"][:120]}'
                    f'{"…" if len(q["text"]) > 120 else ""}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No questions asked yet.")

    # ── Clear button ──────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("⚙️ Data Management"):
        st.warning("This will permanently delete all progress data.")
        if st.button("🗑 Clear All Progress Data", type="primary"):
            clear_all_progress()
            st.success("All progress data cleared.")
            st.rerun()