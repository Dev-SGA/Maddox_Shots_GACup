import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import pandas as pd
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
import numpy as np
from PIL import Image
from matplotlib.lines import Line2D

# ==========================
# Page Configuration
# ==========================
st.set_page_config(layout="wide", page_title="Shot Map Analysis")

st.title("Shot Map Analysis - Multiple Matches")
st.caption("Click on the icons on the pitch to play the corresponding video analysis and see goal placement.")

# ==========================
# GOAL DIMENSIONS
# ==========================
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44

# ==========================
# Data Setup (agora com goal_x / goal_y)
# type, x, y, xg, goal_x, goal_y, video
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("A gol", 109.70, 23.88, 0.18, 5.32, 0.19, "1 - LA.mp4"),
    ],
    "Vs Slavia Praha": [
        ("Fora", 104.55, 23.88, 0.07, None, None, "1 - SP.mp4"),
    ],
    "Vs Sockers": [
        ("A gol", 94.91, 30.52, 0.10, 0.86, 0.51, "GOL - SK.mp4"),
        ("Gol", 108.04, 43.16, 0.35, 2.22, 0.11, "2 - SK.mp4"),
    ],
}

# Create DataFrames for each match and combined
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(
        events,
        columns=["type", "x", "y", "xg", "goal_x", "goal_y", "video"]
    )

# All games combined
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All shots": df_all}
full_data.update(dfs_by_match)

# ==========================
# Style (parecido com seu shotmap)
# ==========================
def get_style(result_type: str, has_video: bool):
    t = (result_type or "").strip().upper()
    alpha = 0.95 if has_video else 0.85

    if t == "GOL":
        return "*", (239/255, 71/255, 111/255, alpha), 1.5  # #EF476F
    if t in ("A GOL", "NO ALVO"):
        return "h", (6/255, 214/255, 160/255, alpha), 1.5   # #06D6A0
    if t == "FORA":
        return "o", (255/255, 209/255, 102/255, alpha), 1.5 # #FFD166
    if t == "BLOQUEADO":
        return "s", (17/255, 138/255, 178/255, alpha), 1.5  # #118AB2

    return "o", (0.6, 0.6, 0.6, alpha), 1.2

def size_from_xg(xg: float, scale: float = 1400.0):
    return (float(xg) * scale) + 60

# ==========================
# Goal chart (mesmo visual do seu código)
# ==========================
def draw_goal(selected_event: pd.Series | None):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    # goal frame
    ax.plot([0, GOAL_WIDTH], [GOAL_HEIGHT, GOAL_HEIGHT], color="white", lw=3)
    ax.plot([0, 0], [0, GOAL_HEIGHT], color="white", lw=3)
    ax.plot([GOAL_WIDTH, GOAL_WIDTH], [0, GOAL_HEIGHT], color="white", lw=3)

    # 3x3 grid
    x1 = GOAL_WIDTH / 3
    x2 = 2 * GOAL_WIDTH / 3
    y1 = GOAL_HEIGHT / 3
    y2 = 2 * GOAL_HEIGHT / 3
    ax.plot([x1, x1], [0, GOAL_HEIGHT], color="white", alpha=0.2)
    ax.plot([x2, x2], [0, GOAL_HEIGHT], color="white", alpha=0.2)
    ax.plot([0, GOAL_WIDTH], [y1, y1], color="white", alpha=0.2)
    ax.plot([0, GOAL_WIDTH], [y2, y2], color="white", alpha=0.2)

    # limits/axes
    ax.set_xlim(-0.5, GOAL_WIDTH + 0.5)
    ax.set_ylim(0, GOAL_HEIGHT + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Goal View (Shot Placement)", color="white")

    # plot selected event shot placement
    if selected_event is not None:
        gx = selected_event.get("goal_x")
        gy = selected_event.get("goal_y")

        if pd.notna(gx) and pd.notna(gy):
            t = (selected_event.get("type") or "").strip().upper()
            if t == "GOL":
                c, m = "#EF476F", "*"
            elif t in ("A GOL", "NO ALVO"):
                c, m = "#06D6A0", "h"
            elif t == "FORA":
                c, m = "#FFD166", "o"
            elif t == "BLOQUEADO":
                c, m = "#118AB2", "s"
            else:
                c, m = "#999999", "o"

            ax.scatter(
                gx, gy,
                color=c,
                marker=m,
                s=120,
                edgecolors="white",
                linewidth=1.5,
                zorder=3
            )
        else:
            # sem coords do gol: não plota nada
            pass

    plt.tight_layout()
    return fig

# ==========================
# Sidebar Configuration
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

df_base = full_data[selected_match].copy()
result_options = sorted(df_base["type"].unique().tolist())
selected_results = st.sidebar.multiselect("Shot Result", result_options, default=result_options)

st.sidebar.divider()
st.sidebar.caption("Match filtered by selected options above")

df = df_base[df_base["type"].isin(selected_results)].copy()

# ==========================
# Main Layout
# ==========================
col_map, col_panel = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Shot Map")

    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0e0e0e", line_color="#e0e0e0")
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in df.iterrows():
        has_vid = row["video"] is not None
        marker, color, lw = get_style(row["type"], has_vid)

        pitch.scatter(
            row.x, row.y,
            marker=marker,
            s=size_from_xg(row["xg"]),
            color=color,
            edgecolors="#ffffff",
            linewidths=lw,
            ax=ax,
            zorder=3
        )

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='none', label='Gol',
               markerfacecolor="#EF476F", markeredgecolor="#ffffff", markersize=11),
        Line2D([0], [0], marker='h', color='none', label='A gol / No alvo',
               markerfacecolor="#06D6A0", markeredgecolor="#ffffff", markersize=9),
        Line2D([0], [0], marker='o', color='none', label='Fora',
               markerfacecolor="#FFD166", markeredgecolor="#ffffff", markersize=9),
        Line2D([0], [0], marker='s', color='none', label='Bloqueado',
               markerfacecolor="#118AB2", markeredgecolor="#ffffff", markersize=9),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#111111",
        edgecolor="#444444",
        fontsize="small",
        title="Shot Events",
        title_fontsize="medium",
        labelspacing=1.0,
        borderpad=0.8,
        framealpha=0.95,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_title().set_color("#eaeaea")
    for text in legend.get_texts():
        text.set_color("#eaeaea")

    # Convert plot to image for coordinate tracking (IGUAL ao seu Duel Map)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)
    click = streamlit_image_coordinates(img_obj, width=700)

# ==========================
# Interaction Logic (IGUAL ao seu Duel Map)
# ==========================
selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    mpl_pixel_y = real_h - pixel_y
    coords = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    df["dist"] = np.sqrt((df["x"] - field_x) ** 2 + (df["y"] - field_y) ** 2)

    RADIUS = 5
    candidates = df[df["dist"] < RADIUS]
    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Panel: Video + Goal chart
# ==========================
with col_panel:
    st.subheader("Event Details")

    if selected_event is not None:
        st.success(f"**Selected Event:** {selected_event['type']}")
        st.info(f"**Position:** X: {selected_event['x']:.2f}, Y: {selected_event['y']:.2f}")
        st.write(f"**xG:** {selected_event['xg']:.2f}")

        if selected_event["video"]:
            try:
                st.video(selected_event["video"])
            except Exception:
                st.error(f"Video file not found: {selected_event['video']}")
        else:
            st.warning("Não há vídeo carregado para este evento.")

        st.divider()
        st.subheader("Shot Placement (Goal View)")

        # Se não tem coords de gol, avisa e mostra o gol vazio mesmo assim (opcional)
        if pd.isna(selected_event.get("goal_x")) or pd.isna(selected_event.get("goal_y")):
            st.warning("Este evento não possui coordenadas do gol (goal_x/goal_y).")

        goal_fig = draw_goal(selected_event)
        st.pyplot(goal_fig, clear_figure=True)

    else:
        st.info("Select a marker on the pitch to view event details.")
        st.divider()
        st.subheader("Shot Placement (Goal View)")
        goal_fig = draw_goal(None)
        st.pyplot(goal_fig, clear_figure=True)
