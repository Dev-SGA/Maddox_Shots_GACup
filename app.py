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
st.caption("Click on the icons on the pitch to play the corresponding video analysis.")

# ==========================
# Data Setup (chutes por partida)
# type/resultado, x, y, xg, video
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("A gol", 109.70, 23.88, 0.18, None),
    ],
    "Vs Slavia Praha": [
        ("Fora", 104.55, 23.88, 0.07, None),
    ],
    "Vs Sockers": [
        ("A gol", 94.91, 30.52, 0.10, None),
        ("Gol", 108.04, 43.16, 0.35, None),
    ],
}

# Create DataFrames for each match and combined
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(events, columns=["type", "x", "y", "xg", "video"])

# All games combined
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All shots": df_all}
full_data.update(dfs_by_match)

# ==========================
# Style (parecido com seu 1º shotmap)
# ==========================
def get_style(result_type: str, has_video: bool):
    """Returns marker, color, size, and linewidth based on shot result type."""
    t = (result_type or "").strip().upper()

    # Ajuste de alpha se tiver vídeo (opcional, fica sutil)
    alpha = 0.95 if has_video else 0.85

    if t == "GOL":
        return "*", (239/255, 71/255, 111/255, alpha), 160, 1.5  # #EF476F
    if t in ("A GOL", "NO ALVO"):
        return "h", (6/255, 214/255, 160/255, alpha), 140, 1.5   # #06D6A0
    if t == "FORA":
        return "o", (255/255, 209/255, 102/255, alpha), 130, 1.5 # #FFD166
    if t == "BLOQUEADO":
        return "s", (17/255, 138/255, 178/255, alpha), 130, 1.5  # #118AB2

    return "o", (0.6, 0.6, 0.6, alpha), 120, 1.2

def size_from_xg(xg: float, scale: float = 1400.0):
    # Mesma ideia do seu shotmap original: tamanho cresce com xG
    try:
        return (float(xg) * scale) + 60
    except Exception:
        return 120

# ==========================
# Sidebar Configuration (All shots ou por partida)
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

# Optional filter por resultado
df_base = full_data[selected_match].copy()
result_options = sorted(df_base["type"].unique().tolist())
selected_results = st.sidebar.multiselect("Shot Result", result_options, default=result_options)

st.sidebar.divider()
st.sidebar.caption("Match filtered by selected options above")

df = df_base[df_base["type"].isin(selected_results)].copy()

# ==========================
# Main Layout
# ==========================
col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Shot Map")

    # Campo (mesmo approach do Duel Map)
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0e0e0e", line_color="#e0e0e0")
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in df.iterrows():
        has_vid = row["video"] is not None
        marker, color, _, lw = get_style(row["type"], has_vid)

        pitch.scatter(
            row.x,
            row.y,
            marker=marker,
            s=size_from_xg(row["xg"]),
            color=color,
            edgecolors="#ffffff",   # borda branca (como seu shotmap)
            linewidths=lw,
            ax=ax,
            zorder=3,
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
    for text in legend.get_texts():
        text.set_color("#eaeaea")
    legend.get_title().set_color("#eaeaea")

    # Convert plot to image for coordinate tracking (IGUAL ao Duel Map)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)

    # Use fixed width to ensure coordinate scaling works (IGUAL)
    click = streamlit_image_coordinates(img_obj, width=700)

# ==========================
# Interaction Logic (IGUAL ao seu Duel Map)
# ==========================
selected_event = None

if click is not None:
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    # Map pixel click to actual image pixels
    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    # Invert Y for Matplotlib logic and transform to pitch data coordinates
    mpl_pixel_y = real_h - pixel_y
    coords = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))
    field_x, field_y = coords[0], coords[1]

    # Calculate distance to markers
    df["dist"] = np.sqrt((df["x"] - field_x) ** 2 + (df["y"] - field_y) ** 2)

    # Radius threshold for easier selection (IGUAL)
    RADIUS = 5
    candidates = df[df["dist"] < RADIUS]

    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Video Display (com mensagem quando não tem vídeo)
# ==========================
with col_vid:
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
    else:
        st.info("Select a marker on the pitch to view event details.")

    st.divider()
    st.subheader("Filtered data")
    st.dataframe(df.drop(columns=["dist"], errors="ignore"), use_container_width=True)
