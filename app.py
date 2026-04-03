import streamlit as st
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
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

st.title("Shot Map Analysis - Interactive")
st.caption("Click on the icons on the pitch to play the corresponding video analysis.")

# ==========================
# Data Setup (MESMO PADRÃO DO 2º CÓDIGO: type, x, y, video + xg)
# ==========================
matches_data = {
    "All shots": [
        ("Gol",       105.00, 40.00, 0.30, None),
        ("Fora",      102.00, 30.00, 0.12, None),
        ("Bloqueado",  98.00, 50.00, 0.05, None),
        ("Gol",       110.00, 45.00, 0.45, None),
        ("Fora",       95.00, 35.00, 0.08, None),
        ("No Alvo",   108.00, 42.00, 0.20, None),
        ("Bloqueado", 100.00, 38.00, 0.15, None),
    ],
}

# DataFrames por "match" (igual ao 2º)
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(
        events,
        columns=["type", "x", "y", "xg", "video"]
    )

df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All games": df_all}
full_data.update(dfs_by_match)

# ==========================
# Sidebar Configuration (igual ao 2º)
# ==========================
st.sidebar.header("📋 Filter Configuration")
selected_match = st.sidebar.radio("Select a match", list(full_data.keys()), index=0)

st.sidebar.divider()

filter_types = st.sidebar.multiselect(
    "Shot Result (type)",
    options=sorted(full_data[selected_match]["type"].unique().tolist()),
    default=sorted(full_data[selected_match]["type"].unique().tolist()),
)

scale = st.sidebar.slider("xG size scale", min_value=200, max_value=4000, value=1400, step=100)
RADIUS = st.sidebar.slider("Click radius threshold", min_value=1, max_value=15, value=5, step=1)

df = full_data[selected_match].copy()
df = df[df["type"].isin(filter_types)].copy()

# ==========================
# Style (mapa de tiros)
# ==========================
def get_style(event_type: str, has_video: bool):
    event_type = event_type.strip().upper()

    if event_type == "GOL":
        return "*", "#EF476F", 1.5
    if event_type in ("NO ALVO", "NO ALVO " , "NO ALVO".upper()):
        return "h", "#06D6A0", 1.5
    if event_type == "FORA":
        return "o", "#FFD166", 1.5
    if event_type == "BLOQUEADO":
        return "s", "#118AB2", 1.5

    return "o", "#999999", 1.0

# ==========================
# Main Layout
# ==========================
col_map, col_vid = st.columns([1, 1])

with col_map:
    st.subheader("Interactive Pitch Map")

    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pitch_color="#0e0e0e",
        line_color="#e0e0e0"
    )

    fig, ax = pitch.draw(figsize=(10, 7))

    # Plot (igual ao 2º: iterrows + style por linha)
    for _, row in df.iterrows():
        has_vid = row["video"] is not None
        marker, color, lw = get_style(row["type"], has_vid)

        # Mantém a estética do seu 1º: borda branca
        pitch.scatter(
            row["x"], row["y"],
            s=(row["xg"] * scale) + 60,
            marker=marker,
            c=color,
            edgecolors="#ffffff",
            linewidth=lw,
            ax=ax,
            zorder=3
        )

    # Legenda (mesma ideia do seu 1º)
    legend_elements = [
        Line2D([0], [0], marker='*', color='none', label='Gol',
               markerfacecolor='#EF476F', markeredgecolor='#ffffff', markersize=11),
        Line2D([0], [0], marker='h', color='none', label='No alvo',
               markerfacecolor='#06D6A0', markeredgecolor='#ffffff', markersize=9),
        Line2D([0], [0], marker='o', color='none', label='Fora',
               markerfacecolor='#FFD166', markeredgecolor='#ffffff', markersize=9),
        Line2D([0], [0], marker='s', color='none', label='Bloqueado',
               markerfacecolor='#118AB2', markeredgecolor='#ffffff', markersize=9),
    ]

    legend = ax.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=True,
        fontsize=10,
        labelspacing=0.6,
        handletextpad=0.4,
        columnspacing=1.2,
        borderpad=0.6
    )
    legend.get_frame().set_facecolor("#111111")
    legend.get_frame().set_edgecolor("#444444")
    legend.get_frame().set_linewidth(1.0)
    for text in legend.get_texts():
        text.set_color("#eaeaea")

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Converter plot em imagem para capturar clique (EXATAMENTE como no 2º)
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_obj = Image.open(buf)
    plt.close(fig)

    click = streamlit_image_coordinates(img_obj, width=700)

# ==========================
# Interaction Logic (COPIADO DO 2º: pixel->data coords->dist)
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

    df_sel = df.copy()
    df_sel["dist"] = np.sqrt((df_sel["x"] - field_x)**2 + (df_sel["y"] - field_y)**2)

    candidates = df_sel[df_sel["dist"] < RADIUS]
    if not candidates.empty:
        selected_event = candidates.loc[candidates["dist"].idxmin()]

# ==========================
# Video Display (MESMO PADRÃO DO 2º)
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
            st.warning("No video footage available for this specific event.")
    else:
        st.info("Select a marker on the pitch to view event details.")

    st.divider()
    st.subheader("Data (debug)")
    st.dataframe(df, use_container_width=True)
