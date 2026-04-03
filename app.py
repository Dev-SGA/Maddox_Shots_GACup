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
st.set_page_config(layout="wide", page_title="Shot Map + Goal Map (Interactive)")

st.title("Shot Map + Goal Placement (Interactive)")
st.caption("Clique em um ícone no campo (ou no gol) para selecionar o evento e, se houver, ver o vídeo.")

# ==========================
# GOAL DIMENSIONS (em metros)
# ==========================
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44

# ==========================
# Data Setup (match -> lista de eventos)
# Campos:
# type: resultado (Gol, A gol, Fora, Bloqueado)
# x,y: coordenadas do campo (StatsBomb)
# goal_x,goal_y: coordenadas do gol (largura 0..7.32, altura 0..2.44)
# video: link/path ou None
# ==========================
matches_data = {
    "Vs Los Angeles": [
        ("A gol", 109.70, 23.88, 5.32, 0.19, None),
    ],
    "Vs Slavia Praha": [
        ("Fora", 104.55, 23.88, None, None, None),
    ],
    "Vs Sockers": [
        ("A gol", 94.91, 30.52, 0.86, 0.51, None),
        ("Gol", 108.04, 43.16, 2.22, 0.11, None),
    ],
}

# DataFrames por partida
dfs_by_match = {}
for match_name, events in matches_data.items():
    dfs_by_match[match_name] = pd.DataFrame(
        events,
        columns=["type", "x", "y", "goal_x", "goal_y", "video"]
    )

# Tudo combinado (sidebar: "All shots")
df_all = pd.concat(dfs_by_match.values(), ignore_index=True)
full_data = {"All shots": df_all}
full_data.update(dfs_by_match)

# ==========================
# Style helpers (campo)
# ==========================
FIELD_STYLE = {
    "GOL":       dict(marker="*", color="#EF476F"),
    "A GOL":     dict(marker="h", color="#06D6A0"),   # "No alvo" / "A gol"
    "NO ALVO":   dict(marker="h", color="#06D6A0"),
    "FORA":      dict(marker="o", color="#FFD166"),
    "BLOQUEADO": dict(marker="s", color="#118AB2"),
}

def normalize_type(t: str) -> str:
    t = (t or "").strip().upper()
    # normalizar variações
    if t in ("A GOL", "A GOL)", "A GOL."):
        return "A GOL"
    return t

def get_field_style(t: str):
    t = normalize_type(t)
    return FIELD_STYLE.get(t, dict(marker="o", color="#999999"))

def fig_to_image(fig, dpi=120):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)

# ==========================
# Draw functions
# ==========================
def draw_field(df_plot: pd.DataFrame, size_scale: int):
    pitch = VerticalPitch(
        half=True,
        pitch_type="statsbomb",
        pitch_color="#0e0e0e",
        line_color="#e0e0e0"
    )
    fig, ax = pitch.draw(figsize=(10, 7))

    # tamanho fixo (como não temos xG nos dados novos)
    base_size = 140

    for _, row in df_plot.iterrows():
        stl = get_field_style(row["type"])
        pitch.scatter(
            row["x"], row["y"],
            s=base_size,
            marker=stl["marker"],
            c=stl["color"],
            edgecolors="#ffffff",
            linewidth=1.5,
            ax=ax,
            zorder=3
        )

    # legenda (compacta)
    legend_elements = [
        Line2D([0], [0], marker='*', color='none', label='Gol',
               markerfacecolor=FIELD_STYLE["GOL"]["color"], markeredgecolor="#ffffff", markersize=11),
        Line2D([0], [0], marker='h', color='none', label='A gol',
               markerfacecolor=FIELD_STYLE["A GOL"]["color"], markeredgecolor="#ffffff", markersize=9),
        Line2D([0], [0], marker='o', color='none', label='Fora',
               markerfacecolor=FIELD_STYLE["FORA"]["color"], markeredgecolor="#ffffff", markersize=9),
        Line2D([0], [0], marker='s', color='none', label='Bloqueado',
               markerfacecolor=FIELD_STYLE["BLOQUEADO"]["color"], markeredgecolor="#ffffff", markersize=9),
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="lower center",
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
    return fig, ax

def draw_goal(df_plot: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    # postes e travessão
    ax.plot([0, GOAL_WIDTH], [GOAL_HEIGHT, GOAL_HEIGHT], color="white", lw=3)
    ax.plot([0, 0], [0, GOAL_HEIGHT], color="white", lw=3)
    ax.plot([GOAL_WIDTH, GOAL_WIDTH], [0, GOAL_HEIGHT], color="white", lw=3)

    # zonas (grade 3x3)
    x1 = GOAL_WIDTH / 3
    x2 = 2 * GOAL_WIDTH / 3
    y1 = GOAL_HEIGHT / 3
    y2 = 2 * GOAL_HEIGHT / 3
    ax.plot([x1, x1], [0, GOAL_HEIGHT], color="white", alpha=0.2)
    ax.plot([x2, x2], [0, GOAL_HEIGHT], color="white", alpha=0.2)
    ax.plot([0, GOAL_WIDTH], [y1, y1], color="white", alpha=0.2)
    ax.plot([0, GOAL_WIDTH], [y2, y2], color="white", alpha=0.2)

    # plotar apenas eventos com goal_x/goal_y
    df_goal = df_plot.dropna(subset=["goal_x", "goal_y"]).copy()
    for _, row in df_goal.iterrows():
        t = normalize_type(row["type"])
        # mesma paleta por resultado
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
            row["goal_x"], row["goal_y"],
            color=c,
            marker=m,
            s=90,
            edgecolors="white",
            linewidth=1.5,
            zorder=3
        )

    ax.set_xlim(-0.5, GOAL_WIDTH + 0.5)
    ax.set_ylim(0, GOAL_HEIGHT + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Goal View (Shot Placement)", color="white")
    plt.tight_layout()
    return fig, ax

# ==========================
# Sidebar filters
# ==========================
st.sidebar.header("📋 Filtros")
selected_match = st.sidebar.radio("Selecionar partida", list(full_data.keys()), index=0)

df = full_data[selected_match].copy()

all_types = sorted(df["type"].unique().tolist())
selected_types = st.sidebar.multiselect("Resultado", options=all_types, default=all_types)
df = df[df["type"].isin(selected_types)].copy()

st.sidebar.divider()
FIELD_RADIUS = st.sidebar.slider("Raio de clique (campo)", 1, 15, 5, 1)
GOAL_RADIUS = st.sidebar.slider("Raio de clique (gol)", 0.05, 1.00, 0.25, 0.05)

# ==========================
# Main layout: Campo | Gol | Detalhes/Vídeo
# ==========================
col_field, col_goal, col_details = st.columns([1, 1, 1])

# --- Render campo (como no seu 2º código: figura->png->click coords)
with col_field:
    st.subheader("Mapa de chutes (campo)")

    fig_f, ax_f = draw_field(df, size_scale=1400)
    img_field = fig_to_image(fig_f, dpi=110)
    plt.close(fig_f)

    click_field = streamlit_image_coordinates(img_field, width=520)

# --- Render gol (figura->png->click coords)
with col_goal:
    st.subheader("Mapa do gol (colocação)")

    fig_g, ax_g = draw_goal(df)
    img_goal = fig_to_image(fig_g, dpi=140)
    plt.close(fig_g)

    click_goal = streamlit_image_coordinates(img_goal, width=520)

# ==========================
# Interaction logic (seleciona evento pelo clique no CAMPO ou no GOL)
# ==========================
selected_event = None
selected_from = None

def pick_event_from_field(click, img_obj, ax, df_in: pd.DataFrame):
    if click is None:
        return None

    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    mpl_pixel_y = real_h - pixel_y
    field_x, field_y = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))

    df_sel = df_in.copy()
    df_sel["dist"] = np.sqrt((df_sel["x"] - field_x) ** 2 + (df_sel["y"] - field_y) ** 2)
    candidates = df_sel[df_sel["dist"] < FIELD_RADIUS]
    if candidates.empty:
        return None
    return candidates.loc[candidates["dist"].idxmin()]

def pick_event_from_goal(click, img_obj, ax, df_in: pd.DataFrame):
    if click is None:
        return None

    # Converter clique para coords do plot
    real_w, real_h = img_obj.size
    disp_w, disp_h = click["width"], click["height"]

    pixel_x = click["x"] * (real_w / disp_w)
    pixel_y = click["y"] * (real_h / disp_h)

    mpl_pixel_y = real_h - pixel_y
    gx, gy = ax.transData.inverted().transform((pixel_x, mpl_pixel_y))

    # Só eventos que têm coords de gol
    df_goal = df_in.dropna(subset=["goal_x", "goal_y"]).copy()
    if df_goal.empty:
        return None

    df_goal["dist_goal"] = np.sqrt((df_goal["goal_x"] - gx) ** 2 + (df_goal["goal_y"] - gy) ** 2)
    candidates = df_goal[df_goal["dist_goal"] < GOAL_RADIUS]
    if candidates.empty:
        return None
    return candidates.loc[candidates["dist_goal"].idxmin()]

picked_field = pick_event_from_field(click_field, img_field, ax_f, df)
picked_goal = pick_event_from_goal(click_goal, img_goal, ax_g, df)

# Prioridade: se clicou no gol por último, o streamlit_image_coordinates não dá "timestamp",
# então a gente prioriza "gol" se houver clique nele; senão usa "campo".
if picked_goal is not None:
    selected_event = picked_goal
    selected_from = "goal"
elif picked_field is not None:
    selected_event = picked_field
    selected_from = "field"

# ==========================
# Details / Video panel
# ==========================
with col_details:
    st.subheader("Detalhes do evento")

    if selected_event is None:
        st.info("Clique em um ponto no campo ou no gol para selecionar um evento.")
    else:
        st.success(f"**Selected Event:** {selected_event['type']}  ({'gol' if selected_from=='goal' else 'campo'})")
        st.info(f"**Field Position:** X: {selected_event['x']:.2f}, Y: {selected_event['y']:.2f}")

        if pd.notna(selected_event.get("goal_x")) and pd.notna(selected_event.get("goal_y")):
            st.info(f"**Goal Position:** X: {selected_event['goal_x']:.2f}, Y: {selected_event['goal_y']:.2f}")
        else:
            st.warning("Este evento não tem coordenadas de gol (goal_x/goal_y).")

        if selected_event.get("video"):
            try:
                st.video(selected_event["video"])
            except Exception:
                st.error(f"Video file not found: {selected_event['video']}")
        else:
            st.warning("No video footage available for this specific event.")

    st.divider()
    st.subheader("Tabela (filtrada)")
    st.dataframe(df, use_container_width=True)
