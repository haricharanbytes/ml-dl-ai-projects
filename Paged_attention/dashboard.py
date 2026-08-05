"""
Run with:  streamlit run dashboard.py
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from simulator import run_batch, verify_correctness, cow_demo, utilisation_benchmark

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(page_title="PagedAttention, Visualized", layout="wide", page_icon="▦")

BG = "#0B0E14"
PANEL = "#131826"
GRID_LINE = "#1F2733"
TEXT = "#E6E9EF"
MUTED = "#7A8699"
FREE = "#1F2733"
ACCENT_TEAL = "#4FD1C5"
ACCENT_AMBER = "#F5A623"
ACCENT_CORAL = "#E85D5D"
REQ_PALETTE = ["#4FD1C5", "#F5A623", "#8B7CF6", "#5DA3E8", "#E85D5D", "#6FCF6F", "#F2C94C", "#C77DFF"]

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    html, body, [class*="css"] {{
        font-family: 'JetBrains Mono', 'SF Mono', ui-monospace, monospace;
    }}
    h1, h2, h3 {{
        font-family: 'Inter', -apple-system, sans-serif !important;
        letter-spacing: -0.01em;
    }}
    .metric-card {{
        background: {PANEL}; border: 1px solid {GRID_LINE}; border-radius: 4px;
        padding: 14px 18px; margin-bottom: 8px;
    }}
    .metric-label {{ color: {MUTED}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ font-size: 1.6rem; font-weight: 600; color: {TEXT}; }}
    .eyebrow {{ color: {ACCENT_TEAL}; font-size: 0.75rem; letter-spacing: 0.12em; text-transform: uppercase; }}
    section[data-testid="stSidebar"] {{ background-color: {PANEL}; border-right: 1px solid {GRID_LINE}; }}
    .stTabs [data-baseweb="tab"] {{ font-family: 'Inter', sans-serif; }}
    div[data-testid="stMetricValue"] {{ color: {ACCENT_TEAL}; }}
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, sub=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div style="color:{MUTED}; font-size:0.75rem;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.markdown('<div class="eyebrow">Model shape</div>', unsafe_allow_html=True)
num_layers = st.sidebar.slider("Layers", 1, 32, 8)
num_heads = st.sidebar.slider("Heads", 1, 32, 8)
head_dim = st.sidebar.select_slider("Head dim", [16, 32, 64, 128], value=64)
embed_dim = num_heads * head_dim

st.sidebar.markdown('<div class="eyebrow">Paging</div>', unsafe_allow_html=True)
page_size = st.sidebar.select_slider("Page size (tokens)", [8, 16, 32, 64], value=16)
total_pages = st.sidebar.slider("Physical pages in pool", 32, 1024, 256, step=32)

st.sidebar.markdown('<div class="eyebrow">Workload</div>', unsafe_allow_html=True)
num_requests = st.sidebar.slider("Concurrent requests", 2, 8, 5)
max_seq = st.sidebar.slider("Max sequence length (naive alloc)", 512, 4096, 2048, step=256)
seed = st.sidebar.number_input("Random seed", value=42)

rng = np.random.default_rng(int(seed))
token_counts = list(np.clip(rng.normal(max_seq * 0.22, max_seq * 0.1, num_requests).astype(int), 16, max_seq))

kv_bytes_per_token = 2 * num_layers * num_heads * head_dim * 2  # fp16 reference, matches original script
kv_mb_per_token = kv_bytes_per_token / 1024 / 1024

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="eyebrow">Systems demo · real attention math, not just token counters</div>', unsafe_allow_html=True)
st.title("PagedAttention, Visualized")
st.caption(
    "Every number below comes from actual numpy attention computed over a real paged KV cache -- "
    "not a simulated abstraction. The correctness tab proves paged output is bit-identical to a plain cache."
)

layers, cache, requests, events = run_batch(
    num_layers, num_heads, head_dim, embed_dim, page_size, total_pages, token_counts, seed=int(seed)
)

# ----------------------------------------------------------------------
# Top-line metrics
# ----------------------------------------------------------------------
naive_mb_total = sum(max_seq * kv_mb_per_token for _ in token_counts)
paged_mb_total = cache.bytes_used / 1024 / 1024
naive_used_mb = sum(t * kv_mb_per_token for t in token_counts)

c1, c2, c3, c4 = st.columns(4)
with c1: metric_card("Naive pre-allocation", f"{naive_mb_total:,.0f} MB", f"{num_requests} req × {max_seq} tok max")
with c2: metric_card("Paged allocation", f"{paged_mb_total:,.1f} MB", f"{total_pages - len(cache.free_pages)} / {total_pages} pages used")
with c3:
    waste_pct = (1 - naive_used_mb / naive_mb_total) * 100
    metric_card("Naive waste", f"{waste_pct:.1f}%", "memory reserved but never written")
with c4:
    saved = naive_mb_total - paged_mb_total
    metric_card("Memory saved by paging", f"{saved:,.0f} MB", f"{saved / naive_mb_total * 100:.1f}% reduction")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["▦ Page grid", "✓ Correctness proof", "⧉ Copy-on-Write", "▤ Utilisation benchmark"])

# ----------------------------------------------------------------------
# TAB 1 -- Page grid (the signature visual)
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Physical page ownership")
    st.caption("Each square is one physical page in the pool. Color = owning request. Grey = free, reusable immediately.")

    cols = min(32, total_pages)
    rows = math.ceil(total_pages / cols)
    owner = ["free"] * total_pages
    for i, r in enumerate(requests):
        color = REQ_PALETTE[i % len(REQ_PALETTE)]
        for pid in r.block_table:
            owner[pid] = r.req_id

    z = []
    text = []
    for row in range(rows):
        zrow, trow = [], []
        for col in range(cols):
            pid = row * cols + col
            if pid >= total_pages:
                zrow.append(-1); trow.append("")
                continue
            o = owner[pid]
            if o == "free":
                zrow.append(0)
            else:
                idx = int(o.split("-")[1])
                zrow.append(idx + 1)
            trow.append(f"page {pid}<br>{o}")
        z.append(zrow); text.append(trow)

    colorscale = [[0.0, FREE]]
    n_req = len(requests)
    for i in range(n_req):
        frac_lo = (i + 1) / (n_req + 1)
        frac_hi = (i + 2) / (n_req + 1)
        c = REQ_PALETTE[i % len(REQ_PALETTE)]
        colorscale.append([frac_lo, c])
        colorscale.append([frac_hi, c])
    colorscale[-1][0] = 1.0

    fig = go.Figure(data=go.Heatmap(
        z=z, text=text, hovertemplate="%{text}<extra></extra>",
        colorscale=colorscale, showscale=False, xgap=2, ygap=2, zmin=-1, zmax=n_req + 1,
    ))
    fig.update_layout(
        plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
        yaxis=dict(autorange="reversed", showgrid=False, showticklabels=False, zeroline=False),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        height=max(220, rows * 26), margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    legend_cols = st.columns(len(requests) + 1)
    for i, r in enumerate(requests):
        with legend_cols[i]:
            st.markdown(
                f'<span style="color:{REQ_PALETTE[i % len(REQ_PALETTE)]}">■</span> '
                f'**{r.req_id}** · {r.num_tokens} tok · {len(r.block_table)} pages',
                unsafe_allow_html=True,
            )
    with legend_cols[-1]:
        st.markdown(f'<span style="color:{FREE}">■</span> free', unsafe_allow_html=True)

    st.markdown("##### Try it: free a request")
    free_choice = st.selectbox("Release a request's pages back to the pool", [r.req_id for r in requests])
    if st.button("Free pages"):
        target = next(r for r in requests if r.req_id == free_choice)
        for pid in target.block_table:
            cache.release_page(pid)
        st.success(f"{free_choice} freed {len(target.block_table)} pages -- they're immediately reusable by any request.")
        target.block_table = []

# ----------------------------------------------------------------------
# TAB 2 -- Correctness proof
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Does paging change the math? No.")
    st.caption(
        "For each request, attention is computed twice with the SAME random query: once by gathering "
        "K/V from scattered physical pages via the block table, once from a plain contiguous array. "
        "The max absolute difference below should be 0.0 (up to float rounding)."
    )
    results = verify_correctness(layers, cache, [r for r in requests if r.block_table])
    df = pd.DataFrame([{"request": r["req_id"], "tokens": next(x.num_tokens for x in requests if x.req_id == r["req_id"]),
                         "max |diff| paged vs contiguous": r["max_diff"]} for r in results])
    if len(df):
        st.dataframe(df, use_container_width=True, hide_index=True)
        all_zero = (df["max |diff| paged vs contiguous"] < 1e-5).all()
        if all_zero:
            st.success("✓ Every request: paged attention output is numerically identical to the naive contiguous cache.")
        else:
            st.warning("Some diffs are non-zero -- check page_size / total_pages configuration.")
    else:
        st.info("Free some requests above, or increase concurrent requests, to populate this table.")

# ----------------------------------------------------------------------
# TAB 3 -- Copy-on-Write
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Sharing a system prompt across requests")
    n_users = st.slider("Number of requests sharing this system prompt", 2, 30, 10)
    sys_tokens = st.slider("System prompt length (tokens)", 16, 512, 200, step=16)
    cow = cow_demo(num_layers, num_heads, head_dim, embed_dim, page_size, total_pages, sys_tokens, n_users, seed=int(seed))

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Naive (N copies)", f"{cow['naive_bytes']/1024/1024:.1f} MB")
    with c2: metric_card("CoW (1 shared copy)", f"{cow['cow_bytes']/1024/1024:.2f} MB")
    with c3: metric_card("Saved", f"{cow['saved_bytes']/1024/1024:.1f} MB", f"{cow['saved_bytes']/cow['naive_bytes']*100:.1f}% reduction")

    st.markdown("##### Reference counts on shared pages")
    ref_df = pd.DataFrame({
        "page_id": cow["shared_pids"],
        "refs before divergence": [cow["ref_counts_before"][p] for p in cow["shared_pids"]],
        "refs after divergence":  [cow["ref_counts_after"][p] for p in cow["shared_pids"]],
    })
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
    st.caption(
        f"One request diverged mid-generation: page **{cow['old_pid']}** was copied to new physical page "
        f"**{cow['new_pid']}** (ref count dropped by 1), while the other {n_users - 1} requests keep sharing "
        f"the original page untouched."
    )

# ----------------------------------------------------------------------
# TAB 4 -- Utilisation benchmark
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Utilisation vs. batch size")
    st.caption("Naive utilisation stays low regardless of batch size (always dividing by max_seq_len). Paged stays high because waste is capped at under one page per request.")
    batch_options = st.multiselect("Batch sizes to test", [5, 10, 25, 50, 100, 200, 400], default=[10, 25, 50, 100, 200])
    if batch_options:
        bench = utilisation_benchmark(sorted(batch_options), max_seq=max_seq, page_size=page_size, seed=int(seed))
        bdf = pd.DataFrame(bench)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bdf.batch_size, y=bdf.naive_pct, mode="lines+markers", name="Naive",
                                   line=dict(color=ACCENT_CORAL, width=3)))
        fig2.add_trace(go.Scatter(x=bdf.batch_size, y=bdf.paged_pct, mode="lines+markers", name="Paged",
                                   line=dict(color=ACCENT_TEAL, width=3)))
        fig2.update_layout(
            plot_bgcolor=BG, paper_bgcolor=BG, font_color=TEXT,
            xaxis=dict(title="Batch size", gridcolor=GRID_LINE), yaxis=dict(title="Utilisation %", gridcolor=GRID_LINE, range=[0, 105]),
            legend=dict(bgcolor=BG), height=420, margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(bdf.rename(columns={"batch_size": "Batch", "naive_pct": "Naive %", "paged_pct": "Paged %"}),
                     use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Built as a from-scratch teaching simulation of vLLM's PagedAttention (Kwon et al., 2023). "
    "Attention math is real numpy; GPU execution, kernels, and scheduling are simplified for clarity."
)