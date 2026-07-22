#!/usr/bin/env python3
"""Interactive 3-D 'structure from nothing' explorer.

Thirty-one MEANING-FREE tokens (Apple, River, Chair, Willow, ...) given only a
declared parent/child hierarchy -- and asked NOTHING structural (neutral queries)
-- arrange themselves into depth bands: a tree, built from tokens with no prior
structure to inherit. This is the paper's arbitrary-token result, made rotatable.

Depth buttons grow the tree (depth 2 -> 3 -> 4) so you can watch the structure
appear. Nodes are the actual arbitrary tokens (labeled), colored by tree depth;
edges are the declared parent/child links.

Data (committed, no GPU): data/geometry/demo_tree_gemma-4-31B-it_d4.npz
(24 neutral queries x 31 nodes; R = per-(query,node) residual, NODE = node id,
perm = the 31 token names). Consensus position per node = mean over the queries.

    python figures/build_interactive_arbtree.py
Out: assets/arbtree_3d.html
"""
import os
import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CIK_DATA", os.path.normpath(os.path.join(HERE, "..", "data")))
ASSETS = os.environ.get("CIK_ASSETS", os.path.normpath(os.path.join(HERE, "..", "assets")))
os.makedirs(ASSETS, exist_ok=True)
TAG = "gemma-4-31B-it"

N = 31
PARENT = {i: (i - 1) // 2 for i in range(1, N)}
depth = np.floor(np.log2(np.arange(N) + 1)).astype(int)
EDGES = [(i, PARENT[i]) for i in range(1, N)]

z = np.load(os.path.join(DATA, "geometry", f"demo_tree_{TAG}_d4.npz"), allow_pickle=True)
R = z["R"].astype(np.float64)
NODE = z["NODE"]
tokens = [str(t) for t in z["perm"]][:N]          # the 31 meaning-free tokens, in node order

# joint 3-D PCA over all (query, node) residuals; consensus position per node
mu = R.mean(0)
_, _, Vt = np.linalg.svd(R - mu, full_matrices=False)
P = (R - mu) @ Vt[:3].T
M = np.stack([P[NODE == k].mean(0) for k in range(N)])   # 31 x 3 consensus tree


def edge_xyz(cap):
    xs, ys, zs = [], [], []
    for a, b in EDGES:
        if depth[a] <= cap:
            xs += [M[a, 0], M[b, 0], None]; ys += [M[a, 1], M[b, 1], None]; zs += [M[a, 2], M[b, 2], None]
    return xs, ys, zs


CAPS = [(2, "Depth 2  (3 tokens)"), (3, "Depth 3  (7 tokens)"), (4, "Depth 4  (all 31)")]
fig = go.Figure()
cap_traces = {}   # cap -> [trace indices]
ti = 0
for cap, _lab in CAPS:
    keep = np.where(depth <= cap)[0]
    ex, ey, ez = edge_xyz(cap)
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="rgba(110,110,110,0.7)", width=4),
                               hoverinfo="skip", showlegend=False, visible=(cap == 4)))
    e_i = ti; ti += 1
    fig.add_trace(go.Scatter3d(
        x=M[keep, 0], y=M[keep, 1], z=M[keep, 2], mode="markers+text",
        text=[tokens[k] for k in keep], textposition="top center", textfont=dict(size=9),
        marker=dict(size=6, color=depth[keep], colorscale="Viridis", cmin=0, cmax=4,
                    line=dict(color="white", width=1),
                    colorbar=dict(orientation="h", x=0.5, xanchor="center", y=-0.02, len=0.35,
                                  thickness=12, title=dict(text="tree depth", side="top"),
                                  tickvals=[0, 1, 2, 3, 4])),
        hovertext=[f"{tokens[k]} - depth {depth[k]}" for k in keep], hoverinfo="text",
        showlegend=False, visible=(cap == 4)))
    n_i = ti; ti += 1
    cap_traces[cap] = [e_i, n_i]

n = ti
def _vis(cap):
    keep = set(cap_traces[cap])
    return [i in keep for i in range(n)]

scene = dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
             aspectmode="cube", camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)))
fig.update_layout(
    title=dict(text="Structure from nothing: 31 meaning-free tokens under a declared hierarchy "
                    "arrange into depth bands<br><sub>neutral queries (nothing structural is asked) "
                    "&nbsp;|&nbsp; nodes are arbitrary tokens, colored by tree depth &nbsp;|&nbsp; "
                    "drag to rotate, use the buttons to grow the tree</sub>", x=0.5, font=dict(size=15)),
    scene=scene, margin=dict(l=0, r=0, t=70, b=40), width=1050, height=720, paper_bgcolor="white",
    updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center", y=1.12, showactive=True,
                      buttons=[dict(label=lab, method="update", args=[{"visible": _vis(cap)}]) for cap, lab in CAPS])],
)
out = os.path.join(ASSETS, "arbtree_3d.html")
fig.write_html(out, include_plotlyjs=True, full_html=True)
print("WROTE", out, f"(31 arbitrary tokens, depth-4 tree)")
