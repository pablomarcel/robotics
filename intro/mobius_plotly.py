#!/usr/bin/env python3
import numpy as np, os
import plotly.graph_objects as go

def mobius_uv(R=1.0, w=0.35, nu=400, nv=50):
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(-w, w, nv)
    U, V = np.meshgrid(u, v)
    X = (R + V*np.cos(U/2.0)) * np.cos(U)
    Y = (R + V*np.cos(U/2.0)) * np.sin(U)
    Z =  V*np.sin(U/2.0)
    return X, Y, Z

if __name__ == "__main__":
    X, Y, Z = mobius_uv()
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale="Turbo")])
    fig.update_layout(title="Möbius strip (Plotly)", scene=dict(aspectmode="data"))
    os.makedirs("intro/out", exist_ok=True)
    out = "intro/out/mobius_plotly.html"
    fig.write_html(out)
    print(f"Wrote interactive HTML → {out}")
