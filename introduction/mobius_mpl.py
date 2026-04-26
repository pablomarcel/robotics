#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

def mobius_uv(R=1.0, w=0.3, nu=300, nv=40):
    """
    Standard 2-parameter Möbius strip:
      u ∈ [0, 2π) (angle around the circle)
      v ∈ [-w, w] (half-width across the strip)

    x = (R + v*cos(u/2)) * cos(u)
    y = (R + v*cos(u/2)) * sin(u)
    z =  v*sin(u/2)
    """
    u = np.linspace(0, 2*np.pi, nu)
    v = np.linspace(-w, w, nv)
    U, V = np.meshgrid(u, v)
    X = (R + V*np.cos(U/2.0)) * np.cos(U)
    Y = (R + V*np.cos(U/2.0)) * np.sin(U)
    Z =  V*np.sin(U/2.0)
    return X, Y, Z

if __name__ == "__main__":
    X, Y, Z = mobius_uv(R=1.0, w=0.35)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", linewidth=0.15, alpha=1.0)

    # make aspect roughly equal
    max_range = (np.ptp(X) + np.ptp(Y) + np.ptp(Z)) / 3
    mid = np.array([X.mean(), Y.mean(), Z.mean()])
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range/2, m + max_range/2)

    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Möbius strip (Matplotlib)")
    out = "introduction/out/mobius_mpl.png"
    import os; os.makedirs("introduction/out", exist_ok=True)
    plt.tight_layout(); plt.savefig(out, dpi=200)
    print(f"Saved {out}")
