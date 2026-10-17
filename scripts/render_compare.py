"""Side-by-side particle video of BladeMaster and TopoCut-style MLS-MPM runs."""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

run, out, title = sys.argv[1], sys.argv[2], sys.argv[3]
panels = [("blademaster", "BladeMaster"), ("topocut_shear0.1", "MLS-MPM, damaged shear x0.1"), ("topocut_shear1", "MLS-MPM, damaged shear x1")]
data = [np.load(f"{run}/{name}.npz") for name, _ in panels]
times = data[0]["times"]
colors = plt.get_cmap("tab10")
everything = np.concatenate([d["positions"].reshape(-1, 3) for d in data])
finite = everything[np.isfinite(everything).all(1)]
lo, hi = np.percentile(finite, 0.5, axis=0), np.percentile(finite, 99.5, axis=0)
centre, span = (lo + hi) / 2, (hi - lo).max() / 2 * 1.05

fig = plt.figure(figsize=(15, 5.4))
axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]
plots = []
for ax, d, (_, label) in zip(axes, data, panels):
    x = d["positions"][0]
    c = colors(d["labels"] % 10)
    plots.append(ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=1.2, c=c, depthshade=False, linewidths=0))
    ax.set_title(label, fontsize=11)
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[1] - span, centre[1] + span)
    ax.set_zlim(max(0, centre[2] - span), max(0, centre[2] - span) + 2 * span)
    ax.view_init(elev=22, azim=-60)
    ax.set_axis_off()
caption = fig.suptitle("")
writer = FFMpegWriter(fps=25, bitrate=4000)
with writer.saving(fig, out, dpi=90):
    for i, t in enumerate(times):
        for plot, d in zip(plots, data):
            x = d["positions"][min(i, len(d["positions"]) - 1)]
            plot._offsets3d = (x[:, 0], x[:, 1], x[:, 2])
        caption.set_text(f"{title}   t = {t:.2f} s   (colour = final piece)")
        writer.grab_frame()
print(out, len(times), "frames")
