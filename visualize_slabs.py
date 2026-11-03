"""Inspect samples saved by construct_toy_dataset.py, without importing FiPy.

Examples:
    python visualize_slabs.py sample_0000.npz
    python visualize_slabs.py sample_0000.npz --gif preview.gif --no-show
    python visualize_slabs.py sample_0000.npz --frame 5 --snapshot frame.png --no-show

Dependencies: numpy, matplotlib; pillow for GIF export.
"""
import argparse
from pathlib import Path
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider


def visualize(video, conductivity, frame_times, cell_size=0.0005,
              background_k=0.20, frame=0):
    """Return (figure, update_frame, slider).

    video: temperature rise (K), shape (frames, ny, nx).
    conductivity: W/(m K), shape (nz, ny, nx).
    frame_times: seconds since heating ends.
    Geometry is ground truth, not information available to an infrared camera.
    """
    video = np.asarray(video, dtype=float)
    conductivity = np.asarray(conductivity, dtype=float)
    frame_times = np.asarray(frame_times, dtype=float)
    if video.ndim != 3 or conductivity.ndim != 3:
        raise ValueError("Video and conductivity must both be 3D arrays.")
    nf, ny, nx = video.shape
    nz, ky, kx = conductivity.shape
    if min(nf, ny, nx, nz) < 1 or (ny, nx) != (ky, kx):
        raise ValueError("Video and conductivity spatial shapes do not match.")
    if frame_times.shape != (nf,) or np.any(np.diff(frame_times) <= 0):
        raise ValueError("Provide one increasing time per frame.")
    if not all(np.isfinite(a).all() for a in (video, conductivity, frame_times)):
        raise ValueError("Inputs contain non-finite values.")
    if not np.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be positive and finite.")
    if not np.isfinite(background_k) or not 0 <= frame < nf:
        raise ValueError("Invalid background conductivity or frame index.")

    spacing = cell_size * 1000  # display millimetres
    lx, ly, lz = nx * spacing, ny * spacing, nz * spacing
    mask = ~np.isclose(conductivity, background_k, rtol=1e-5, atol=1e-8)
    contrast = video - video.mean(axis=(1, 2), keepdims=True)
    contrast_limit = max(float(np.abs(contrast).max()) * 1000, 1e-3)
    tmin, tmax = float(video.min()), float(video.max())
    if tmax - tmin < 1e-8:
        tmin -= 0.01
        tmax += 0.01

    fig = plt.figure(figsize=(14, 7.3))
    gs = fig.add_gridspec(2, 3, width_ratios=(1.15, 1, 1),
                          height_ratios=(1, .55))
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_temp = fig.add_subplot(gs[0, 1])
    ax_contrast = fig.add_subplot(gs[0, 2])
    ax_trace = fig.add_subplot(gs[1, 1:])
    fig.subplots_adjust(left=.04, right=.97, top=.87, bottom=.22,
                        wspace=.38, hspace=.5)
    fig.suptitle("ThermoFlow | ground-truth interior and post-heating surface", y=.97)
    fig.text(.5, .925, "Insulated model: surface cooling is heat redistribution into the slab",
             ha="center", fontsize=10)

    # Convert (z,y,x) storage to the (x,y,z) ordering expected by voxels().
    if mask.any():
        ax3d.voxels(np.arange(nx+1)[:, None, None] * spacing,
                    np.arange(ny+1)[None, :, None] * spacing,
                    np.arange(nz+1)[None, None, :] * spacing,
                    mask.transpose(2, 1, 0), facecolors="tab:blue",
                    edgecolors="none", shade=False, alpha=.8)
    else:
        ax3d.text(lx/2, ly/2, lz/2, "No inclusions", ha="center")
    # Wireframe slab; z=0 is the heated and observed face.
    for z in (0, lz):
        ax3d.plot([0, lx, lx, 0, 0], [0, 0, ly, ly, 0], [z]*5,
                  color="gray", linewidth=.8)
    for x, y in ((0, 0), (lx, 0), (lx, ly), (0, ly)):
        ax3d.plot([x, x], [y, y], [0, lz], color="gray", linewidth=.8)
    ax3d.set(xlim=(0, lx), ylim=(0, ly), zlim=(lz, 0),
             xlabel="x (mm)", ylabel="y (mm)", zlabel="Depth z (mm)")
    ax3d.set_box_aspect((lx, ly, lz))
    ax3d.view_init(elev=24, azim=-60)
    ax3d.set_title("Ground truth: inclusions\nObserved face at z = 0", pad=18)

    extent = (0, lx, 0, ly)
    image = ax_temp.imshow(video[frame], origin="lower", extent=extent,
                           cmap="inferno", vmin=tmin, vmax=tmax,
                           interpolation="nearest", aspect="equal")
    difference = ax_contrast.imshow(contrast[frame]*1000, origin="lower",
                                    extent=extent, cmap="RdBu_r",
                                    vmin=-contrast_limit, vmax=contrast_limit,
                                    interpolation="nearest", aspect="equal")
    for ax in (ax_temp, ax_contrast):
        ax.set(xlabel="x (mm)", ylabel="y (mm)")
    fig.colorbar(image, ax=ax_temp, shrink=.8, label="Temperature rise (K)")
    fig.colorbar(difference, ax=ax_contrast, shrink=.8,
                 label="Deviation from frame mean (mK)")

    ax_trace.plot(frame_times, video.mean(axis=(1, 2)), label="Surface mean",
                   color="tab:orange")
    ax_trace.fill_between(frame_times, video.min(axis=(1, 2)),
                           video.max(axis=(1, 2)), color="tab:orange",
                           alpha=.2, label="Surface min–max")
    marker = ax_trace.axvline(frame_times[frame], color="tab:blue",
                              linestyle="--", linewidth=1)
    ax_trace.set(xlabel="Time since heating ends (s)", ylabel="Temperature rise (K)")
    ax_trace.grid(alpha=.2)
    ax_trace.legend(loc="best", fontsize=9)

    slider_ax = fig.add_axes((.22, .095, .65, .025))
    slider = Slider(slider_ax, "Frame", 0, max(nf-1, 1), valinit=frame,
                    valstep=1, valfmt="%0.0f")
    if nf == 1:
        slider.set_active(False)
    fig.text(.5, .035,
              "Both image color scales stay fixed across frames. Contrast subtracts each frame's spatial mean; it is not a defect-free reference.",
              ha="center", fontsize=9)

    def update_frame(index):
        index = int(index)
        if not 0 <= index < nf:
            raise ValueError("Frame index out of range.")
        image.set_data(video[index])
        difference.set_data(contrast[index]*1000)
        ax_temp.set_title(f"Surface video | τ = {frame_times[index]:.2f} s")
        ax_contrast.set_title("Spatial temperature contrast")
        marker.set_xdata([frame_times[index], frame_times[index]])
        return image, difference, marker

    def on_slider(value):
        update_frame(min(int(value), nf-1))
        fig.canvas.draw_idle()
    slider.on_changed(on_slider)
    update_frame(frame)
    # Keep widgets alive when callers retain only the figure.
    fig._thermoflow_slider = slider
    return fig, update_frame, slider


def load_sample(path):
    """Read saved physical times rather than guessing from frame count."""
    with np.load(path, allow_pickle=False) as data:
        if "frame_times" not in data:
            raise ValueError(f"{path}: missing frame_times; save recording times in the NPZ")
        if "cell_size" not in data:
            warnings.warn(f"{path}: missing cell_size; assuming 0.5 mm")
        return (Path(path).stem, data["video"].copy(), data["conductivity"].copy(),
                data["frame_times"].copy(),
                float(data["cell_size"]) if "cell_size" in data else .0005)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", type=Path, help="NPZ saved by construct_toy_dataset.py")
    parser.add_argument("--snapshot", type=Path, help="Save the selected frame as an image")
    parser.add_argument("--gif", type=Path, help="Save all frames as an animated GIF")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--fps", type=float, default=4,
                        help="Playback frames/second, not physical recording rate")
    parser.add_argument("--background-k", type=float, default=.20,
                        help="Background conductivity, W/(m K)")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or not np.isfinite(args.fps):
        parser.error("--fps must be finite and positive")
    if args.no_show and not (args.snapshot or args.gif):
        parser.error("--no-show requires --snapshot or --gif")
    if args.no_show:
        plt.switch_backend("Agg")
    _, video, field, times, spacing = load_sample(args.sample)
    fig, update, slider = visualize(video, field, times, spacing,
                                    args.background_k, args.frame)
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.snapshot, dpi=160)
        print(f"Saved snapshot: {args.snapshot}")
    if args.gif:
        args.gif.parent.mkdir(parents=True, exist_ok=True)
        slider.ax.set_visible(False)
        animation = FuncAnimation(fig, update, frames=len(video),
                                   interval=1000/args.fps, blit=False)
        animation.save(args.gif, writer=PillowWriter(fps=args.fps), dpi=100)
        slider.ax.set_visible(True)
        update(args.frame)
        print(f"Saved GIF: {args.gif}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
