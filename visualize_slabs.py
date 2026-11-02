"""Inspect samples saved by construct_toy_dataset.py, without importing FiPy.

Examples:
    python visualize_slabs.py sample_0000.npz
    python visualize_slabs.py shallow.npz --compare deep.npz --noise-std 0.05
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


def compare_samples(samples, background_k=.20, frame=0, noise_std=0,
                    noise_seed=0, contrast_limit_mk=None):
    """Synchronized comparison with shared scales across every case and time.

    Noise is synthetic, independent Gaussian pixel noise, added once to a copy.
    Spatial RMS and width describe surface contrast, not reconstruction accuracy.
    """
    times = np.asarray(samples[0][3], dtype=float)
    if (times.ndim != 1 or not len(times) or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0) or not 0 <= frame < len(times)):
        raise ValueError("Invalid frame times or starting frame")
    rng = np.random.default_rng(noise_seed)
    cases = []
    for name, video, field, t, spacing in samples:
        video, field = np.asarray(video, float), np.asarray(field, float)
        if (video.ndim != 3 or field.ndim != 3 or video.shape[1:] != field.shape[1:]
                or video.shape[0] != len(times) or np.shape(t) != times.shape
                or not np.allclose(t, times) or not np.isfinite(video).all()
                or not np.isfinite(field).all() or not np.isfinite(spacing) or spacing <= 0):
            raise ValueError("Samples need finite, compatible fields and identical frame times")
        # Preserve the clean simulation for a direct before/after noise comparison.
        measured = video + rng.normal(0, noise_std, video.shape) if noise_std else video.copy()
        clean = video - video.mean(axis=(1, 2), keepdims=True)
        contrast = measured - measured.mean(axis=(1, 2), keepdims=True)
        rms = np.sqrt(np.mean(clean**2, axis=(1, 2))) * 1000
        noisy_rms = np.sqrt(np.mean(contrast**2, axis=(1, 2))) * 1000
        # RMS radius of absolute contrast around its weighted centroid (mm).
        yy, xx = np.indices(video.shape[1:], dtype=float)
        xx, yy = (xx + .5)*spacing*1000, (yy + .5)*spacing*1000
        weights = np.abs(clean)
        total = weights.sum(axis=(1, 2))
        denom = np.maximum(total, np.finfo(float).tiny)
        cx = (weights*xx).sum(axis=(1, 2))/denom
        cy = (weights*yy).sum(axis=(1, 2))/denom
        radius = np.sqrt((weights*((xx-cx[:, None, None])**2 +
                                   (yy-cy[:, None, None])**2)).sum(axis=(1, 2))/denom)
        radius[total < 1e-12] = np.nan
        mask = ~np.isclose(field, background_k, rtol=1e-5, atol=1e-8)
        depths = np.where(mask, np.arange(field.shape[0])[:, None, None]*spacing*1000, np.inf)
        depth_map = depths.min(axis=0)
        depth_map[~np.isfinite(depth_map)] = np.nan
        cases.append((name, measured, contrast*1000, rms, noisy_rms, radius,
                      depth_map, spacing, field.shape[0]))
    limit = contrast_limit_mk or max(1e-3, max(np.abs(c[2]).max() for c in cases))
    low, high = min(c[1].min() for c in cases), max(c[1].max() for c in cases)
    if high-low < 1e-8:
        low, high = low-.01, high+.01
    depth_max = max(c[7]*c[8]*1000 for c in cases)
    rms_max = max(max(c[3].max(), c[4].max()) for c in cases)
    fig, axes = plt.subplots(len(cases), 4, squeeze=False,
                             figsize=(18, max(5.5, 3.8*len(cases))))
    fig.subplots_adjust(left=.045, right=.95, bottom=.19, top=.86, wspace=.65, hspace=.55)
    fig.suptitle(f"ThermoFlow | synchronized surface comparison | noise σ = {noise_std:g} K", y=.98)
    fig.text(.5, .925, "Shared image scales across all samples and frames. Ground-truth depth is not an observed input.", ha="center")
    artists, cursors = [], []
    for row, c in zip(axes, cases):
        name, measured, contrast, rms, noisy_rms, radius, depth_map, spacing, nz = c
        ny, nx = measured.shape[1:]
        extent = (0, nx*spacing*1000, 0, ny*spacing*1000)
        depth_image = row[0].imshow(depth_map, origin="lower", extent=extent,
                                    vmin=0, vmax=depth_max, cmap="viridis_r")
        row[0].set_title(f"{name}\nNearest inclusion depth (ground truth)")
        a = row[1].imshow(measured[frame], origin="lower", extent=extent,
                           vmin=low, vmax=high, cmap="inferno")
        b = row[2].imshow(contrast[frame], origin="lower", extent=extent,
                           vmin=-limit, vmax=limit, cmap="RdBu_r")
        for ax in row[:3]:
            ax.set(xlabel="x (mm)", ylabel="y (mm)")
        fig.colorbar(depth_image, ax=row[0], shrink=.75, label="Depth (mm)")
        fig.colorbar(a, ax=row[1], shrink=.75, label="Temperature rise (K)")
        fig.colorbar(b, ax=row[2], shrink=.75, label="Contrast (mK)")
        ax = row[3]
        ax.plot(times, rms, label="Clean RMS contrast", color="tab:blue")
        if noise_std:
            ax.plot(times, noisy_rms, label="With noise", color="tab:orange", alpha=.8)
        ax.set(xlabel="Time after heating (s)", ylabel="Spatial RMS contrast (mK)",
               ylim=(0, max(rms_max*1.15, .001)))
        ax.grid(alpha=.2)
        twin = ax.twinx()
        twin.plot(times, radius, "--", color="tab:green", label="Clean contrast radius")
        twin.set_ylabel("Contrast RMS radius (mm)", color="tab:green")
        twin.set_ylim(0, max(np.hypot(d[1].shape[1], d[1].shape[2])*d[7]*1000 for d in cases)/2)
        lines = ax.lines + twin.lines
        ax.legend(lines, [line.get_label() for line in lines], fontsize=7, loc="best")
        peak = int(np.argmax(rms))
        ax.set_title(f"Clean peak: {rms[peak]:.2f} mK at {times[peak]:.2f} s")
        print(f"{name}: peak spatial RMS={rms[peak]:.4g} mK at τ={times[peak]:.4g} s; "
              f"clean RMS radius={radius[peak]:.4g} mm; assumed pixel noise σ={noise_std*1000:g} mK")
        cursor = ax.axvline(times[frame], color="gray", linestyle=":")
        artists.append((a, b, row[1], row[2], measured, contrast))
        cursors.append(cursor)
    slider = Slider(fig.add_axes((.22, .085, .60, .025)), "Frame", 0,
                    max(1, len(times)-1), valinit=frame, valstep=1, valfmt="%0.0f")
    if len(times) == 1:
        slider.set_active(False)
    fig.text(.5, .025, "Contrast subtracts the frame mean, not a defect-free reference. Radius is descriptive; noise RMS is not a detection threshold.",
             ha="center", fontsize=9)
    def update(index):
        index = int(index)
        if not 0 <= index < len(times):
            raise ValueError("Frame index out of range")
        result = []
        for a, b, ta, tb, measured, contrast in artists:
            a.set_data(measured[index]); b.set_data(contrast[index])
            ta.set_title(f"Surface | τ = {times[index]:.2f} s")
            tb.set_title("Spatial contrast")
            result.extend((a, b))
        for cursor in cursors:
            cursor.set_xdata([times[index]]*2)
        return tuple(result + cursors)
    def on_slider(value):
        update(min(int(value), len(times)-1))
        fig.canvas.draw_idle()
    slider.on_changed(on_slider)
    update(frame)
    fig._thermoflow_slider = slider
    return fig, update, slider


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", type=Path, help="NPZ saved by simulate.py")
    parser.add_argument("--compare", type=Path, nargs="+",
                        help="Additional samples; synchronized views share color scales")
    parser.add_argument("--noise-std", type=float, default=0,
                        help="Independent Gaussian pixel noise standard deviation, K")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--contrast-limit-mk", type=float,
                        help="Fixed symmetric contrast limit for repeatable comparisons")
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
    if not np.isfinite(args.noise_std) or args.noise_std < 0:
        parser.error("--noise-std must be finite and nonnegative")
    if args.contrast_limit_mk is not None and (
            not np.isfinite(args.contrast_limit_mk) or args.contrast_limit_mk <= 0):
        parser.error("--contrast-limit-mk must be finite and positive")
    samples = [load_sample(path) for path in [args.sample, *(args.compare or [])]]
    fig, update, slider = compare_samples(
        samples, args.background_k, args.frame, args.noise_std,
        args.noise_seed, args.contrast_limit_mk)
    video = samples[0][1]
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
