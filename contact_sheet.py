"""Browse a thermal-tomography dataset without importing FiPy.

Examples (run in the ThermoFlow project folder):
    python contact_sheet.py dataset
    python contact_sheet.py dataset --per-page 20 --columns 5 --frame 3
    python contact_sheet.py dataset --export-all contact_sheets --no-show

Dependencies: numpy, matplotlib. Each tile pairs surface contrast with the
nearest inclusion depth at each (x,y). Depth is ground truth, not camera data.
"""
import argparse
from math import ceil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.widgets import Button, Slider


def load_samples(folder, limit=None, background_k=.20):
    """Take a snapshot of completed files and compute fixed dataset-wide scales."""
    paths = sorted(folder.glob("sample_*.npz"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No sample_*.npz files found in {folder}")
    samples = []
    times = None
    contrast_max, depth_max = 0., 0.
    for i, path in enumerate(paths):
        try:
            with np.load(path, allow_pickle=False) as data:
                video = np.array(data["video"], dtype=np.float32)
                field = np.asarray(data["conductivity"])
                t = np.asarray(data["frame_times"], dtype=float)
                spacing = float(data["cell_size"])
                if (video.ndim != 3 or field.ndim != 3
                        or min(video.shape) < 1 or min(field.shape) < 1
                        or video.shape[1:] != field.shape[1:]
                        or t.shape != (len(video),) or not np.isfinite(t).all()
                        or np.any(np.diff(t) <= 0) or not np.isfinite(video).all()
                        or not np.isfinite(field).all()
                        or not np.isfinite(spacing) or spacing <= 0):
                    raise ValueError("Invalid arrays or recording metadata")
                if times is None:
                    times = t.copy()
                elif t.shape != times.shape or not np.allclose(t, times):
                    raise ValueError("All specimens must have the same frame times")
                # Contrast removes uniform warming; it is not a reference subtraction.
                contrast = (video - video.mean(axis=(1, 2), keepdims=True))*1000
                mask = ~np.isclose(field, background_k, rtol=1e-5, atol=1e-8)
                depth = np.where(mask, np.arange(field.shape[0])[:, None, None]
                                 * spacing*1000, np.inf).min(axis=0)
                depth[~np.isfinite(depth)] = np.nan
                finite_depths = depth[np.isfinite(depth)]
                depth_label = (f"front depth {finite_depths.min():g}–{finite_depths.max():g} mm"
                               if len(finite_depths) else "No inclusions")
                ny, nx = video.shape[1:]
                samples.append(dict(path=path, contrast=contrast, depth=depth,
                                    extent=(0, nx*spacing*1000, 0, ny*spacing*1000),
                                    depth_label=depth_label))
                contrast_max = max(contrast_max, float(np.abs(contrast).max()))
                depth_max = max(depth_max, field.shape[0]*spacing*1000)
        except Exception as error:
            raise ValueError(f"Cannot read {path}: {error}") from error
        if (i+1) % 10 == 0 or i+1 == len(paths):
            print(f"Loaded {i+1}/{len(paths)} specimens", flush=True)
    return samples, times, max(contrast_max, .001), depth_max


class ContactSheet:
    """A fixed color scale spans all loaded specimens and recording times."""
    def __init__(self, samples, times, contrast_limit, depth_max,
                 columns=4, per_page=12, frame=0):
        self.samples, self.times = samples, times
        self.per_page, self.page, self.frame = per_page, 0, frame
        self.pages = ceil(len(samples)/per_page)
        self.limit = contrast_limit
        rows = ceil(min(per_page, len(samples))/columns)
        columns = min(columns, len(samples))
        self.fig = plt.figure(figsize=(columns*4, rows*3.2+1.8))
        grid = self.fig.add_gridspec(rows, columns, left=.025, right=.985,
                                     bottom=.23, top=.87, wspace=.18, hspace=.5)
        self.tiles = []
        for slot in range(min(per_page, len(samples))):
            sub = grid[slot//columns, slot % columns].subgridspec(1, 2, wspace=.08)
            axes = [self.fig.add_subplot(sub[0, j]) for j in range(2)]
            images = [axes[0].imshow(np.zeros((2, 2)), origin="lower", cmap="RdBu_r",
                                     vmin=-contrast_limit, vmax=contrast_limit,
                                     interpolation="nearest"),
                      axes[1].imshow(np.zeros((2, 2)), origin="lower", cmap="viridis_r",
                                     vmin=0, vmax=depth_max, interpolation="nearest")]
            for ax in axes:
                ax.set_xticks([]); ax.set_yticks([])
            self.tiles.append((axes, images))
        # Separate bars explain the two physically different quantities.
        self.fig.colorbar(ScalarMappable(norm=Normalize(-contrast_limit, contrast_limit), cmap="RdBu_r"),
                          cax=self.fig.add_axes((.10, .16, .33, .015)), orientation="horizontal",
                          label="Surface contrast (mK)")
        self.fig.colorbar(ScalarMappable(norm=Normalize(0, depth_max), cmap="viridis_r"),
                          cax=self.fig.add_axes((.57, .16, .33, .015)), orientation="horizontal",
                          label="Depth (mm); white = background")
        self.title = self.fig.suptitle("", y=.98, fontsize=14)
        self.fig.text(.5, .925,
                      "Each pair: surface contrast | ground-truth depth. All maps show the heated face from above; x →, y ↑.",
                      ha="center", fontsize=10)
        slider_ax = self.fig.add_axes((.25, .055, .5, .025))
        self.slider = Slider(slider_ax, "Frame", 0, max(1, len(times)-1),
                             valinit=frame, valstep=1, valfmt="%0.0f")
        self.slider.on_changed(self.set_frame)
        if len(times) == 1:
            self.slider.set_active(False)
        self.previous = Button(self.fig.add_axes((.04, .045, .12, .045)), "Previous page")
        self.next = Button(self.fig.add_axes((.84, .045, .12, .045)), "Next page")
        self.previous.on_clicked(lambda event: self.set_page(self.page-1))
        self.next.on_clicked(lambda event: self.set_page(self.page+1))
        self.fig.text(.5, .012, "Contrast scales are fixed across all pages and times. Depth maps hide deeper overlapping inclusions.",
                      ha="center", fontsize=9)
        self.refresh()

    def set_frame(self, value):
        self.frame = min(int(value), len(self.times)-1)
        self.refresh()

    def set_page(self, page):
        self.page = max(0, min(page, self.pages-1))
        self.refresh()

    def refresh(self):
        start = self.page*self.per_page
        for slot, (axes, images) in enumerate(self.tiles):
            i = start+slot
            for ax in axes:
                ax.set_visible(i < len(self.samples))
            if i >= len(self.samples):
                continue
            sample = self.samples[i]
            images[0].set_data(sample["contrast"][self.frame])
            images[1].set_data(sample["depth"])
            for ax, image in zip(axes, images):
                image.set_extent(sample["extent"])
                ax.set_xlim(sample["extent"][:2]); ax.set_ylim(sample["extent"][2:])
            axes[0].set_title(sample["path"].stem, fontsize=9, loc="left")
            axes[1].set_title(sample["depth_label"], fontsize=8)
        self.title.set_text(f"ThermoFlow | {len(self.samples)} specimens | page {self.page+1}/{self.pages}"
                            f" | τ = {self.times[self.frame]:.2f} s")
        self.fig.canvas.draw_idle()

    def save(self, path):
        """Export the current view without interactive controls."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        widgets = (self.slider.ax, self.previous.ax, self.next.ax)
        try:
            for ax in widgets:
                ax.set_visible(False)
            self.fig.savefig(path, dpi=150)
        finally:
            for ax in widgets:
                ax.set_visible(True)
        print(f"Saved {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path, nargs="?", default=Path("dataset"))
    parser.add_argument("--columns", type=int, default=4, help="Specimen pairs per row")
    parser.add_argument("--per-page", type=int, default=12, help="Specimens per page")
    parser.add_argument("--limit", type=int, help="Load only the first N files")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--page", type=int, default=1, help="Initial page, counting from 1")
    parser.add_argument("--background-k", type=float, default=.20)
    parser.add_argument("--contrast-limit-mk", type=float, help="Optional fixed symmetric color limit")
    parser.add_argument("--snapshot", type=Path, help="Save the selected page and frame")
    parser.add_argument("--export-all", type=Path, help="Save every page as PNG at the selected frame")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.columns < 1 or args.per_page < 1 or args.page < 1 or args.frame < 0 or (args.limit is not None and args.limit < 1):
        parser.error("Counts and page must be positive; frame must be nonnegative")
    if not np.isfinite(args.background_k):
        parser.error("--background-k must be finite")
    if args.contrast_limit_mk is not None and (not np.isfinite(args.contrast_limit_mk) or args.contrast_limit_mk <= 0):
        parser.error("--contrast-limit-mk must be finite and positive")
    if args.no_show and not (args.snapshot or args.export_all):
        parser.error("--no-show requires --snapshot or --export-all")
    if args.no_show:
        plt.switch_backend("Agg")
    samples, times, contrast_max, depth_max = load_samples(args.dataset, args.limit, args.background_k)
    if args.frame >= len(times) or args.page > ceil(len(samples)/args.per_page):
        parser.error("Requested frame or page is out of range")
    sheet = ContactSheet(samples, times, args.contrast_limit_mk or contrast_max,
                         depth_max, args.columns, args.per_page, args.frame)
    sheet.set_page(args.page-1)
    if args.snapshot:
        sheet.save(args.snapshot)
    if args.export_all:
        for page in range(sheet.pages):
            sheet.set_page(page)
            sheet.save(args.export_all / f"contact_frame_{args.frame:02d}_page_{page+1:03d}.png")
        sheet.set_page(args.page-1)
    if not args.no_show:
        plt.show()
    plt.close(sheet.fig)


if __name__ == "__main__":
    main()
