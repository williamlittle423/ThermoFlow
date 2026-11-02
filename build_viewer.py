"""Build an offline interactive HTML viewer from ThermoFlow NPZ samples.

python build_viewer.py dataset --output thermoflow_viewer.html --limit 100
Requires NumPy only. Re-run to include newly generated specimens.
"""
import argparse
import base64
import json
from pathlib import Path

import numpy as np


def load_sample(path, background):
    with np.load(path, allow_pickle=False) as data:
        video = np.asarray(data['video'], dtype=np.float32)
        field = np.asarray(data['conductivity'])
        times = np.asarray(data['frame_times'], dtype=float)
        h = float(data['cell_size'])*1000
        if (video.ndim != 3 or field.ndim != 3 or min(field.shape) < 1
                or min(video.shape) < 1 or video.shape[1:] != field.shape[1:]
                or times.shape != (len(video),) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or not np.isfinite(video).all()
                or not np.isfinite(field).all() or not np.isfinite(h) or h <= 0):
            raise ValueError(f'{path}: incompatible arrays or metadata')
        mask = ~np.isclose(field, background, rtol=1e-5, atol=1e-8)
        nz, ny, nx = field.shape
        faces = []
        # Export only exposed material faces; internal voxel faces are omitted.
        for z, y, x in np.argwhere(mask):
            for axis, sign in ((0,-1),(0,1),(1,-1),(1,1),(2,-1),(2,1)):
                adjacent = [int(z), int(y), int(x)]
                adjacent[axis] += sign
                if (all(0 <= adjacent[a] < field.shape[a] for a in range(3))
                        and mask[tuple(adjacent)]):
                    continue
                faces.append([int(x), int(y), int(z), axis, sign])
        contrast = video-video.mean(axis=(1,2), keepdims=True)
        depth = np.where(mask, np.arange(nz)[:,None,None]*h, np.inf).min(axis=0)
        depth[~np.isfinite(depth)] = -1
        occupied = np.argwhere(mask)
        return dict(name=path.stem, shape=list(video.shape), dims=[nx*h,ny*h,nz*h],
                    cell=h, times=times.tolist(), faces=faces,
                    video=base64.b64encode(video.astype('<f4').tobytes()).decode('ascii'),
                    depth=depth.tolist(),
                    min=float(video.min()), max=float(video.max()),
                    contrastMax=float(np.abs(contrast).max())*1000,
                    mean=video.mean(axis=(1,2)).tolist(),
                    rms=(np.sqrt((contrast**2).mean(axis=(1,2)))*1000).tolist(),
                    depthRange=([float(occupied[:,0].min()*h),float((occupied[:,0].max()+1)*h)]
                                if len(occupied) else None),
                    fraction=float(mask.mean()),
                    materials=[float(v) for v in np.unique(field)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--output', type=Path, default=Path('thermoflow_viewer.html'))
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--background-k', type=float, default=.20)
    args = parser.parse_args()
    if args.limit < 1 or not np.isfinite(args.background_k):
        parser.error('Require a positive limit and finite background conductivity')
    paths = sorted(args.dataset.glob('sample_*.npz'))[:args.limit]
    if not paths:
        parser.error('No sample_*.npz files found')
    samples = []
    for i,path in enumerate(paths):
        samples.append(load_sample(path,args.background_k))
        if (i+1)%10 == 0 or i+1 == len(paths):
            print(f'Loaded {i+1}/{len(paths)}',flush=True)
    template = Path(__file__).with_name('viewer_template.html').read_text()
    payload = json.dumps(dict(samples=samples,background=args.background_k),
                         separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(template.replace('/*DATA_PAYLOAD*/',payload))
    print(f'Saved {args.output.resolve()} ({args.output.stat().st_size/1e6:.1f} MB)',flush=True)


if __name__ == '__main__':
    main()
