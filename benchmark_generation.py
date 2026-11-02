"""Benchmark cached solver throughput and timestep errors on the actual slab.

python benchmark_generation.py --count 4 --output benchmark_results
This writes separate benchmark files, not the training dataset. Count must be >=4.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from construct_toy_dataset import generate_dataset, simulateVideo


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count',type=int,default=4)
    parser.add_argument('--output',type=Path,default=Path('benchmark_results'))
    args=parser.parse_args()
    if args.count<4:parser.error('--count must be at least 4')
    if args.output.exists():parser.error('Use a new output folder for unbiased timing')
    args.output.mkdir(parents=True)
    report={'throughput':[],'timesteps':[]}
    for workers in (1,2,4):
        started=perf_counter()
        generate_dataset(args.count,args.output/f'workers_{workers}',n_cubes=1,workers=workers)
        duration=perf_counter()-started
        row=dict(workers=workers,wall_seconds=duration,samples_per_minute=args.count*60/duration)
        report['throughput'].append(row);print(json.dumps(row),flush=True)
    # Worker count must not change the results or the seed/specimen mapping.
    for i in range(args.count):
        with np.load(args.output/'workers_1'/f'sample_{i:04d}.npz') as base:
            for workers in (2,4):
                with np.load(args.output/f'workers_{workers}'/f'sample_{i:04d}.npz') as test:
                    np.testing.assert_array_equal(base['conductivity'],test['conductivity'])
                    np.testing.assert_array_equal(base['video'],test['video'])
    # Select seeds that cover shallow, middle and deep permitted cube positions.
    seeds={}
    for seed in range(1000):
        rng=np.random.default_rng(seed)
        rng.integers(2,43);rng.integers(2,43);z=int(rng.integers(2,7))
        if z in (2,4,6) and z not in seeds:seeds[z]=seed
        if len(seeds)==3:break
    for z,seed in sorted(seeds.items()):
        base,_=simulateVideo(seed=seed,n_cubes=1,dt=.5,verbose=False)
        base_c=base-base.mean(axis=(1,2),keepdims=True)
        for dt in (1.,2.):
            started=perf_counter();video,_=simulateVideo(seed=seed,n_cubes=1,dt=dt,verbose=False)
            error=video-base
            ce=(video-video.mean(axis=(1,2),keepdims=True))-base_c
            row=dict(seed=seed,front_depth_mm=z*.5,dt=dt,wall_seconds=perf_counter()-started,
                     temperature_rmse_K=float(np.sqrt(np.mean(error**2))),
                     temperature_max_error_K=float(np.abs(error).max()),
                     initial_frame_rmse_K=float(np.sqrt(np.mean(error[0]**2))),
                     contrast_rmse_mK=float(np.sqrt(np.mean(ce**2)))*1000,
                     contrast_max_error_mK=float(np.abs(ce).max())*1000,
                     reference_peak_contrast_mK=float(np.abs(base_c).max())*1000)
            report['timesteps'].append(row);print(json.dumps(row),flush=True)
    path=args.output/'report.json';path.write_text(json.dumps(report,indent=2))
    print(f'Agreement across worker counts passed. Saved {path}',flush=True)

if __name__=='__main__':main()
