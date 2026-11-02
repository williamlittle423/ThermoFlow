"""Run single-, double-, and 3–10-cube generation concurrently.

python benchmark_cluster.py --samples 50 --workers-per-group 2 --output cluster_benchmark
Run on one allocated compute node. Requests no scheduler resources itself.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
from time import perf_counter, sleep

import numpy as np


def stop_jobs(jobs):
    for job in jobs:
        proc=job['process']
        if proc.poll() is None:
            if os.name=='posix':os.killpg(proc.pid,signal.SIGTERM)
            else:proc.terminate()
    for job in jobs:
        proc=job['process']
        try:proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name=='posix':os.killpg(proc.pid,signal.SIGKILL)
            else:proc.kill()
            proc.wait()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=50,help='Specimens per group')
    parser.add_argument('--workers-per-group',type=int,default=2)
    parser.add_argument('--target',type=int,default=2500,help='Projected final specimens per group')
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path,default=Path('cluster_benchmark'))
    parser.add_argument('--nx',type=int,default=48)
    parser.add_argument('--ny',type=int,default=48)
    parser.add_argument('--nz',type=int,default=12)
    parser.add_argument('--cube-cells',type=int,default=4)
    parser.add_argument('--dt',type=float,default=.5)
    args=parser.parse_args()
    if min(args.samples,args.workers_per_group,args.target,args.nx,args.ny,args.nz,args.cube_cells)<1 or args.seed<0:
        parser.error('Counts/dimensions must be positive and seed nonnegative')
    if args.nz<2 or not np.isfinite(args.dt) or args.dt<=0:parser.error('Require nz >= 2 and finite dt > 0')
    if args.target<args.samples:parser.error('--target must be at least --samples')
    if args.output.exists():parser.error('Use a new --output directory; existing samples invalidate timing')
    script=Path(__file__).with_name('construct_toy_dataset.py').resolve()
    if not script.is_file():parser.error(f'Missing generator: {script}')
    args.output.mkdir(parents=True)
    # Set before child startup, including the one-worker benchmark case.
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env['PYTHONUNBUFFERED']='1'
    groups=[('single_cube',['--cubes','1']),('two_cubes',['--cubes','2']),
            ('mixed_cubes',['--cube-count-range','3','10'])]
    jobs=[]
    report=dict(started_utc=datetime.now(timezone.utc).isoformat(),
                samples_per_group=args.samples,workers_per_group=args.workers_per_group,
                total_solver_workers=3*args.workers_per_group,target_per_group=args.target,
                note='Three groups run concurrently on one node. Extrapolations include startup '
                     'and assume comparable load, memory and I/O for the longer run.',groups=[])
    if hasattr(os,'sched_getaffinity'):report['visible_cpu_affinity']=len(os.sched_getaffinity(0))
    started=perf_counter()
    print(f'Starting 3 groups × {args.samples} specimens, {args.workers_per_group} workers each. '
          f'Use an allocation with at least {3*args.workers_per_group} CPU cores plus controller overhead.',flush=True)
    def save_report():
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for i,(name,mode) in enumerate(groups):
            folder=(args.output/name).resolve()
            command=[sys.executable,'-u',str(script),'--count',str(args.samples),
                     '--workers',str(args.workers_per_group),'--seed',str(args.seed+i*100000),
                     '--output',str(folder),'--nx',str(args.nx),'--ny',str(args.ny),
                     '--nz',str(args.nz),'--cube-cells',str(args.cube_cells),'--dt',str(args.dt),*mode]
            log=args.output/f'{name}.log'
            handle=log.open('w')
            try:proc=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,
                                      env=env,start_new_session=(os.name=='posix'))
            except BaseException:
                handle.close();raise
            jobs.append(dict(name=name,folder=folder,command=command,log=str(log),
                             handle=handle,process=proc,started=perf_counter(),done=False))
            print(f'{name}: logging to {log}',flush=True)
        while not all(job['done'] for job in jobs):
            for job in jobs:
                code=job['process'].poll()
                if job['done'] or code is None:continue
                elapsed=perf_counter()-job['started'];job['done']=True;job['handle'].close()
                files=sorted(job['folder'].glob('sample_*.npz'))
                row=dict(group=job['name'],returncode=code,wall_seconds=elapsed,
                         completed_samples=len(files),log=job['log'],command=job['command'])
                if code==0 and len(files)==args.samples:
                    counts=Counter()
                    for path in files:
                        with np.load(path,allow_pickle=False) as data:counts[int(data['n_cubes'])]+=1
                    resume=job['command'].copy();resume[resume.index('--count')+1]=str(args.target);resume.append('--resume')
                    row.update(status='complete',samples_per_minute=args.samples*60/elapsed,
                               projected_total_hours=elapsed/args.samples*args.target/3600,
                               projected_remaining_hours=elapsed/args.samples*(args.target-args.samples)/3600,
                               cube_count_histogram=dict(sorted(counts.items())),resume_command=shlex.join(resume))
                    print(f"{job['name']}: {elapsed:.1f} s, {row['samples_per_minute']:.2f} specimens/min; "
                          f"{args.target}-specimen estimate {row['projected_total_hours']:.2f} h",flush=True)
                else:
                    row['status']='failed'
                    print(f"{job['name']} failed (exit {code}); inspect {job['log']}",flush=True)
                report['groups'].append(row);save_report()
            sleep(.5)
        report['benchmark_wall_seconds']=perf_counter()-started
        successful=[r for r in report['groups'] if r['status']=='complete']
        if len(successful)==3:
            report['projected_concurrent_total_hours']=max(r['projected_total_hours'] for r in successful)
            print(f"All 3 groups complete. Concurrent {args.target}-per-group estimate: "
                  f"{report['projected_concurrent_total_hours']:.2f} h",flush=True)
            print('To extend these same folders, launch the three resume commands concurrently:',flush=True)
            for row in successful:print(row['resume_command'],flush=True)
        save_report()
        print(f"Report: {args.output/'report.json'}",flush=True)
        return 0 if len(successful)==3 else 1
    except BaseException:
        stop_jobs(jobs)
        report['interrupted']=True;save_report();raise
    finally:
        for job in jobs:job['handle'].close()


if __name__=='__main__':
    raise SystemExit(main())
