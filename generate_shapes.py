"""Generate reproducible, two-material interiors and post-pulse thermal videos.

python generate_shapes.py --count-per-family 20 --workers 4 --output datasets/shapes
"""
# Limit numerical-library threads before imports; parallelism is across specimens.
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import ndimage

FAMILIES=('cubes','blocks','ellipsoids','compound','blobs')
SHAPE=(12,48,48)
VERSION=1


def geometry(family, seed):
    """Sample correlated structure, then translate it inside a two-cell skin."""
    rng=np.random.default_rng(seed)
    z,y,x=np.indices(SHAPE,dtype=float)
    points=np.stack((x-24,y-24,z-6),axis=-1)
    for attempt in range(1000):
        params={}
        if family=='cubes':
            side=int(rng.integers(3,7))
            mask=np.all((points>=-side/2)&(points<side/2),axis=-1)
            params['side_cells']=side
        elif family in ('blocks','ellipsoids','compound'):
            def primitive(kind, center):
                # A random orthogonal frame allows rotations about all three axes.
                rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
                q=(points-center)@rotation
                lengths=rng.uniform([3,2,2],[10,7,5])
                if kind=='blocks':m=np.all(np.abs(q)<lengths/2,axis=-1)
                else:m=np.sum((q/(lengths/2))**2,axis=-1)<1
                return m,dict(center_cells=np.asarray(center).tolist(),lengths_cells=lengths.tolist(),rotation=rotation.tolist())
            if family=='compound':
                mask=np.zeros(SHAPE,bool);parts=[]
                for _ in range(int(rng.integers(2,5))):
                    m,p=primitive('ellipsoids',rng.uniform([-3,-3,-1],[3,3,1]));mask|=m;parts.append(p)
                params['parts']=parts
            else:
                mask,p=primitive(family,np.zeros(3));params.update(p)
        elif family=='blobs':
            # Threshold a smooth random field inside a local envelope. Opening
            # removes one-voxel tendrils; tiny detached components are discarded.
            ell=float(rng.uniform(1.,2.2))
            smooth=ndimage.gaussian_filter(rng.normal(size=SHAPE),ell)
            radii=rng.uniform([4,4,2],[9,9,3.8])
            support=np.sum((points/radii)**2,axis=-1)<1
            fraction=float(rng.uniform(.35,.75))
            threshold=float(np.quantile(smooth[support],1-fraction))
            mask=ndimage.binary_opening((smooth>threshold)&support)
            labels,n=ndimage.label(mask)
            sizes=np.bincount(labels.ravel());sizes[0]=0
            mask=np.isin(labels,np.flatnonzero(sizes>=8))
            params.update(correlation_length_cells=ell,envelope_radii_cells=radii.tolist(),threshold_quantile=1-fraction)
        else:raise ValueError(f'Unknown family: {family}')
        coords=np.argwhere(mask)
        if not 32<=len(coords)<=512:continue
        lo=coords.min(0);hi=coords.max(0)+1;extent=hi-lo
        if np.any(extent>np.array(SHAPE)-4):continue
        crop=mask[tuple(slice(a,b) for a,b in zip(lo,hi))]
        lower=np.array([rng.integers(2,n-2-int(w)+1) for n,w in zip(SHAPE,extent)])
        out=np.zeros(SHAPE,bool)
        out[tuple(slice(a,a+w) for a,w in zip(lower,extent))]=crop
        # Record actual voxel geometry; a sampled primitive can split on this grid.
        n=int(ndimage.label(out)[1])
        return out,dict(family=family,seed=int(seed),attempts=attempt+1,parameters=params,
                        components=n,positive_voxels=int(out.sum()),front_depth_mm=float(lower[0]*.5),
                        bounding_box_cells_zyx=extent.tolist())
    raise RuntimeError(f'Could not sample {family} within the slab; seed={seed}')


def _job(job):
    family,index,seed,folder,dt,resume=job
    path=Path(folder)/family/f'sample_{index:04d}.npz'
    config=dict(generator='generate_shapes',generator_version=VERSION,family=family,seed=int(seed),
                dt=dt,heat_flux=800.,heat_capacity=1.5e6,pulse_seconds=5.,k_background=.2,k_cube=.5)
    mask,meta=geometry(family,seed)
    if resume and path.exists():
        with np.load(path,allow_pickle=False) as d:
            if (json.loads(d['config_json'].item())!=config or not np.array_equal(d['conductivity'],np.where(mask,.5,.2).astype('f4'))
                    or d['video'].shape!=(20,48,48) or not np.isfinite(d['video']).all()
                    or not np.array_equal(d['frame_times'],np.linspace(0,60,20)) or float(d['cell_size'])!=.0005):
                raise ValueError(f'Existing sample does not match this run: {path}')
        return dict(index=index,status='verified',seconds=0.,**meta,
                    geometry_sha256=hashlib.sha256(mask.astype('u1').tobytes()).hexdigest())
    if path.exists():raise FileExistsError(f'{path} exists; use --resume or a new output')
    from construct_toy_dataset import simulateVideo,FRAME_TIMES,CELL_SIZE
    started=perf_counter()
    field=np.where(mask,.5,.2)
    video,field=simulateVideo(conductivity=field,dt=dt,verbose=False)
    path.parent.mkdir(parents=True,exist_ok=True)
    # Write atomically so interrupted workers leave no apparently complete sample.
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,video=video,conductivity=field,frame_times=FRAME_TIMES,cell_size=CELL_SIZE,
                            config_json=json.dumps(config,sort_keys=True),geometry_json=json.dumps(meta,sort_keys=True))
    temporary.replace(path)
    return dict(index=index,status='saved',seconds=perf_counter()-started,**meta,
                geometry_sha256=hashlib.sha256(mask.astype('u1').tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('datasets/shapes'))
    p.add_argument('--count-per-family',type=int,default=20)
    p.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES))
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--seed',type=int,default=300000)
    p.add_argument('--dt',type=float,default=.5)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if min(a.count_per_family,a.workers)<1 or a.seed<0 or not np.isfinite(a.dt) or a.dt<=0:p.error('Invalid count, workers, seed or timestep')
    if len(set(a.families))!=len(a.families):p.error('Families must be unique')
    a.output.mkdir(parents=True,exist_ok=True)
    settings=dict(version=VERSION,families=a.families,count_per_family=a.count_per_family,seed=a.seed,dt=a.dt)
    settings_path=a.output/'generation_config.json'
    if settings_path.exists() and json.loads(settings_path.read_text())!=settings:
        p.error('Output belongs to a different run; use a new output directory')
    settings_path.write_text(json.dumps(settings,indent=2)+'\n')
    # Independent seed streams are stable across worker counts and family subsets.
    jobs=[(family,i,int(np.random.SeedSequence([a.seed,FAMILIES.index(family),i]).generate_state(1,dtype=np.uint64)[0]),str(a.output),a.dt,a.resume)
          for i in range(a.count_per_family) for family in a.families]
    start=perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for done,row in enumerate(pool.map(_job,jobs),1):
            rows.append(row)
            print(f"[{done}/{len(jobs)}] {row['family']}/sample_{row['index']:04d}: {row['status']}; "
                  f"{row['positive_voxels']} voxels, {row['components']} components, front={row['front_depth_mm']:.1f} mm; "
                  f"{row['seconds']:.1f}s; elapsed={perf_counter()-start:.1f}s",flush=True)
    summary=dict(**settings,elapsed_seconds=perf_counter()-start,samples=rows)
    (a.output/'generation_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Complete:',a.output.resolve(),flush=True)


if __name__=='__main__':main()
