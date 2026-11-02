"""Create a spatially clustered 8/1/1 split of single-cube ThermoFlow data.

python cluster_dataset.py dataset --output dataset_splits --seed 42
Uses standardized cube-center coordinates, not thermal videos. Requires NumPy,
SciPy, and Matplotlib. Input files are unchanged; output is a frozen snapshot.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2, ClusterError


def read_geometry(path, background):
    with np.load(path, allow_pickle=False) as data:
        field = np.array(data['conductivity'], dtype='<f4')
        h = float(data['cell_size'])
        if field.ndim != 3 or not np.isfinite(field).all() or not np.isfinite(h) or h <= 0:
            raise ValueError(f'{path}: invalid conductivity or spacing')
        occupied = np.argwhere(~np.isclose(field,background,rtol=1e-5,atol=1e-8))
        if not len(occupied):
            raise ValueError(f'{path}: no cube; this splitter requires exactly one fixed-size cube')
        lo, hi = occupied.min(axis=0), occupied.max(axis=0)+1
        size = hi-lo
        if not np.all(size == size[0]) or len(occupied) != int(np.prod(size)):
            raise ValueError(f'{path}: geometry is not one solid axis-aligned cube')
        # Every voxel within the bounding box is occupied, so this is one cube.
        center = ((lo+hi)/2*h*1000)[::-1]
        physical = json.loads(str(data['config_json'].item())) if 'config_json' in data else {}
        physical.pop('seed',None)
        physical.pop('solver',None)  # The original and cached solvers are equivalent.
        signature = (field.shape, h, tuple(size), json.dumps(physical,sort_keys=True))
        digest = hashlib.sha256(field.tobytes()+repr((field.shape,h)).encode()).hexdigest()
        return center, digest, signature


def make_split(dataset, output, seed=42, background=.20):
    if output.exists():
        raise FileExistsError(f'{output} already exists: choose a new folder to preserve frozen splits')
    paths = sorted(dataset.glob('sample_*.npz'))
    if not paths:
        raise ValueError(f'No completed sample files in {dataset}')
    rows, unique = [], {}
    signature = None
    for i,path in enumerate(paths):
        for attempt in range(3):
            try:
                center, digest, this_signature = read_geometry(path,background)
                break
            except OSError as error:
                print(f'Read failed for {path.name} (attempt {attempt+1}/3): {error}',flush=True)
                if attempt == 2:raise

        if signature is None:signature=this_signature
        if this_signature != signature:
            raise ValueError(f'{path}: mixed geometry sizes, acquisition settings or grids')
        if digest not in unique:unique[digest] = (len(unique),center)
        rows.append((path.name,digest,unique[digest][0],center))
        if (i+1)%10 == 0 or i+1 == len(paths):
            print(f'Read {i+1}/{len(paths)} specimens',flush=True)
    centers = np.array([item[1] for item in unique.values()])
    if len(centers) < 10:
        raise ValueError('At least 10 unique interiors are required')
    mean, scale = centers.mean(axis=0), centers.std(axis=0)
    scale[scale < 1e-12] = 1
    x = (centers-mean)/scale
    rng = np.random.default_rng(seed)
    best = None
    # Multiple k-means++ initializations reduce sensitivity to a poor starting set.
    for _ in range(30):
        try:
            c, _ = kmeans2(x,10,iter=100,minit='++',missing='raise',rng=rng)
        except ClusterError:
            continue
        distances = ((x[:,None,:]-c[None,:,:])**2).sum(axis=2)
        labels = distances.argmin(axis=1)
        if len(np.unique(labels)) != 10:continue
        inertia = float(distances[np.arange(len(x)),labels].sum())
        if best is None or inertia < best[0]:best=(inertia,c,labels)
    if best is None:raise RuntimeError('Could not form 10 nonempty clusters')
    inertia, c, _ = best
    # Canonical ordering gives readable IDs independent of the algorithm's labels.
    order = np.lexsort((c[:,2],c[:,1],c[:,0]))
    c = c[order]
    labels = ((x[:,None,:]-c[None,:,:])**2).sum(axis=2).argmin(axis=1)
    permutation = np.random.default_rng(seed).permutation(10)
    val, test = int(permutation[-2]), int(permutation[-1])
    split_for = {i:('validation' if i==val else 'test' if i==test else 'train') for i in range(10)}
    assignments = []
    for name,digest,index,center in rows:
        cluster = int(labels[index])
        assignments.append(dict(sample=name,geometry_sha256=digest,cluster=cluster,
                                split=split_for[cluster],x_mm=float(center[0]),
                                y_mm=float(center[1]),z_mm=float(center[2])))
    # Every duplicate interior must remain in exactly one partition.
    seen = {}
    for row in assignments:
        if row['geometry_sha256'] in seen:
            assert seen[row['geometry_sha256']] == row['split']
        seen[row['geometry_sha256']] = row['split']
    output.mkdir(parents=True)
    with (output/'assignments.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(assignments[0]))
        writer.writeheader();writer.writerows(assignments)
    counts={}
    for split in ('train','validation','test'):
        names=[row['sample'] for row in assignments if row['split']==split]
        (output/f'{split}.txt').write_text('\n'.join(names)+'\n')
        counts[split]=len(names)
    group_dir=output/'groups'
    group_dir.mkdir()
    for cluster in range(10):
        names=[row['sample'] for row in assignments if row['cluster']==cluster]
        (group_dir/f'cluster_{cluster:02d}.txt').write_text('\n'.join(names)+'\n')
    clusters=[]
    for i in range(10):
        items=[r for r in assignments if r['cluster']==i]
        clusters.append(dict(cluster=i,split=split_for[i],samples=len(items),
                             unique_geometries=int(np.sum(labels==i)),
                             center_mm=(c[i]*scale+mean).tolist()))
    summary=dict(method='k-means++ on standardized single-cube center (x,y,z)',
                 restarts=30,iterations=100,seed=seed,background_k=background,
                 dataset=str(dataset.resolve()),samples=len(rows),unique_geometries=len(centers),
                 duplicate_samples=len(rows)-len(centers),counts=counts,
                 validation_cluster=val,test_cluster=test,
                 feature_mean_mm=mean.tolist(),feature_scale_mm=scale.tolist(),
                 centroids_standardized=c.tolist(),inertia=inertia,clusters=clusters,
                 note='Spatial holdout, not an IID random split. Cluster sizes are unequal. '
                      'Lists contain filenames relative to dataset. Snapshot is frozen; newly '
                      'generated samples are not included. Do not refit splits after model evaluation.')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    plot(assignments,clusters,output/'split_overview.png')
    print(json.dumps({k:summary[k] for k in ('samples','unique_geometries','duplicate_samples','counts',
                                           'validation_cluster','test_cluster')},indent=2),flush=True)
    return summary


def plot(rows,clusters,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(13,5.5),layout='constrained')
    ax=fig.add_subplot(121,projection='3d')
    palette=plt.get_cmap('tab10')
    for c in clusters:
        group=[r for r in rows if r['cluster']==c['cluster']]
        a=np.array([[r['x_mm'],r['y_mm'],r['z_mm']] for r in group])
        ax.scatter(*a.T,color=palette(c['cluster']),s=18,alpha=.8,
                   label=f"{c['cluster']}: {c['samples']} ({c['split']})")
    ax.set(xlabel='Cube center x (mm)',ylabel='Cube center y (mm)',zlabel='Center depth (mm)',
           title='10 spatial clusters; duplicate interiors grouped')
    ax.invert_zaxis()
    ax.legend(fontsize=7,loc='upper left',bbox_to_anchor=(-.07,1.02))
    bx=fig.add_subplot(122)
    for split,color,marker in [('train','#4477aa','o'),('validation','#ee9944','^'),('test','#aa3377','s')]:
        group=[r for r in rows if r['split']==split]
        bx.scatter([r['x_mm'] for r in group],[r['y_mm'] for r in group],s=30,
                   color=color,marker=marker,alpha=.75,label=f'{split}: {len(group)}')
    bx.set(xlabel='Cube center x (mm)',ylabel='Cube center y (mm)',aspect='equal',
           title='8 clusters train / 1 validation / 1 test')
    bx.legend();bx.grid(alpha=.2)
    fig.suptitle('ThermoFlow — spatial holdout split',fontsize=14)
    fig.savefig(path,dpi=160);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--output',type=Path,default=Path('dataset_splits'))
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--background-k',type=float,default=.20)
    args=parser.parse_args()
    if args.seed<0 or not np.isfinite(args.background_k):parser.error('Invalid seed or background')
    make_split(args.dataset,args.output,args.seed,args.background_k)

if __name__=='__main__':main()
