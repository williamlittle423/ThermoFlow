"""Train ThermoNet on single-cube thermal videos.

python train_thermonet.py --data datasets/single_cube --output runs/single_cube --cache
Requires NumPy and PyTorch. Does not import FiPy or read the old spatial split.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from thermonet import ThermoNet


def atomic_json(path, value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    os.replace(temporary,path)


def atomic_checkpoint(path, state):
    temporary=path.with_name(path.name+'.tmp')
    torch.save(state,temporary)
    os.replace(temporary,path)


def load_pair(path, background, inclusion):
    with np.load(path,allow_pickle=False) as data:
        video=np.array(data['video'],dtype=np.float32)
        field=np.asarray(data['conductivity'])
        times=np.asarray(data['frame_times'],dtype=float)
        spacing=float(data['cell_size'])
        config=json.loads(str(data['config_json'].item())) if 'config_json' in data else {}
    if (video.ndim!=3 or field.ndim!=3 or min(field.shape)<1 or min(video.shape)<1 or min(video.shape[1:])<4
            or video.shape[1:]!=field.shape[1:] or times.shape!=(len(video),)
            or not np.isfinite(video).all() or not np.isfinite(field).all()
            or not np.isfinite(times).all() or np.any(np.diff(times)<=0)
            or not np.isfinite(spacing) or spacing<=0):
        raise ValueError(f'{path}: invalid arrays or acquisition metadata')
    target=np.isclose(field,inclusion,rtol=1e-5,atol=1e-8)
    if not np.all(target | np.isclose(field,background,rtol=1e-5,atol=1e-8)):
        raise ValueError(f'{path}: conductivity does not match the two specified materials')
    occupied=np.argwhere(target)
    if not len(occupied):raise ValueError(f'{path}: no inclusion')
    lo,hi=occupied.min(axis=0),occupied.max(axis=0)+1
    if not np.all(hi-lo==(hi-lo)[0]) or len(occupied)!=int(np.prod(hi-lo)):
        raise ValueError(f'{path}: expected exactly one solid axis-aligned cube')
    # Different acquisition settings must not silently enter the same benchmark.
    physical={k:config[k] for k in ('dt','heat_flux','heat_capacity','pulse_seconds',
                                   'k_background','k_cube') if k in config}
    signature=dict(video_shape=list(video.shape),target_shape=list(field.shape),
                   cell_size=spacing,frame_times=times.tolist(),
                   cube_cells=int((hi-lo)[0]),physical=physical)
    return video,target.astype(np.uint8),signature


def index_dataset(folder, background, inclusion, limit, cache):
    paths=sorted(folder.glob('sample_*.npz'))
    if limit is not None:paths=paths[:limit]
    if not paths:raise ValueError(f'No sample_*.npz files found in {folder}')
    records=[];cached={};signature=None
    for i,path in enumerate(paths):
        video,target,current=load_pair(path,background,inclusion)
        if signature is None:signature=current
        if current!=signature:raise ValueError(f'{path}: inconsistent grid, cube size or acquisition')
        geometry=hashlib.sha256(target.tobytes()).hexdigest()
        content=hashlib.sha256(video.astype('<f4').tobytes()+target.tobytes()).hexdigest()
        records.append(dict(sample=path.name,geometry_sha256=geometry,
                            content_sha256=content,positive_voxels=int(target.sum())))
        if cache:cached[path.name]=(video,target)
        if (i+1)%100==0 or i+1==len(paths):print(f'Indexed {i+1}/{len(paths)} specimens',flush=True)
    return records,cached,signature


def split_records(records, seed):
    groups={}
    for record in records:groups.setdefault(record['geometry_sha256'],[]).append(record['sample'])
    keys=sorted(groups)
    if len(keys)<10:raise ValueError('Need at least 10 unique interiors for train/validation/test splitting')
    np.random.default_rng(seed).shuffle(keys)
    ntrain=int(.70*len(keys));nval=max(1,int(.15*len(keys)))
    partitions={}
    for split,chosen in [('train',keys[:ntrain]),('validation',keys[ntrain:ntrain+nval]),
                         ('test',keys[ntrain+nval:])]:
        partitions[split]=sorted(name for key in chosen for name in groups[key])
    return partitions


class ThermalDataset(Dataset):
    def __init__(self, folder, names, cached, background, inclusion, normalization,
                 noise_std=0., seed=42, training=False):
        self.folder=folder;self.names=names;self.cached=cached
        self.background=background;self.inclusion=inclusion
        self.normalization=normalization;self.noise_std=noise_std
        self.seed=seed;self.training=training;self.epoch=0

    def __len__(self):return len(self.names)

    def raw_pair(self, name):
        if name in self.cached:return self.cached[name]
        video,target,_=load_pair(self.folder/name,self.background,self.inclusion)
        return video,target

    def __getitem__(self, index):
        name=self.names[index];video,target=self.raw_pair(name)
        x=torch.from_numpy(video.copy())
        if self.noise_std:
            # Fresh training noise each epoch; fixed evaluation noise. This is
            # independent of worker scheduling, sample order and resumed runs.
            key=f'{self.seed}:{self.epoch if self.training else 0}:{name}'
            noise_seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
            generator=torch.Generator().manual_seed(noise_seed)
            x=x+self.noise_std*torch.randn(x.shape,generator=generator)
        x=(x-self.normalization['mean_K'])/self.normalization['std_K']
        return x,torch.from_numpy(target.copy()).float(),name


def training_statistics(dataset):
    total=squared=0.;count=positive=0
    for name in dataset.names:
        video,target=dataset.raw_pair(name)
        total+=float(video.sum(dtype=np.float64))
        squared+=float(np.square(video,dtype=np.float64).sum())
        count+=video.size;positive+=int(target.sum())
    mean=total/count;std=float(np.sqrt(max(0.,squared/count-mean*mean)))
    if std<1e-12:raise ValueError('Training videos have zero temperature variation')
    voxels=len(dataset)*int(np.prod(dataset.raw_pair(dataset.names[0])[1].shape))
    return dict(mean_K=mean,std_K=std),float((voxels-positive)/positive)


def loss_and_metrics(logits, target, bce, dice_weight, conductivity_span, spacing_mm):
    probabilities=torch.sigmoid(logits)
    axes=(1,2,3)
    intersection=(probabilities*target).sum(dim=axes)
    soft_dice=(2*intersection+1e-6)/(probabilities.sum(dim=axes)+target.sum(dim=axes)+1e-6)
    loss=bce(logits,target)+dice_weight*(1-soft_dice).mean()
    # Thresholded structural metrics are per specimen, then averaged equally.
    with torch.no_grad():
        predicted=probabilities>=.5;truth=target.bool()
        tp=(predicted & truth).sum(dim=axes).float()
        p=predicted.sum(dim=axes).float();t=truth.sum(dim=axes).float()
        z=torch.arange(target.shape[1],device=target.device,dtype=target.dtype)+.5
        pred_z=(predicted.float().sum(dim=(2,3))*z).sum(dim=1)/p.clamp_min(1)*spacing_mm
        true_z=(target.sum(dim=(2,3))*z).sum(dim=1)/t*spacing_mm
        depth_error=torch.where(p>0,(pred_z-true_z).abs(),torch.full_like(p,float('nan')))
        front=torch.where(truth.any(dim=(2,3)),z-.5,torch.full_like(z,float('inf'))).min(dim=1).values*spacing_mm
        values=dict(dice=2*tp/(p+t).clamp_min(1),iou=tp/(p+t-tp).clamp_min(1),
                    precision=tp/p.clamp_min(1),recall=tp/t,
                    conductivity_mse=(conductivity_span**2*(probabilities-target).square()).mean(dim=axes),
                    predicted_nonempty=(p>0).float(),depth_error_mm=depth_error,
                    true_front_depth_mm=front)
        metrics={k:v.detach().cpu().tolist() for k,v in values.items()}
    return loss,metrics


def run_epoch(model, loader, device, bce, args, spacing_mm, optimizer=None):
    model.train(optimizer is not None)
    loss_total=0.;rows=[];count=0
    for videos,targets,names in loader:
        videos=videos.to(device,non_blocking=True);targets=targets.to(device,non_blocking=True)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(videos)
            loss,metrics=loss_and_metrics(logits,targets,bce,args.dice_weight,
                                         args.inclusion_k-args.background_k,spacing_mm)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss; inspect input scaling and learning rate')
            if optimizer is not None:
                loss.backward();optimizer.step()
        size=len(names);loss_total+=float(loss.detach())*size;count+=size
        for i,name in enumerate(names):
            row=dict(sample=name)
            row.update({k:(None if not np.isfinite(v[i]) else float(v[i])) for k,v in metrics.items()})
            rows.append(row)
    summary={'loss':loss_total/count,'samples':count}
    for key in rows[0]:
        if key in ('sample','true_front_depth_mm'):continue
        values=[row[key] for row in rows if row[key] is not None]
        summary[key]=float(np.mean(values)) if values else None
    summary['depth_error_evaluated_samples']=sum(r['depth_error_mm'] is not None for r in rows)
    return summary,rows


def select_device(requested):
    if requested=='auto':
        if torch.cuda.is_available():return torch.device('cuda')
        if torch.backends.mps.is_available():return torch.device('mps')
        return torch.device('cpu')
    if requested=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable; enable a GPU runtime or use --device cpu')
    if requested=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS is unavailable')
    return torch.device(requested)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('datasets/single_cube'))
    parser.add_argument('--output',type=Path,default=Path('runs/single_cube'))
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--base-channels',type=int,default=32)
    parser.add_argument('--patience',type=int,default=20,help='Early stopping on validation loss; 0 disables')
    parser.add_argument('--dice-weight',type=float,default=1.)
    parser.add_argument('--noise-std',type=float,default=0.,help='Gaussian observation noise, K, applied before normalization')
    parser.add_argument('--background-k',type=float,default=.20)
    parser.add_argument('--inclusion-k',type=float,default=.50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','cuda','mps'],default='auto')
    parser.add_argument('--num-workers',type=int,default=0,help='Data-loading workers, independent of simulation workers')
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--cache',action='store_true',help='Cache clean videos and masks in system RAM')
    parser.add_argument('--limit',type=int,help='Use the first N files for a pipeline smoke test')
    parser.add_argument('--resume',type=Path,help='Resume a last.pt checkpoint using matching training settings')
    args=parser.parse_args()
    if (min(args.epochs,args.batch_size,args.base_channels,args.cpu_threads)<1 or args.patience<0
            or args.num_workers<0 or args.seed<0 or (args.limit is not None and args.limit<10)
            or not np.isfinite([args.lr,args.dice_weight,args.noise_std,args.background_k,args.inclusion_k]).all()
            or args.lr<=0 or args.dice_weight<0 or args.noise_std<0 or args.background_k<=0
            or args.inclusion_k<=0 or args.background_k==args.inclusion_k):
        parser.error('Invalid training parameters; --limit must be >= 10')
    device=select_device(args.device)
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error('Output directory is not empty; choose a new run or use --resume last.pt')
    print(f'Device: {device}'+(f' ({torch.cuda.get_device_name(device)})' if device.type=='cuda' else ''),flush=True)
    records,cached,signature=index_dataset(args.data,args.background_k,args.inclusion_k,args.limit,args.cache)
    partitions=split_records(records,args.seed)
    manifest=dict(method='geometry-grouped random 70/15/15 split',seed=args.seed,
                  signature=signature,records=records,partitions=partitions,
                  note='All copies of an identical interior stay in one partition. This is not a spatial holdout.')
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(args).items()}
    checkpoint=None
    if args.resume is not None:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
        if checkpoint['manifest']!=manifest:raise ValueError('Checkpoint dataset/split differs from current data')
        for key in ('seed','batch_size','lr','base_channels','dice_weight','noise_std','background_k','inclusion_k','patience'):
            if checkpoint['config'][key]!=config[key]:raise ValueError(f'Resume requires the same --{key.replace("_","-")}')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists() and json.loads((args.output/'manifest.json').read_text())!=manifest:
        raise ValueError('Existing output contains a different dataset manifest')
    atomic_json(args.output/'manifest.json',manifest);atomic_json(args.output/'config.json',config)
    for split,names in partitions.items():
        (args.output/f'{split}.txt').write_text('\n'.join(names)+'\n')
    print('Split specimens: '+', '.join(f'{k}={len(v)}' for k,v in partitions.items())+
          f'; unique interiors={len({r["geometry_sha256"] for r in records})}',flush=True)
    placeholder=dict(mean_K=0.,std_K=1.)
    datasets={split:ThermalDataset(args.data,names,cached,args.background_k,args.inclusion_k,placeholder,
                                   args.noise_std,args.seed,training=split=='train') for split,names in partitions.items()}
    normalization,pos_weight=training_statistics(datasets['train'])
    if checkpoint is not None and checkpoint['normalization']!=normalization:raise ValueError('Training normalization changed')
    for dataset in datasets.values():dataset.normalization=normalization
    atomic_json(args.output/'normalization.json',dict(**normalization,pos_weight=pos_weight))
    print(f'Training-only normalization: {normalization}; BCE positive weight={pos_weight:.2f}',flush=True)
    generator=torch.Generator()
    loaders={split:DataLoader(dataset,batch_size=args.batch_size,shuffle=split=='train',
                              num_workers=args.num_workers,pin_memory=device.type=='cuda',
                              generator=generator if split=='train' else None)
             for split,dataset in datasets.items()}
    model_config=dict(in_channels=signature['video_shape'][0],out_channels=signature['target_shape'][0],base_channels=args.base_channels)
    model=ThermoNet(**model_config).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    history=[];start=1;best=float('inf');bad_epochs=0
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history=checkpoint['history'];start=checkpoint['epoch']+1
        best=checkpoint['best_validation_loss'];bad_epochs=checkpoint['bad_epochs']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type=='cuda' and checkpoint.get('cuda_rng_states'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
        if not (args.output/'best.pt').exists():
            raise ValueError('Resume output must contain best.pt from the same run; copy it alongside last.pt')
    spacing_mm=signature['cell_size']*1000
    for epoch in range(start,args.epochs+1):
        began=perf_counter();datasets['train'].epoch=epoch;generator.manual_seed(args.seed+epoch)
        train,_=run_epoch(model,loaders['train'],device,bce,args,spacing_mm,optimizer)
        validation,_=run_epoch(model,loaders['validation'],device,bce,args,spacing_mm)
        improved=validation['loss']<best
        if improved:best=validation['loss'];bad_epochs=0
        else:bad_epochs+=1
        history.append(dict(epoch=epoch,seconds=perf_counter()-began,train=train,validation=validation))
        state=dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                   model_config=model_config,config=config,manifest=manifest,normalization=normalization,
                   pos_weight=pos_weight,history=history,best_validation_loss=best,bad_epochs=bad_epochs,
                   torch_rng_state=torch.get_rng_state(),
                   cuda_rng_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        if improved:atomic_checkpoint(args.output/'best.pt',state)
        atomic_checkpoint(args.output/'last.pt',state);atomic_json(args.output/'history.json',history)
        print(f"Epoch {epoch:03d}/{args.epochs}: train loss={train['loss']:.4f}; "
              f"val loss={validation['loss']:.4f}, Dice={validation['dice']:.4f}, IoU={validation['iou']:.4f}; "
              f"{history[-1]['seconds']:.1f} s"+(' [best]' if improved else ''),flush=True)
        if args.patience and bad_epochs>=args.patience:
            print('Early stopping on validation loss.',flush=True);break
    best_checkpoint=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    test,rows=run_epoch(model,loaders['test'],device,bce,args,spacing_mm)
    per_depth={}
    for depth in sorted({r['true_front_depth_mm'] for r in rows}):
        group=[r for r in rows if r['true_front_depth_mm']==depth]
        per_depth[str(round(depth,6))]=dict(samples=len(group),dice=float(np.mean([r['dice'] for r in group])),
                                          iou=float(np.mean([r['iou'] for r in group])),
                                          predicted_nonempty=float(np.mean([r['predicted_nonempty'] for r in group])))
    results=dict(checkpoint='best.pt',best_epoch=best_checkpoint['epoch'],test=test,
                 by_true_front_depth_mm=per_depth,
                 depth_metric_note='Centroid depth MAE is conditional on a nonempty thresholded prediction; '
                                   'predicted_nonempty reports the fraction evaluated. Dice/IoU include empty predictions.')
    atomic_json(args.output/'test_metrics.json',results)
    with (args.output/'test_per_sample.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('Final test (best validation checkpoint): '+json.dumps(test,allow_nan=False),flush=True)
    print(f'Artifacts: {args.output.resolve()}',flush=True)


if __name__=='__main__':main()
