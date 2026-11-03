"""Train a fresh ThermoNet on all three cohorts; keep the single-cube trainer intact.

python train_thermonet_all.py --data datasets --output runs/all_cohorts --cache
Use --single-checkpoint best_single.pt to preserve the original single-cube split.
"""
import argparse
import csv
import hashlib
from io import BytesIO
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch.utils.data import DataLoader

from thermonet import ThermoNet
from train_thermonet import (ThermalDataset,atomic_json,atomic_checkpoint,
                            training_statistics,loss_and_metrics,select_device)

COHORTS=('single_cube','two_cubes','mixed_cubes')


def load_multi(path, background, inclusion):
    # One filesystem read avoids repeated seeks against mounted/cloud storage.
    with np.load(BytesIO(path.read_bytes()),allow_pickle=False) as d:
        video=np.array(d['video'],dtype=np.float32);field=np.asarray(d['conductivity'])
        times=np.asarray(d['frame_times'],dtype=float);h=float(d['cell_size'])
        config=json.loads(d['config_json'].item()) if 'config_json' in d else {}
    if (video.ndim!=3 or field.ndim!=3 or min(video.shape)<1 or min(field.shape)<1
            or min(video.shape[1:])<4 or video.shape[1:]!=field.shape[1:]
            or times.shape!=(len(video),) or np.any(np.diff(times)<=0)
            or not all(np.isfinite(a).all() for a in (video,field,times))
            or not np.isfinite(h) or h<=0):raise ValueError(f'{path}: invalid arrays/metadata')
    target=np.isclose(field,inclusion,rtol=1e-5,atol=1e-8)
    if not np.all(target|np.isclose(field,background,rtol=1e-5,atol=1e-8)):
        raise ValueError(f'{path}: different material conductivities')
    labels,n=ndimage.label(target)
    sizes=[]
    for bounds in ndimage.find_objects(labels):
        if bounds is None:raise ValueError(f'{path}: incomplete connected component')
        sides=tuple(s.stop-s.start for s in bounds)
        block=target[bounds]
        if len(set(sides))!=1 or not block.all():raise ValueError(f'{path}: expected separated solid cubes')
        sizes.append(sides[0])
    if not sizes or len(set(sizes))!=1:raise ValueError(f'{path}: missing or inconsistent cube sizes')
    if 'n_cubes' in config and config['n_cubes']!=n:raise ValueError(f'{path}: cube count disagrees with metadata')
    physical={k:config[k] for k in ('dt','heat_flux','heat_capacity','pulse_seconds','k_background','k_cube') if k in config}
    signature=dict(video_shape=list(video.shape),target_shape=list(field.shape),cell_size=h,
                   frame_times=times.tolist(),cube_cells=sizes[0],physical=physical)
    return video,target.astype(np.uint8),signature,int(n)


def index_all(root, background, inclusion, limit, cache):
    records=[];cached={};signature=None
    for cohort in COHORTS:
        paths=sorted((root/cohort).glob('sample_*.npz'))
        if limit is not None:paths=paths[:limit]
        if not paths:raise ValueError(f'No sample files in {root/cohort}')
        for i,path in enumerate(paths):
            video,target,current,n=load_multi(path,background,inclusion)
            if (cohort=='single_cube' and n!=1) or (cohort=='two_cubes' and n!=2) or (cohort=='mixed_cubes' and not 3<=n<=10):
                raise ValueError(f'{path}: wrong inclusion count for {cohort}')
            if signature is None:signature=current
            if current!=signature:raise ValueError(f'{path}: inconsistent grid, cube size, materials or acquisition')
            name=path.relative_to(root).as_posix()
            records.append(dict(sample=name,cohort=cohort,true_cubes=n,
                                geometry_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                                content_sha256=hashlib.sha256(video.astype('<f4').tobytes()+target.tobytes()).hexdigest(),
                                positive_voxels=int(target.sum())))
            if cache:cached[name]=(video,target)
            if (i+1)%100==0 or i+1==len(paths):print(f'Indexed {cohort}: {i+1}/{len(paths)}',flush=True)
    return records,cached,signature


def make_partitions(records, seed, reference=None, signature=None):
    """Group exact duplicate fields globally; stratify unique fields by cube count."""
    grouped={}
    for record in records:grouped.setdefault(record['geometry_sha256'],[]).append(record)
    for group in grouped.values():
        if len({(r['cohort'],r['true_cubes']) for r in group})!=1:
            raise ValueError('Identical geometry has inconsistent cohort/count labels')
    partitions={s:[] for s in ('train','validation','test')}
    pinned={}
    if reference is not None:
        old=reference['manifest']
        if old['signature']!=signature:raise ValueError('Original single-cube checkpoint uses a different physical model/grid')
        old_records={r['sample']:r for r in old['records']}
        singles={r['sample'].split('/',1)[1]:r for r in records if r['cohort']=='single_cube'}
        if set(singles)!=set(old_records):raise ValueError('To preserve the original split, supply its complete single-cube dataset')
        for name,r in singles.items():
            if r['content_sha256']!=old_records[name]['content_sha256']:
                raise ValueError(f'{name}: single-cube data changed since the original experiment')
        assigned=set()
        for split,names in old['partitions'].items():
            for name in names:
                record=singles[name];geometry=record['geometry_sha256']
                if name in assigned:raise ValueError('Repeated specimen in original split')
                if geometry in pinned and pinned[geometry]!=split:raise ValueError('Original split leaks duplicate interiors')
                assigned.add(name);pinned[geometry]=split
        if assigned!=set(singles):raise ValueError('Original split does not cover all single-cube data')
    strata={}
    for geometry,group in grouped.items():
        if geometry in pinned:
            partitions[pinned[geometry]].extend(r['sample'] for r in group)
        else:strata.setdefault(group[0]['true_cubes'],[]).append(geometry)
    for count,keys in sorted(strata.items()):
        keys=sorted(keys)
        if len(keys)<3:raise ValueError(f'Need at least 3 unique geometries with {count} cubes to populate every partition')
        np.random.default_rng(np.random.SeedSequence([seed,count])).shuffle(keys)
        ntrain=min(len(keys)-2,max(1,int(.70*len(keys))));nval=max(1,int(.15*len(keys)))
        nval=min(nval,len(keys)-ntrain-1)
        for split,chosen in [('train',keys[:ntrain]),('validation',keys[ntrain:ntrain+nval]),('test',keys[ntrain+nval:])]:
            for geometry in chosen:partitions[split].extend(r['sample'] for r in grouped[geometry])
    for names in partitions.values():names.sort()
    assigned=[name for names in partitions.values() for name in names]
    assert len(assigned)==len(set(assigned))==len(records)
    lookup={r['sample']:r for r in records}
    hashes={s:{lookup[n]['geometry_sha256'] for n in names} for s,names in partitions.items()}
    assert not(hashes['train']&hashes['validation'] or hashes['train']&hashes['test'] or hashes['validation']&hashes['test'])
    return partitions


class AllCohortDataset(ThermalDataset):
    def raw_pair(self,name):
        if name in self.cached:return self.cached[name]
        video,target,_,_=load_multi(self.folder/name,self.background,self.inclusion)
        return video,target


def component_metrics(truth,probability):
    pred=probability>=.5;gl,ng=ndimage.label(truth);pl,npred=ndimage.label(pred)
    pairs=np.bincount((gl*(npred+1)+pl).ravel(),minlength=(ng+1)*(npred+1)).reshape(ng+1,npred+1)[1:,1:]
    a=np.bincount(gl.ravel(),minlength=ng+1)[1:];b=np.bincount(pl.ravel(),minlength=npred+1)[1:]
    overlaps=pairs/np.maximum(a[:,None]+b[None,:]-pairs,1);matched=0
    if npred:
        i,j=linear_sum_assignment((overlaps>=.5)*(ng+1)+overlaps,maximize=True)
        matched=int((overlaps[i,j]>=.5).sum())
    return dict(predicted_components=int(npred),matched_objects=matched)


def summarize(rows):
    summary=dict(samples=len(rows))
    for key in ('dice','iou','precision','recall','conductivity_mse','predicted_nonempty'):
        summary[key]=float(np.mean([r[key] for r in rows]))
    if 'matched_objects' in rows[0]:
        matched=sum(r['matched_objects'] for r in rows);true=sum(r['true_cubes'] for r in rows)
        predicted=sum(r['predicted_components'] for r in rows)
        summary.update(true_objects=true,predicted_components=predicted,matched_objects=matched,
                       object_recall=matched/true,object_precision=matched/max(1,predicted))
    return summary


def epoch_pass(model,loader,device,bce,args,spacing,lookup,optimizer=None,objects=False):
    model.train(optimizer is not None);total=0.;count=0;rows=[]
    for x,y,names in loader:
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        if optimizer is not None:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits=model(x)
            loss,metrics=loss_and_metrics(logits,y,bce,args.dice_weight,args.inclusion_k-args.background_k,spacing)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            if optimizer is not None:loss.backward();optimizer.step()
        total+=float(loss.detach())*len(names);count+=len(names)
        if objects:
            predictions=torch.sigmoid(logits.detach()).cpu().numpy();targets=y.detach().cpu().numpy().astype(bool)
        for i,name in enumerate(names):
            row=dict(sample=name,cohort=lookup[name]['cohort'],true_cubes=lookup[name]['true_cubes'])
            # Whole-volume centroid depth is misleading for several inclusions.
            for key in ('dice','iou','precision','recall','conductivity_mse','predicted_nonempty'):
                row[key]=float(metrics[key][i])
            if objects:row.update(component_metrics(targets[i],predictions[i]))
            rows.append(row)
    return dict(loss=total/count,**summarize(rows)),rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('datasets'))
    parser.add_argument('--output',type=Path,default=Path('runs/all_cohorts'))
    parser.add_argument('--single-checkpoint',type=Path,help='Preserve the old single-cube split; weights are not loaded')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--base-channels',type=int,default=32)
    parser.add_argument('--patience',type=int,default=20)
    parser.add_argument('--dice-weight',type=float,default=1.)
    parser.add_argument('--noise-std',type=float,default=0.)
    parser.add_argument('--background-k',type=float,default=.20)
    parser.add_argument('--inclusion-k',type=float,default=.50)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=('auto','cpu','cuda','mps'),default='auto')
    parser.add_argument('--num-workers',type=int,default=0)
    parser.add_argument('--cpu-threads',type=int,default=2)
    parser.add_argument('--cache',action='store_true')
    parser.add_argument('--limit-per-cohort',type=int,help='Smoke-test subset; must cover >=3 unique geometries per count')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if (min(args.epochs,args.batch_size,args.base_channels,args.cpu_threads)<1 or args.patience<0
            or args.num_workers<0 or args.seed<0 or (args.limit_per_cohort is not None and args.limit_per_cohort<1)
            or not np.isfinite([args.lr,args.dice_weight,args.noise_std,args.background_k,args.inclusion_k]).all()
            or args.lr<=0 or args.dice_weight<0 or args.noise_std<0 or min(args.background_k,args.inclusion_k)<=0
            or args.background_k==args.inclusion_k):parser.error('Invalid training parameters')
    if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
        parser.error('Use a new output directory or --resume last.pt')
    device=select_device(args.device);torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(args.seed)
    print('Device:',device,flush=True)
    records,cached,signature=index_all(args.data,args.background_k,args.inclusion_k,args.limit_per_cohort,args.cache)
    reference=None;reference_hash=None
    if args.single_checkpoint:
        raw=args.single_checkpoint.read_bytes();reference_hash=hashlib.sha256(raw).hexdigest()
        reference=torch.load(BytesIO(raw),map_location='cpu',weights_only=True)
    partitions=make_partitions(records,args.seed,reference,signature)
    distribution={}
    lookup={r['sample']:r for r in records}
    for split,names in partitions.items():
        distribution[split]={str(n):sum(lookup[name]['true_cubes']==n for name in names) for n in range(1,11)}
    manifest=dict(method='geometry-grouped random split, stratified by cube count',seed=args.seed,
                  signature=signature,records=records,partitions=partitions,cube_count_distribution=distribution,
                  original_single_checkpoint_sha256=reference_hash,original_single_split_preserved=reference is not None,
                  note='Relative paths include cohort names. Exact duplicate fields never cross splits. '
                       'Single-cube weights are never imported; optional checkpoint supplies only split metadata.')
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    checkpoint=None
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=True)
        if checkpoint['manifest']!=manifest:raise ValueError('Resume dataset or partition changed')
        for key in ('seed','batch_size','lr','base_channels','dice_weight','noise_std','background_k','inclusion_k','patience'):
            if checkpoint['config'][key]!=config[key]:raise ValueError(f'Resume requires the same {key}')
        if not (args.output/'best.pt').exists():raise ValueError('Resume requires best.pt beside the current output last.pt')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'manifest.json').exists() and json.loads((args.output/'manifest.json').read_text())!=manifest:
        raise ValueError('Output contains a different manifest')
    atomic_json(args.output/'manifest.json',manifest);atomic_json(args.output/'config.json',config)
    for split,names in partitions.items():(args.output/f'{split}.txt').write_text('\n'.join(names)+'\n')
    print('Split:',{k:len(v) for k,v in partitions.items()},flush=True)
    print('Cube-count distribution:',distribution,flush=True)
    datasets={s:AllCohortDataset(args.data,names,cached,args.background_k,args.inclusion_k,
                                 dict(mean_K=0.,std_K=1.),args.noise_std,args.seed,training=s=='train') for s,names in partitions.items()}
    normalization,pos_weight=training_statistics(datasets['train'])
    if checkpoint and checkpoint['normalization']!=normalization:raise ValueError('Training normalization changed')
    for dataset in datasets.values():dataset.normalization=normalization
    atomic_json(args.output/'normalization.json',dict(**normalization,pos_weight=pos_weight))
    generator=torch.Generator()
    loaders={s:DataLoader(d,batch_size=args.batch_size,shuffle=s=='train',num_workers=args.num_workers,
                          pin_memory=device.type=='cuda',generator=generator if s=='train' else None) for s,d in datasets.items()}
    model_config=dict(in_channels=signature['video_shape'][0],out_channels=signature['target_shape'][0],base_channels=args.base_channels)
    model=ThermoNet(**model_config).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    history=[];start=1;best=float('inf');bad=0
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        history=checkpoint['history'];start=checkpoint['epoch']+1;best=checkpoint['best_validation_loss'];bad=checkpoint['bad_epochs']
        torch.set_rng_state(checkpoint['torch_rng_state'])
        if device.type=='cuda' and checkpoint['cuda_rng_states']:torch.cuda.set_rng_state_all(checkpoint['cuda_rng_states'])
    for epoch in range(start,args.epochs+1):
        began=perf_counter();datasets['train'].epoch=epoch;generator.manual_seed(args.seed+epoch)
        train,_=epoch_pass(model,loaders['train'],device,bce,args,signature['cell_size']*1000,lookup,optimizer)
        val,_=epoch_pass(model,loaders['validation'],device,bce,args,signature['cell_size']*1000,lookup)
        improved=val['loss']<best
        if improved:best=val['loss'];bad=0
        else:bad+=1
        history.append(dict(epoch=epoch,seconds=perf_counter()-began,train=train,validation=val))
        state=dict(epoch=epoch,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                   model_config=model_config,config=config,manifest=manifest,normalization=normalization,pos_weight=pos_weight,
                   history=history,best_validation_loss=best,bad_epochs=bad,torch_rng_state=torch.get_rng_state(),
                   cuda_rng_states=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
        if improved:atomic_checkpoint(args.output/'best.pt',state)
        atomic_checkpoint(args.output/'last.pt',state);atomic_json(args.output/'history.json',history)
        print(f"Epoch {epoch:03d}/{args.epochs}: train={train['loss']:.4f}; val={val['loss']:.4f}, "
              f"Dice={val['dice']:.4f}, IoU={val['iou']:.4f}; {history[-1]['seconds']:.1f}s"+(' [best]' if improved else ''),flush=True)
        if args.patience and bad>=args.patience:print('Early stopping.',flush=True);break
    best_state=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(best_state['model_state_dict'])
    test,rows=epoch_pass(model,loaders['test'],device,bce,args,signature['cell_size']*1000,lookup,objects=True)
    results=dict(checkpoint='best.pt',best_epoch=best_state['epoch'],test=test,
                 by_cohort={s:summarize([r for r in rows if r['cohort']==s]) for s in COHORTS},
                 by_true_cube_count={str(n):summarize([r for r in rows if r['true_cubes']==n]) for n in sorted({r['true_cubes'] for r in rows})},
                 original_single_split_preserved=reference is not None,
                 metric_note='Voxel metrics are equal-weight specimen means; object precision/recall pool counts. '
                             'Predicted objects are 6-connected components at probability >=0.5, matched one-to-one at IoU >=0.5. '
                             'No whole-volume centroid depth error is reported for multiple inclusions.')
    atomic_json(args.output/'test_metrics.json',results)
    with (args.output/'test_per_sample.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('Final test (best validation checkpoint):',json.dumps(test),flush=True)
    print('Saved:',args.output.resolve(),flush=True)


if __name__=='__main__':main()
