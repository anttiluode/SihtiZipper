#!/usr/bin/env python3
"""Experiment 002: a genuinely trained conditional pixel DDPM versus explicit tracking.

All methods receive the SAME causal occluded observations. Actor colors are persistent
appearance cues, not identity labels passed through a hidden oracle. The test's
latent coordinates are used only when scoring. The DDPM is a pixel reconstruction
model; an identical, explicit readout determines whether an inferred actor is
present. A zipper makes explicit link/abstain decisions; no future lookahead.
"""
from __future__ import annotations
import argparse, json, math, pathlib, random, time
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT=pathlib.Path(__file__).resolve().parent
H=W=24; T=17; N=18; BANDS=(9,15)
torch.set_num_threads(min(4,torch.get_num_threads()))


def gen(seed, mode='mixed'):
    r=np.random.default_rng(seed)
    v=float(r.uniform(1.03,1.35)); y0=float(r.uniform(8,11));y1=float(r.uniform(13,16))
    vy0=float(r.uniform(-.14,.14));vy1=float(r.uniform(-.14,.14))
    # A reversal occurs only once the identities become hidden by the opaque screen.
    reversal=(mode=='reverse') or (mode=='mixed' and r.random()<.36)
    reverse_at=int(r.integers(7,10))
    X=np.zeros((N,2,H,W),np.float32)
    truth=np.zeros((N,2,2),np.float32)
    yy,xx=np.mgrid[:H,:W];
    for t in range(N):
        travel=t if not reversal else (t if t<=reverse_at else 2*reverse_at-t)
        c0=(2.5+v*travel,y0+vy0*t)
        c1=(21.5-v*travel,y1+vy1*t)
        for i,(cx,cy) in enumerate([c0,c1]):
            truth[t,i]=[cx,cy]
            X[t,i]=np.exp(-.5*((xx-cx)**2+(yy-cy)**2)/(1.32**2))
    mask=np.ones((1,H,W),np.float32);mask[:,:,BANDS[0]:BANDS[1]]=0
    # Modest sensor noise, zeroed by an actual opaque occluder.
    obs=np.clip(X+r.normal(0,.016,X.shape).astype('float32'),0,1)*mask
    # Independent camera ghosts: a false bright same-appearance patch can appear
    # outside the opaque screen while the true actor remains hidden behind it.
    # They are observations, not a shifted model read or hidden truth.
    ghosts=[]
    for t in range(4,N):
        for k in range(2):
            cx=truth[t,k,0]
            if BANDS[0]+.5<cx<BANDS[1]-.5 and r.random()<.18:
                gx=float(r.choice([3.5,20.2])+r.uniform(-.9,.9))
                gy=float(r.uniform(6,18))
                obs[t,k]=np.clip(obs[t,k]+.85*np.exp(-.5*((xx-gx)**2+(yy-gy)**2)/(1.15**2))*mask[0],0,1)
                ghosts.append([t,k,round(gx,2),round(gy,2)])
    return {'clean':X,'obs':obs,'mask':mask,'truth':truth,'reverse':bool(reversal),'reverse_at':reverse_at,'seed':seed,'ghosts':ghosts}


def condition(scene,t):
    z=scene['obs'];o0=z[t];o1=z[max(0,t-1)];o2=z[max(0,t-2)];m=scene['mask'];
    return np.concatenate([o0,o1,o2,m],axis=0)


def collect(seeds):
    cond=[];clean=[];metadata=[]
    for s in seeds:
        scene=gen(s)
        for t in range(2,N):
            cond.append(condition(scene,t));clean.append(scene['clean'][t]);metadata.append((s,t))
    return np.stack(cond).astype('float32'),np.stack(clean).astype('float32'),metadata

class ConvPair(nn.Module):
    def __init__(self,inc,outc):
        super().__init__();self.a=nn.Conv2d(inc,outc,3,padding=1);self.b=nn.Conv2d(outc,outc,3,padding=1)
        self.g=nn.GroupNorm(4,outc)
    def forward(self,x):return F.silu(self.b(F.silu(self.g(self.a(x)))))

class TinyDDPM(nn.Module):
    def __init__(self,width=24):
        super().__init__();
        self.stem=ConvPair(2+7+1,width)
        self.down=nn.Conv2d(width,2*width,3,stride=2,padding=1)
        self.mid=ConvPair(2*width,2*width)
        self.up=ConvPair(width*3,width)
        self.out=nn.Conv2d(width,2,3,padding=1)
    def forward(self,x,condition,step):
        # x: noisy target pixels, condition: THREE observed frames + mask.
        t=step.view(-1,1,1,1).expand(-1,1,x.shape[2],x.shape[3])
        z=self.stem(torch.cat((x,condition,t),1))
        u=self.mid(F.silu(self.down(z)))
        u=F.interpolate(u,scale_factor=2,mode='nearest')
        return self.out(self.up(torch.cat((u,z),1)))

class Diffuser:
    def __init__(self,model,device='cpu',n_steps=18):
        self.model=model.to(device);self.device=device;self.n_steps=n_steps
        self.beta=torch.linspace(.04,.36,n_steps,device=device)
        self.alpha=1-self.beta;self.ab=torch.cumprod(self.alpha,0)
    def train(self,seeds,steps=300,batch=48,seed=401):
        torch.manual_seed(seed);np.random.seed(seed)
        x,y,_=collect(seeds)
        C=torch.from_numpy(x).to(self.device)
        # Network sees -1/+1 noisy clean image in DDPM and 0/1 observed history.
        Y=torch.from_numpy(y).to(self.device)*2-1
        optim=torch.optim.Adam(self.model.parameters(),lr=.0015)
        self.model.train();hist=[];tm=time.monotonic()
        for i in range(steps):
            ids=torch.randint(len(C),(batch,),device=self.device)
            timesteps=torch.randint(self.n_steps,(batch,),device=self.device)
            x0=Y[ids];eps=torch.randn_like(x0)
            ab=self.ab[timesteps].view(-1,1,1,1)
            xt=torch.sqrt(ab)*x0+torch.sqrt(1-ab)*eps
            pred=self.model(xt,C[ids],timesteps.float()/(self.n_steps-1))
            # Conventional noise-prediction loss, with extra foreground emphasis:
            # rare objects should not be dominated by black background pixels.
            weight=1+2.5*(x0>-.6).float()
            loss=((pred-eps).square()*weight).mean()
            optim.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),2.0);optim.step()
            if i%50==0 or i==steps-1:hist.append([i,round(loss.item(),5)])
        return {'train_steps':steps,'train_examples':len(x),'loss_trace':hist,'training_seconds':round(time.monotonic()-tm,2)}
    @torch.no_grad()
    def sample(self,conds,seed=1984,batch=64):
        torch.manual_seed(seed);self.model.eval();out=[]
        for start in range(0,len(conds),batch):
            c=torch.from_numpy(conds[start:start+batch]).to(self.device)
            x=torch.randn((len(c),2,H,W),device=self.device)
            for t in reversed(range(self.n_steps)):
                ts=torch.full((len(c),),t/(self.n_steps-1),device=self.device)
                eps=self.model(x,c,ts)
                x0=((x-torch.sqrt(1-self.ab[t])*eps)/torch.sqrt(self.ab[t])).clamp(-1,1)
                if t>0:
                    abprev=self.ab[t-1]
                    mean=(torch.sqrt(abprev)*self.beta[t]/(1-self.ab[t])*x0+
                          torch.sqrt(self.alpha[t])*(1-abprev)/(1-self.ab[t])*x)
                    var=self.beta[t]*(1-abprev)/(1-self.ab[t])
                    x=mean+torch.sqrt(var)*torch.randn_like(x)
                else:x=x0
            denoised=((x+1)/2).clamp(0,1)
            # For inpainting, observed pixels are evidence and are never repainted.
            # This exact conditioning is also used by the non-generative tracker.
            denoised=denoised*(1-c[:,6:7])+c[:,0:2]*c[:,6:7]
            out.append(denoised.cpu().numpy())
        return np.concatenate(out)


def detection(image,mask=None,min_peak=.12):
    """Observed/predicted 2D color responses to centroids; no truth access."""
    vals=[];yy,xx=np.mgrid[:H,:W]
    for ch in range(2):
        a=np.asarray(image[ch],dtype=float)
        if mask is not None:a=a*mask
        # Foreground-weighted soft readout resists tiny diffusion speckles.
        a=np.maximum(0,a-.24)
        total=float(a.sum())
        peak=float(np.max(image[ch] if mask is None else image[ch]*mask))
        if total<.65 or peak<min_peak:vals.append(None);continue
        vals.append((float((a*xx).sum()/total),float((a*yy).sum()/total),peak))
    return vals

class ZipperTracks:
    """Explicit object-address seams: observed colors remain separate and immutable.
    Positions are predicted through occlusion, never reported as observed links.
    A returned observation can relock, slip or be rejected as a false association.
    """
    def __init__(self,gating=3.1):
        self.previous=[None,None];self.velocity=[(0.,0.),(0.,0.)];self.age=[0,0]
        self.gating=gating;self.jams=0;self.relocks=0;self.slips=0;self.archive=[]
        self.original_tape=[];self.link_masks=[]
    def unzip(self,frame,channel):
        if not (0<=frame<len(self.link_masks) and channel in (0,1)):
            raise IndexError('Seam index out of bounds')
        self.link_masks[frame][channel]=False
    def rezip(self,frame,channel):
        if not (0<=frame<len(self.link_masks) and channel in (0,1)):
            raise IndexError('Seam index out of bounds')
        action=self.archive[frame]['actions'][channel]
        self.link_masks[frame][channel]=action in ('ACQUIRE','LOCK','RELOCK')
    def step(self,image):
        # The original input tape is immutable data; undoing a link cannot
        # change the sensor observations or erase earlier alternatives.
        self.original_tape.append(np.array(image,copy=True))
        obs=detection(image)
        decisions=[];pos=[]
        for k,hit in enumerate(obs):
            prev=self.previous[k];vx,vy=self.velocity[k]
            if prev is None:
                if hit is None:
                    pos.append(None);decisions.append('UNKNOWN');continue
                self.previous[k]=hit[:2];self.age[k]=0
                pos.append(hit[:2]);decisions.append('ACQUIRE');continue
            guess=(prev[0]+vx,prev[1]+vy)
            if hit is None:
                self.previous[k]=guess;self.age[k]+=1
                pos.append(guess);decisions.append('PREDICT_ONLY');continue
            dist=math.dist(guess,hit[:2]);
            if dist>self.gating+min(5,self.age[k])*.7:
                self.jams+=1;self.previous[k]=guess;self.age[k]+=1
                pos.append(guess);decisions.append('JAM_ABSTAIN');continue
            newv=(hit[0]-prev[0],hit[1]-prev[1]);
            if self.age[k]>0:
                self.relocks+=1;self.velocity[k]=(0.65*vx+.35*newv[0],.65*vy+.35*newv[1])
                decisions.append('RELOCK')
            else:
                self.velocity[k]=(.35*vx+.65*newv[0],.35*vy+.65*newv[1]);decisions.append('LOCK')
            self.previous[k]=hit[:2];self.age[k]=0;pos.append(hit[:2])
        self.link_masks.append([a in ('ACQUIRE','LOCK','RELOCK') for a in decisions])
        self.archive.append({'observed':[(None if h is None else list(h[:2])) for h in obs],
                             'position':pos,'actions':decisions})
        return pos,decisions


def nearest_motion_baseline(scene):
    # Strictly constant velocity and color-conditioned; no learned filters.
    tr=ZipperTracks(gating=1e9)
    return [tr.step(obs) for obs in scene['obs']]


def tracker_render(positions,obs,mask):
    a=np.zeros((2,H,W),dtype=np.float32);yy,xx=np.mgrid[:H,:W]
    for k,p in enumerate(positions):
        if p is not None:
            a[k]=np.exp(-.5*((xx-p[0])**2+(yy-p[1])**2)/(1.32**2))
    return np.clip(a*(1-mask)+obs*mask,0,1)


def assess(scenes,diffused,metadata,make_examples=False):
    lookup={key:i for i,key in enumerate(metadata)}
    totals={name:{'pixels_sse':0.,'pixels_n':0,'foreground_sse':0.,'foreground_n':0,
                  'hidden_centers_error':[],'hidden_abstain':0,'hidden_decisions':0,
                  'observed_wrong':0,'observed_links':0,'observed_abstain':0}
            for name in ['hold_last','constant_velocity','zipper','diffusion']}
    examples=[];sequences=[]
    for scene in scenes:
        tr=ZipperTracks();baseline=ZipperTracks(gating=1e9)
        case={'seed':scene['seed'],'reversal':scene['reverse'],'frames':[]}
        for t in range(N):
            obs=scene['obs'][t];mask=scene['mask'];true=scene['clean'][t]
            gt=scene['truth'][t]
            p,act=tr.step(obs)
            pb,_=baseline.step(obs)
            idx=lookup.get((scene['seed'],t))
            if idx is None:continue
            candidates={
                'hold_last':scene['obs'][t-1],
                'constant_velocity':tracker_render(pb,obs,mask),
                'zipper':tracker_render(p,obs,mask),
                'diffusion':diffused[idx],
            }
            active=[]
            for k in range(2):
                x,y=gt[k];hidden=BANDS[0]<=x<BANDS[1];active.append(hidden)
            frame={'t':t,'actions':act,'truth':gt.round(2).tolist(),'observed_centers':tr.archive[-1]['observed']}
            for name,pred in candidates.items():
                stats=totals[name];area=(1-mask)>0
                delta=np.square(pred-true);stats['pixels_sse']+=float(np.sum(delta*area));stats['pixels_n']+=int(np.sum(area)*2)
                target=(true>.2)&area;stats['foreground_sse']+=float(np.sum(delta*target));stats['foreground_n']+=int(np.sum(target))
                # Position estimates *inside* the unseen strip are hypotheses,
                # not claims that the hidden object was actually observed.
                predcenters=detection(pred,mask=(1-mask)[0])
                if name in ('zipper','constant_velocity'):
                    pp=p if name=='zipper' else pb
                    predcenters=[(*pp[k],1) if pp[k] is not None else None for k in range(2)]
                for k in range(2):
                    if active[k]:
                        stats['hidden_decisions']+=1
                        if predcenters[k] is None:stats['hidden_abstain']+=1
                        else:stats['hidden_centers_error'].append(math.dist(predcenters[k][:2],gt[k]))
                # Observed association is scored ONLY for explicit trackers;
                # a generated diffusion frame has no built-in identity-link
                # assertion. Scoring imagined pixels as observed identity
                # would conflate generation with tracking.
                if name in ('zipper','constant_velocity'):
                    observed_decisions=act if name=='zipper' else baseline.archive[-1]['actions']
                    observed_positions=tr.archive[-1]['observed'] if name=='zipper' else baseline.archive[-1]['observed']
                    for k in range(2):
                        action=observed_decisions[k]
                        if action not in ('ACQUIRE','LOCK','RELOCK') or observed_positions[k] is None:
                            stats['observed_abstain']+=1;continue
                        stats['observed_links']+=1
                        # Matching a phantom outside the masked region is wrong
                        # even if the ghost looks like the target's channel.
                        if math.dist(observed_positions[k],gt[k])>2.5:
                            stats['observed_wrong']+=1
            frame['zipper_prediction']=candidates['zipper'].astype('float16').tolist() if make_examples and scene is scenes[0] and t in (3,6,8,11,14) else None
            if scene is scenes[0] and t in (3,6,8,11,14):
                examples.append({'t':t,'truth':gt.round(2).tolist(),'actions':act,'clean':true.round(3).tolist(),'observed':obs.round(3).tolist(),'diffusion':candidates['diffusion'].round(3).tolist(),'zipper':candidates['zipper'].round(3).tolist()})
            case['frames'].append({'t':t,'actions':act,'gt':gt.round(2).tolist(),
                                   'estimated':[[round(x,2),round(y,2)] if p0 is not None else None for p0 in p for x,y in [p0]]})
        sequences.append(case)
    result={}
    for name,r in totals.items():
        result[name]={'hidden_region_mse':round(r['pixels_sse']/max(1,r['pixels_n']),6),
                      'hidden_foreground_mse':round(r['foreground_sse']/max(1,r['foreground_n']),6),
                      'hidden_center_forecast_mae_px':round(float(np.mean(r['hidden_centers_error'])) if r['hidden_centers_error'] else float('nan'),3),
                      'hidden_center_estimates':len(r['hidden_centers_error']),
                      'hidden_center_unknown':r['hidden_abstain'],
                      'false_observation_joins':r['observed_wrong'] if name in ('zipper','constant_velocity') else None,
                      'observation_joins':r['observed_links'] if name in ('zipper','constant_velocity') else None,
                      'observation_abstentions':r['observed_abstain'] if name in ('zipper','constant_velocity') else None}
    return result,examples,sequences


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=480);parser.add_argument('--train-scenes',type=int,default=320);parser.add_argument('--test-scenes',type=int,default=24)
    parser.add_argument('--device',default='auto',choices=['cpu','cuda','auto']);parser.add_argument('--out',default=str(ROOT/'results'/'receipt.json'))
    args=parser.parse_args();torch.manual_seed(410);np.random.seed(410);random.seed(410)
    device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    trainseeds=range(1000,1000+args.train_scenes)
    testseeds=range(50000,50000+args.test_scenes)
    test=[gen(i) for i in testseeds]
    diffuser=Diffuser(TinyDDPM(),device=device)
    record=diffuser.train(trainseeds,args.steps)
    c=np.stack([condition(s,t) for s in test for t in range(2,N)]).astype('float32')
    keys=[(s['seed'],t) for s in test for t in range(2,N)]
    # A separate validation split chooses one scalar background correction.
    # This is genuine validation, not a post-hoc adjustment to the test receipt.
    val=[gen(i) for i in range(40000,40024)]
    vc=np.stack([condition(scene,t) for scene in val for t in range(2,N)]).astype('float32')
    vy=np.stack([scene['clean'][t] for scene in val for t in range(2,N)]).astype('float32')
    val_raw=diffuser.sample(vc,seed=991)
    hidden=1-vc[:,6:7]
    choices=np.arange(0,.501,.025)
    vals=[float(np.mean(np.square(np.maximum(0,val_raw-b)*hidden-vy*hidden))) for b in choices]
    best_bias=float(choices[int(np.argmin(vals))])
    tm=time.monotonic();raw=diffuser.sample(c,seed=1984);inftime=round(time.monotonic()-tm,2)
    # Outside the mask we preserve real observations. Inside it a calibrated
    # denoiser is no longer allowed to paint a diffuse grey background.
    hidden_test=1-c[:,6:7]
    imgs=c[:,0:2]*(1-hidden_test)+np.maximum(0,raw-best_bias)*hidden_test
    scores,examples,sequences=assess(test,imgs,keys)
    raw_scores,_,_=assess(test,raw,keys)
    scores['diffusion_raw']=raw_scores['diffusion']
    receipt={'experiment':'SihtiZipper Experiment 002: a trained pixel diffusion model vs explicit reversible object tracks',
             'time_basis':'Each prediction uses the current and earlier observations only; scoring against clean test frames is after prediction.',
             'data':{'train_scene_seeds':[1000,1000+args.train_scenes-1], 'test_scene_seeds':[50000,50000+args.test_scenes-1],
                     'video_frames':N,'image_shape':[2,H,W],'occluded_columns':list(BANDS),'identity':'two distinct color appearance channels visible when unoccluded',
                     'reverse_fraction':sum(s['reverse'] for s in test)/len(test),'ghost_observations':sum(len(s['ghosts']) for s in test)},
             'method':{'diffusion':'genuine 18-step DDPM epsilon predictor, learned by Adam on independent synthetic training sequences; conditional on current and previous two masked images',
                       'zipper':'online explicit separate tracking for two appearance channels, 1-step velocity estimates and a jam/relock/abstain gate; not a learned diffusion model',
                       'constant_velocity':'identical track estimate but never rejects a distant candidate',
                       'hold_last':'previous occluded image; no extrapolation',
                       'limit':'appearance channels reveal identity when visible; only hidden-center association probes delayed tracking. This comparison does not establish superiority on unlabelled natural video.'},
             'train':record,'parameters':sum(p.numel() for p in diffuser.model.parameters()),'validation':{'scene_seeds':[40000,40023],'background_bias':best_bias,'candidate_biases':len(choices),'validation_hidden_loss_min':round(min(vals),6)},'inference_seconds':inftime,'device':device,'scores':scores,'examples':examples,
             'case_summaries':[{'seed':s['seed'],'reversed':s['reverse'],'zipper_jams':sum(f['actions'].count('JAM_ABSTAIN') for f in seq['frames'])}for s,seq in zip(test,sequences)]}
    p=pathlib.Path(args.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(receipt,indent=2))
    modelp=p.with_name('tiny_ddpm.pt')
    torch.save({'state_dict':diffuser.model.cpu().state_dict(),'parameters':{'width':24,'ddpm_steps':18,'image_size':24,'channels':2},'train':record},modelp)
    print('Device:',device,'training:',record,'sampling seconds:',inftime,'validation bias:',best_bias)
    for k,v in scores.items():print(k,json.dumps(v))
    print('Wrote',p,'and',modelp)
if __name__=='__main__':main()
