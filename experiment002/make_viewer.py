#!/usr/bin/env python3
"""Build a self-contained 18-frame-per-scene filmstrip from recorded model weights.
Model and seeds are the precise scored run (the same sampling call and batch order).
"""
import base64, json, pathlib, sys
import numpy as np
from PIL import Image,ImageDraw
import torch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from compare import gen,condition,Diffuser,TinyDDPM,ZipperTracks,tracker_render,N,H,W
R=pathlib.Path(__file__).resolve().parent
result=json.loads((R/'results/receipt.json').read_text())
meta=result['data'];seeds=range(*[meta['test_scene_seeds'][0],meta['test_scene_seeds'][1]+1]);scenes=[gen(i) for i in seeds]
net=TinyDDPM();checkpoint=torch.load(R/'results/tiny_ddpm.pt',map_location='cpu',weights_only=True)
net.load_state_dict(checkpoint['state_dict']);d=Diffuser(net)
conds=np.stack([condition(scene,t) for scene in scenes for t in range(2,N)]).astype(np.float32)
raw=d.sample(conds,seed=1984)
mask=1-conds[:,6:7]
bias=result['validation']['background_bias'];cal=conds[:,0:2]*(1-mask)+np.maximum(0,raw-bias)*mask
rows=[];cases=[]
for j in (0,1):
    scene=scenes[j];tr=ZipperTracks();plain=ZipperTracks(gating=1e9)
    case=[]
    for t in range(N):
        p,action=tr.step(scene['obs'][t]);pb,_=plain.step(scene['obs'][t])
        if t<2:continue
        idx=j*(N-2)+t-2
        example={'seed':scene['seed'],'reverse':scene['reverse'],'t':t,'actions':action,
                 'truth':scene['truth'][t].round(2).tolist()}
        data={'clean':scene['clean'][t],'observed':scene['obs'][t],
              'zipper':tracker_render(p,scene['obs'][t],scene['mask']),
              'constant':tracker_render(pb,scene['obs'][t],scene['mask']),
              'diffusion':cal[idx],'raw':raw[idx]}
        for k,a in data.items():example[k]=base64.b64encode((np.clip(a,0,1)*255).astype(np.uint8).tobytes()).decode()
        case.append(example)
    cases.append(case)
score=result['scores'];compact=json.dumps(cases,separators=(',',':'));scorejson=json.dumps(score,separators=(',',':'))
html='''<!DOCTYPE html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SihtiZipper vs genuine pixel diffusion — Experiment 002</title><style>
:root{color-scheme:dark}*{box-sizing:border-box}body{background:#07141b;color:#e4f6f3;font:15px/1.5 system-ui,sans-serif;margin:0}main{max-width:1560px;margin:auto;padding:28px}h1{font-size:clamp(26px,4vw,44px);margin:0 0 7px}p{color:#aec8cd}a{color:#a4eade}.eyebrow{color:#8fe1cf;letter-spacing:.16em;font-size:11px;font-weight:800}.hero{max-width:1050px}.controls{display:flex;align-items:center;gap:15px;flex-wrap:wrap;background:#10242d;padding:15px;border-radius:13px;border:1px solid #29535d;margin:24px 0}button,select{padding:10px 16px;border:1px solid #6daea7;border-radius:8px;color:#ddfaf3;background:#193b41;cursor:pointer;font:inherit}input[type=range]{accent-color:#8edbc3;max-width:380px;width:50vw}.compare{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:15px}.card{padding:13px;border-radius:13px;background:#11232c;border:1px solid #2c4d54;min-width:0}.card h2{font-size:14px;margin:0 0 12px}.card canvas{display:block;width:100%;image-rendering:pixelated;background:black;aspect-ratio:1/1}.card p{font-size:12px;line-height:1.45}.stats{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.stat{padding:16px;background:#122931;border-radius:12px;border:1px solid #2a575d}.stat b{display:block;color:#e8fdef;font-size:21px}.note{border-left:3px solid #dfa66a;padding:12px 17px;background:#1a2830;margin:24px 0}.foot{font-size:12px;color:#9fc0c4;border-top:1px solid #2e5059;padding-top:18px}.hud{font:12px ui-monospace,Consolas,monospace;white-space:pre-wrap;color:#a6d6d0;overflow-wrap:anywhere}
@media(max-width:850px){.compare{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:480px){.compare,.stats{grid-template-columns:1fr}}</style><main><div class="eyebrow">SIHTIZIPPER · EXPERIMENT 002 · TRAINED MINI-DDPM</div><h1>Pixels imagined. Relationships preserved.</h1><p class="hero">These are <strong>two actual 16-frame test sequences</strong> from a 24-video frozen evaluation. A vertical screen hides two colored objects; one test video reverses their motion during occlusion. The diffusion neural net was trained on separate videos. The zipper-style tracker keeps explicit temporal identities and may reject false observations. This is a controlled toy experiment, <strong>not natural video or a general comparison with state-of-the-art diffusion.</strong></p><section class="controls"><label>Scenario <select id="scene"><option value="0">Crossing behind screen</option><option value="1">Reversal while hidden</option></select></label><button id="play">▶ Play</button><button id="prev">Previous</button><label>Frame <strong id="number">2</strong></label><input type="range" id="scrub" min="0" max="15" value="0"><button id="next">Next</button></section><section class="compare"><div class="card"><h2>01 / Hidden truth (test only)</h2><canvas id="clean" width="24" height="24"></canvas><p>The unoccluded clean video. Used for scoring, not shown to inference.</p></div><div class="card"><h2>02 / Camera observation</h2><canvas id="observed" width="24" height="24"></canvas><p>Opaque vertical screen, sensor noise and occasional false highlights. Shared information.</p></div><div class="card"><h2>03 / Diffusion, raw</h2><canvas id="raw" width="24" height="24"></canvas><p>18 steps of a genuinely trained pixel DDPM. Better foreground reproduction, but background hallucinations.</p></div><div class="card"><h2>04 / Diffusion, calibrated</h2><canvas id="diffusion" width="24" height="24"></canvas><p>Same DDPM, with an offset selected on 24 independent validation videos. Often erases the object too.</p></div><div class="card"><h2>05 / Constant velocity, no gate</h2><canvas id="constant" width="24" height="24"></canvas><p>Long-lived identity tracks, but forced acceptance of misleading camera observations.</p></div><div class="card"><h2>06 / Zipper-style jam and relock</h2><canvas id="zipper" width="24" height="24"></canvas><p>Separate object tracks, a jam/abstain state, and no rewriting of older observations.</p></div></section><div class="note"><strong>Do not compare unlike claims.</strong> Pixels are scored by unseen-region MSE. Accepted associations and false joins are scored only for the trackers. The diffusion model makes pictures, not explicit observed-identity assertions. The tracker has a longer remembered state than the DDPM's two-frame conditioning window.</div><div class="stats"><div class="stat">Hidden strip mean-squared error<b id="pixelstats">—</b><span>Lower means better pixel reconstruction.</span></div><div class="stat">False observed associations<b id="linkstats">—</b><span>These counts refer to trackers only; refusing a link sacrifices coverage.</span></div></div><p id="details" class="hud"></p><p class="foot"><a href="README.md">Read protocol &amp; limitations</a> · <a href="results/summary.json">Numerical results</a> · <a href="../index.html">Original SihtiZipper switchyard</a>. This viewer embeds the precomputed frames and runs without any network, Python or camera access.</p></main><script>
const CASES=__CASES__, SCORES=__SCORES__;
const ids=['clean','observed','raw','diffusion','constant','zipper'];let timer=null;const s=document.getElementById('scrub');
function draw(id,b64){const ctx=document.getElementById(id).getContext('2d'),raw=atob(b64),image=ctx.createImageData(24,24);for(let j=0;j<576;j++){const a=raw.charCodeAt(j),b=raw.charCodeAt(j+576),i=j*4;image.data[i]=Math.min(255,a*.99+b*.13+10);image.data[i+1]=Math.min(255,a*.30+b*.91+10);image.data[i+2]=Math.min(255,a*.28+b*.76+13);image.data[i+3]=255;if(id==='observed'&&j%24>=9&&j%24<15){image.data[i]=27;image.data[i+1]=43;image.data[i+2]=53;}}ctx.putImageData(image,0,0)}
function render(){const e=CASES[+document.getElementById('scene').value][+s.value];document.getElementById('number').textContent=e.t;for(let id of ids)draw(id,e[id]);document.getElementById('details').textContent=`Test video seed ${e.seed}  ·  ${e.reverse?'Reversal':'Crossing'}  ·  Frame ${e.t}  ·  Zipper decisions: ${e.actions.join(' / ')}  ·  Truth centers: ${JSON.stringify(e.truth)}`;document.getElementById('pixelstats').textContent=`Zipper ${SCORES.zipper.hidden_region_mse.toFixed(4)} · DDPM raw ${SCORES.diffusion_raw.hidden_region_mse.toFixed(4)} · calibrated ${SCORES.diffusion.hidden_region_mse.toFixed(4)}`;document.getElementById('linkstats').textContent=`Zipper ${SCORES.zipper.false_observation_joins}/${SCORES.zipper.observation_joins} · Ungated ${SCORES.constant_velocity.false_observation_joins}/${SCORES.constant_velocity.observation_joins}`}
document.getElementById('prev').onclick=()=>{s.value=Math.max(0,+s.value-1);render()};document.getElementById('next').onclick=()=>{s.value=Math.min(CASES[0].length-1,+s.value+1);render()};s.max=CASES[0].length-1;s.oninput=render;document.getElementById('scene').onchange=()=>{s.value=0;render()};document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ Play'}else{document.getElementById('play').textContent='Pause';timer=setInterval(()=>{s.value=(+s.value+1)%CASES[0].length;render()},250)}};render();
</script></html>'''.replace('__CASES__',compact).replace('__SCORES__',scorejson)
(R/'viewer.html').write_text(html)
# A five-frame contact sheet from the exact same trained model/sample.
shots=[3,6,8,11,14];ids=['clean','observed','raw','diffusion','constant','zipper']
scale=7;tile=24*scale;pad=10;label=24;out=Image.new('RGB',(len(ids)*(tile+pad)+pad,len(shots)*(tile+label+pad)+pad),'#0b1b23');p=ImageDraw.Draw(out)
for row,t in enumerate(shots):
    e=cases[0][t-2]
    for col,k in enumerate(ids):
        b=base64.b64decode(e[k]);a=np.frombuffer(b,dtype=np.uint8).reshape((2,H,W))/255
        rgb=np.stack([np.clip(a[0]+.13*a[1],0,1),np.clip(.30*a[0]+.90*a[1],0,1),np.clip(.28*a[0]+.75*a[1],0,1)],axis=-1)
        x=pad+col*(tile+pad);y=pad+row*(tile+label+pad)
        out.paste(Image.fromarray((rgb*255).astype('uint8')).resize((tile,tile),Image.Resampling.NEAREST),(x,y+label));p.text((x+3,y+3),f'{k.upper()} / t={t}',fill='#daefea')
        if k=='observed':p.rectangle((x+9*scale,y+label,x+15*scale-1,y+label+tile-1),fill='#1b2b35')
out.save(R/'results/contact_sheet.png',optimize=True)
print('Wrote offline viewer, 2 scenarios x 16 frames, and benchmark contact sheet')
