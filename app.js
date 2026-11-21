(function(){'use strict';
const E=globalThis.SihtiZipper,W=globalThis.SihtiZipperWorld;
const $=id=>document.getElementById(id);
const ctx=id=>$(id).getContext('2d');
const COLORS={core:'#8bdad8',mid:'#f7be77',fast:'#b99bfa',good:'#83e0ac',bad:'#f07888',slip:'#80d8fb'};
let world,head,t=-1,timer=null,last=null,stats,log;
function reset(){stop();const seed=Number($('seed').value)||7,noise=Number($('noise').value);world=W.make(seed,390,{noise});head=new E.Zipper();t=-1;last=null;log=[];stats={zip:{correct:0,wrong:0,open:0},fixed:{correct:0,wrong:0,open:0},greedy:{correct:0,wrong:0,open:0},total:0};$('noise-text').textContent=noise.toFixed(2);$('ledger').replaceChildren();$('bench').textContent='Prediction: not tested. This experiment measures binding correctness, not next-frame forecasting.';render();}
function stop(){if(timer!==null){clearInterval(timer);timer=null;}$('play').textContent='▶ Play';}
function play(){if(timer!==null){stop();return;}timer=setInterval(()=>{if(!step())stop();},1000/Number($('rate').value));$('play').textContent='❚❚ Pause';}
function scoreOne(mode,links){for(const track of E.TRACKS){const l=links.find(x=>x.track===track);if(!l)stats[mode].open++;else if(W.scoreLink(l,t,world.truth))stats[mode].correct++;else stats[mode].wrong++;}}
function step(){if(t>=world.n-1)return false;t++;last=head.append(world.frames[t]);if(t>=5){stats.total+=2;scoreOne('zip',last.links);scoreOne('fixed',E.fixedLinks(head.tape,t));scoreOne('greedy',E.greedyLinks(head.tape,t));}if(t>=2){log.push(last);for(const a of last.actions)if(!['hold','open','search'].includes(a.action))event(`${a.track.toUpperCase()} ${a.action.toUpperCase()} · lag=${a.offset??'?'} · tension=${a.stress.toFixed(2)}`,t);}render();return true;}
function event(message,at=t){const node=document.createElement('div');node.className='event';const strong=document.createElement('strong');strong.textContent=`#${String(at).padStart(3,'0')} `;node.append(strong,document.createTextNode(message));$('ledger').prepend(node);while($('ledger').childNodes.length>80)$('ledger').lastChild.remove();}
function text(id,s){$(id).textContent=s;}
function percent(mode){const c=stats[mode];return stats.total?((c.correct/stats.total)*100).toFixed(1)+'%':'—';}
function drawGrid(c,w,h){c.clearRect(0,0,w,h);c.fillStyle='#08171c';c.fillRect(0,0,w,h);c.strokeStyle='#17343b';c.lineWidth=1;for(let x=0;x<w;x+=34){c.beginPath();c.moveTo(x,0);c.lineTo(x,h);c.stroke();}}
function keyColor(v,name){const base=COLORS[name],intensity=Math.min(1,Math.abs(v?.key?.[0]??0)/.85);return base+(Math.round((.26+intensity*.7)*255).toString(16).padStart(2,'0'));}
function drawTracks(){const canvas=$('tracks'),c=ctx('tracks'),width=canvas.width,height=canvas.height;drawGrid(c,width,height);const yy={core:69,mid:157,fast:245};const left=100,stepX=37,start=Math.max(0,t-25),end=Math.max(26,t+2);const xFor=i=>left+(i-start)*stepX;
 c.font='600 14px ui-monospace,Consolas,monospace';
 for(const track of ['core','mid','fast']){
  c.fillStyle=COLORS[track];c.fillText(track.toUpperCase(),14,yy[track]+5);c.strokeStyle=COLORS[track]+'55';c.lineWidth=2;c.beginPath();c.moveTo(left-14,yy[track]);c.lineTo(width-26,yy[track]);c.stroke();
  for(let j=start;j<=Math.min(t,end);j++){const x=xFor(j);if(x>width-18)break;const raw=head.tape[j]?.[track];const fault=world.truth[j]?.faults[track];c.fillStyle=fault?COLORS.bad:keyColor(raw,track);c.strokeStyle=fault?COLORS.bad:COLORS[track];c.lineWidth=fault?2.7:1.4;c.beginPath();c.arc(x,yy[track],8,0,2*Math.PI);c.fill();c.stroke();if(j%5===0){c.font='10px ui-monospace,monospace';c.fillStyle='#668a92';c.fillText(String(j),x-8,yy[track]+24);}}
 }
 if(t>=2&&last){const x=xFor(t);c.setLineDash([2,5]);c.strokeStyle='#90bdbd';c.beginPath();c.moveTo(x,40);c.lineTo(x,278);c.stroke();c.setLineDash([]);
  for(const name of ['mid','fast']){const l=last.links.find(l=>l.track===name),a=last.actions.find(x=>x.track===name),startx=xFor(t);if(l){const sourcex=xFor(l.source),cy=yy[name];c.strokeStyle=a?.action==='slip'?COLORS.slip:COLORS.good;c.lineWidth=3;c.beginPath();c.moveTo(startx,yy.core+9);c.bezierCurveTo(startx,yy.core+48,sourcex,cy-44,sourcex,cy-9);c.stroke();c.fillStyle=c.strokeStyle;c.beginPath();c.arc(sourcex,cy,12,0,2*Math.PI);c.stroke();}
   else if(a){c.strokeStyle=COLORS.bad;c.lineWidth=3;c.beginPath();c.moveTo(startx-9,yy[name]-9);c.lineTo(startx+9,yy[name]+9);c.moveTo(startx+9,yy[name]-9);c.lineTo(startx-9,yy[name]+9);c.stroke();}
  }
 }
}
function drawHistory(){const canvas=$('history'),c=ctx('history'),w=canvas.width,h=canvas.height;drawGrid(c,w,h);c.font='600 13px ui-monospace,monospace';const show=log.slice(-150),bw=Math.min(7.2,(w-110)/Math.max(1,show.length)),start=90;for(const [track,y] of [['mid',53],['fast',116]]){
 c.fillStyle=COLORS[track];c.fillText(track.toUpperCase(),14,y+5);for(let i=0;i<show.length;i++){const e=show[i],a=e.actions.find(x=>x.track===track);let color=a?.action==='slip'?'#71d6fc':a?.state==='locked'?'#83d4ad':a?.action==='jam'?'#fd9c60':'#bb5969';c.fillStyle=color;c.fillRect(start+i*bw,y-15,Math.max(2,bw-.8),31);}
 }c.fillStyle='#84a9ac';c.font='11px monospace';if(show.length)c.fillText('frames '+show[0].t+' .. '+show[show.length-1].t,w-190,153);
}
function drawReadout(){if(t<2){text('readout','Ready to join independent 4D vector streams. Press Step or Play.');return;}const lines=[`CURRENT CORE tooth #${t} · 4D key = [${head.original(t,'core').key.map(n=>n.toFixed(2)).join(', ')}]`,`The zipper copies no payload; it only saves indices.`,``];for(const track of ['mid','fast']){
 const state=last.actions.find(a=>a.track===track),link=last.links.find(x=>x.track===track);
 lines.push(`${track.toUpperCase().padEnd(5)} ${state.state.toUpperCase()} / ${state.action.toUpperCase()} · lag=${state.offset??'—'} · tension ${(state.stress).toFixed(2)}`);
 if(link){const v=head.original(link.source,track);lines.push(`      tooth #${link.source} · key [${v.key.map(n=>n.toFixed(2)).join(', ')}]`);lines.push(`      PRIVATE payload [${v.payload.map(n=>n.toFixed(2)).join(', ')}] · unchanged`);}else lines.push(`      JAM / OPEN: original vector still exists on the tape.`);
 lines.push('');}text('readout',lines.join('\n'));
}
function render(){text('s-t',`${Math.max(0,t+1)} / ${world.n}`);text('s-zip',percent('zip'));text('s-fixed',percent('fixed'));text('s-greedy',percent('greedy'));text('s-events',`${head.stats.jams} / ${head.stats.slips}`);drawTracks();drawHistory();drawReadout();}
function injectFault(){if(t>=world.n-16)return;const start=Math.max(3,t+2);for(let j=start;j<Math.min(world.n,start+12);j++){const h=world.frames[j].mid;h.key=h.key.map((v,i)=>Math.sin((j+47)*(i+5)*1.91)*3.4);world.truth[j].faults.mid=true;}event('Manual MID broken-tooth interval scheduled at '+start,start);}
function injectSlip(){if(t>=world.n-36)return;const start=Math.max(4,t+3),end=Math.min(world.n-1,start+28);const copy=world.frames.map(f=>structuredClone(f.mid)),source=world.truth.map(v=>v.indices.mid);
 for(let j=start;j<end;j++){world.frames[j].mid=structuredClone(copy[j+1]);world.truth[j].indices.mid=source[j+1];}event('Manual MID +1 source-tick phase offset scheduled at '+start,start);}
function runBench(){const noise=Number($('noise').value),modes=['zip','fixed','greedy'];const totals=Object.fromEntries(modes.map(k=>[k,{correct:0,wrong:0,open:0}]));for(let seed=101;seed<=110;seed++){const e=W.evaluate(W.make(seed,390,{noise}));for(const mode of modes)for(const k of ['correct','wrong','open'])totals[mode][k]+=e.counts[mode][k];}
 const denom=10*770;const lines=[`10 independent seeds (101..110), noise ${noise.toFixed(2)}`,`Method       correct      wrong       open`];for(const m of modes){const a=totals[m];lines.push(`${m.padEnd(10)} ${(100*a.correct/denom).toFixed(1).padStart(6)}% ${(100*a.wrong/denom).toFixed(1).padStart(9)}% ${(100*a.open/denom).toFixed(1).padStart(9)}%`);}lines.push('Scoring uses world truth ONLY after each step.');text('bench',lines.join('\n'));}
function exportReceipt(){const receipt={version:1,kind:'SihtiZipper browser experiment',date:new Date().toISOString(),config:{seed:world.seed,noise:world.noise},score:stats,sliderStats:head.stats,events:log.map(e=>({t:e.t,anchor:e.anchor,links:e.links,actions:e.actions})),source:'Synthetic keyed vector streams. No camera or real Sihti pixel bands. Full raw tape omitted; numerical seam and score only.'};const url=URL.createObjectURL(new Blob([JSON.stringify(receipt,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='sihtizipper_receipt.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);}
$('play').addEventListener('click',play);$('step').addEventListener('click',()=>{stop();step()});$('reset').addEventListener('click',reset);$('seed').addEventListener('change',reset);$('noise').addEventListener('input',()=>{$('noise-text').textContent=Number($('noise').value).toFixed(2)});$('noise').addEventListener('change',reset);$('rate').addEventListener('change',()=>{if(timer!==null){stop();play()}});$('fault').addEventListener('click',injectFault);$('slip').addEventListener('click',injectSlip);$('bench-button').addEventListener('click',runBench);$('export').addEventListener('click',exportReceipt);reset();
})();
