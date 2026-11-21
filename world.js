/* Deterministic, precommitted streams. Ground truth NEVER enters engine.append. */
(function(root,factory){const v=factory();if(typeof module==='object'&&module.exports)module.exports=v;root.SihtiZipperWorld=v;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 function rng(seed){let a=seed>>>0;return()=>{a=(a+0x6D2B79F5)|0;let t=Math.imul(a^(a>>>15),1|a);t^=t+Math.imul(t^(t>>>7),61|t);return((t^(t>>>14))>>>0)/4294967296;};}
 function make(seed=7,n=390,settings={}){
  const random=rng(seed),noise=settings.noise??0.13,hard=settings.hard??false;
  const rand=()=>random()*2-1;
  const latent=Array.from({length:n+16},(_,t)=>{
   const phase=t%23;
   // Mostly distinguishable 4D teeth, plus occasional deliberately repeated
   // patterns which challenge any purely local, instantaneous reader.
   return Array.from({length:4},(_,j)=>0.64*rand()+0.19*Math.sin(.23*t+j*1.7)+0.08*Math.cos(.7*phase+j));
  });
  const frame=[],truth=[];
  for(let t=0;t<n;t++){
   const midLag=(t>=80&&t<144)?2:(t>=196&&t<250)?0:(t>=302&&t<336)?2:1;
   const fastLag=(t>=100&&t<172)?1:(t>=225&&t<290)?2:0;
   const faults={mid:(t>=146&&t<161)||(t>=276&&t<288),fast:(t>=174&&t<189)||(t>=340&&t<355)};
   const indices={core:Math.max(0,t-2),mid:Math.max(0,t-2+midLag),fast:Math.max(0,t-2+fastLag)};
   const readings={};
   for(const name of ['core','mid','fast']){
    const s=latent[indices[name]],fault=faults[name]||false;
    let key=s.map(x=>x+(random()-.5)*2*noise);
    if(fault)key=key.map(()=>rand()*2.9);
    if(hard&&name==='mid'&&t%37===0)key=key.map(x=>x*-.2);
    const payload=[rand(),rand(),rand()];
    readings[name]={key,payload};
   }
   frame.push(readings);truth.push({t,indices,lags:{mid:midLag,fast:fastLag},faults});
  }
  return {seed,noise,n,frames:frame,truth,description:'Core+two private vector tracks, known phase slips and intermittent broken teeth. Keys are exposed to the reader; causal source indices are test-only truth.'};
 }
 // A pair is correct if both stored teeth came from precisely the same source
 // latent tick (not merely if their noisy feature vectors look similar).
 function scoreLink(link,t,truth){return truth[link.source]?.indices?.[link.track]===truth[t]?.indices?.core;}
 function evaluate(world,opts={}){
  const engine=opts.engine||(typeof require==='function'?require('./engine.js'):globalThis.SihtiZipper);
  const zip=new engine.Zipper(opts);const counts={zip:{correct:0,wrong:0,open:0},fixed:{correct:0,wrong:0,open:0},greedy:{correct:0,wrong:0,open:0}};
  const log=[];
  for(let t=0;t<world.n;t++){
   const result=zip.append(world.frames[t]);if(t<5)continue;
   const all={zip:result.links,fixed:engine.fixedLinks(zip.tape,t),greedy:engine.greedyLinks(zip.tape,t,opts.threshold??.058)};
   for(const [mode,links] of Object.entries(all)){
    for(const name of ['mid','fast']){
     const l=links.find(x=>x.track===name);
     if(!l)counts[mode].open++;
     else if(scoreLink(l,t,world.truth))counts[mode].correct++;
     else counts[mode].wrong++;
    }
   }
   log.push({t,links:result.links,actions:result.actions});
  }
  const total=(world.n-5)*2;
  return {seed:world.seed,noise:world.noise,total,counts:Object.fromEntries(Object.entries(counts).map(([k,v])=>[k,{...v,accuracy:v.correct/total,falseJoinRate:v.wrong/total,coverage:(v.correct+v.wrong)/total,utility:(v.correct-2*v.wrong)/total}])),mechanics:{...zip.stats},log,zipper:zip};
 }
 return {rng,make,evaluate,scoreLink};
});
