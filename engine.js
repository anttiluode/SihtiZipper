/* SihtiZipper — explicitly reversible links between original vector tracks.
   This is a small, inspectable online state machine, not trained attention. */
(function(root,factory){const v=factory();if(typeof module==='object'&&module.exports)module.exports=v;root.SihtiZipper=v;})(typeof globalThis!=='undefined'?globalThis:this,function(){
 'use strict';
 const TRACKS=['mid','fast'];
 function distance(a,b){if(!a||!b||!Array.isArray(a.key)||!Array.isArray(b.key)||a.key.length!==b.key.length)return Infinity;let s=0;for(let i=0;i<a.key.length;i++){const d=a.key[i]-b.key[i];s+=d*d;}return s/a.key.length;}
 function clip(v,a,b){return Math.max(a,Math.min(b,v));}
 function candidates(tape,anchorIndex,track,delays=[0,1,2]){
  const ref=tape[anchorIndex]?.core;
  return delays.map(d=>({lag:d,index:anchorIndex-d, cost:distance(ref,tape[anchorIndex-d]?.[track])}));
 }
 class Seam {
  constructor(track,opts={}){this.track=track;this.threshold=opts.threshold??0.058;this.slipMargin=opts.slipMargin??0.008;this.relock=opts.relock??2;this.state='search';this.offset=null;this.streak=0;this.jams=0;this.slips=0;this.detaches=0;this.locks=0;this.lastCost=null;}
  inspect(choices){
   const ordered=choices.slice().sort((a,b)=>a.cost-b.cost),best=ordered[0];
   const incumbent=choices.find(c=>c.lag===this.offset);
   let chosen=null,action='search';
   if(this.offset!==null&&incumbent&&incumbent.cost<=this.threshold){
    // A stored seam resists switching for a near-tie: local phase continuity.
    if(best.lag!==this.offset&&best.cost+this.slipMargin<incumbent.cost&&best.cost<=this.threshold){
      chosen=best;action='slip';this.slips++;this.offset=best.lag;
    }else{chosen=incumbent;action='hold';}
    this.state='locked';this.streak=0;
   }else if(best.cost<=this.threshold){
    if(this.offset!==null&&best.lag!==this.offset){this.slips++;action='slip';}else action='relock';
    chosen=best;this.offset=best.lag;this.state='locked';this.streak=0;this.locks++;
   }else{
    // A broken tooth must not be forced into a match. Old links stay archived.
    if(this.state==='locked'){this.jams++;action='jam';}else action='open';
    this.streak++;this.state='open';if(this.streak>=this.relock&&this.offset!==null){this.offset=null;this.detaches++;action='detach';}
   }
   this.lastCost=chosen?chosen.cost:best.cost;
   return {track:this.track,state:this.state,action,offset:this.offset,match:chosen?{index:chosen.index,lag:chosen.lag,cost:chosen.cost}:null,best:best.cost,incumbent:incumbent?.cost??null,stress:clip(best.cost/this.threshold,0,25),choices:choices.map(c=>({...c}))};
  }
 }
 class Zipper {
  constructor(opts={}){this.opts={threshold:opts.threshold??0.058,slipMargin:opts.slipMargin??0.008,relock:opts.relock??2};this.reset();}
  reset(){this.tape=[];this.seams=[];this.opened=new Set();this.sliders={mid:new Seam('mid',this.opts),fast:new Seam('fast',this.opts)};this.stats={steps:0,linked:0,open:0,jams:0,slips:0};}
  append(observation){
   // Separate incoming immutable observations from diagnostic oracle truth.
   const clean={};for(const name of ['core',...TRACKS]){
    const item=observation[name];if(!item||!Array.isArray(item.key)||!Array.isArray(item.payload))throw Error('Malformed '+name+' vector');
    clean[name]={key:Object.freeze(item.key.slice()),payload:Object.freeze(item.payload.slice())};Object.freeze(clean[name]);
   }
   Object.freeze(clean);const tick=this.tape.length;this.tape.push(clean);
   if(tick<2)return {t:tick,anchor:null,links:[],actions:[],state:'priming'};
   const anchor=tick,links=[],actions=[];
   for(const name of TRACKS){
    const o=this.sliders[name].inspect(candidates(this.tape,anchor,name));actions.push(o);
    if(o.match)links.push({track:name,source:o.match.index,lag:o.match.lag,cost:o.match.cost});
    if(o.action==='jam')this.stats.jams++;
    if(o.action==='slip')this.stats.slips++;
   }
   const entry=Object.freeze({t:tick,anchor,links:links.map(x=>Object.freeze({...x})),actions:actions.map(x=>({track:x.track,state:x.state,action:x.action,offset:x.offset,stress:x.stress,cost:x.match?.cost??x.best}))});
   this.seams.push(entry);this.stats.steps++;this.stats.linked+=links.length;this.stats.open+=TRACKS.length-links.length;
   return {...entry,diagnostics:actions};
  }
  // A moving access bubble can release/restore archived links independently
  // of the original signal and independently of future slider decisions.
  // This is link-level undo, not time travel or a reclassification.
  window(start,end,track=null,open=true){
    if(track!==null&&!TRACKS.includes(track))throw Error('Invalid track');
    let count=0;for(const e of this.seams){
      if(e.t<start||e.t>end)continue;
      for(const l of e.links){if(track!==null&&l.track!==track)continue;
        const key=e.t+':'+l.track,was=this.opened.has(key);
        if(open)this.opened.add(key);else this.opened.delete(key);
        if(was!==this.opened.has(key))count++;
      }
    }return count;
  }
  activeLinks(t){const e=this.seams.find(e=>e.t===t);return e?e.links.filter(l=>!this.opened.has(t+':'+l.track)):[];}
  // Reversibility is literal: every original tooth is preserved in the tape;
  // the seam contains references only and cannot modify a stored vector.
  original(t,track){const x=this.tape[t]?.[track];return x?{key:x.key.slice(),payload:x.payload.slice()}:null;}
  export(){return {version:1,options:{...this.opts},tape:this.tape.map(frame=>Object.fromEntries(['core',...TRACKS].map(n=>[n,this.originalFrom(frame,n)]))),seams:this.seams.map(e=>({t:e.t,anchor:e.anchor,links:e.links.map(x=>({...x})),actions:e.actions.map(x=>({...x}))})),stats:{...this.stats},opened:Array.from(this.opened)};}
  originalFrom(frame,name){return {key:frame[name].key.slice(),payload:frame[name].payload.slice()};}
  detach(track){if(!TRACKS.includes(track))throw Error('Invalid track');const s=this.sliders[track];s.state='open';s.offset=null;s.streak=0;s.detaches++;}
 }
 function fixedLinks(tape,t,offsets={mid:1,fast:0}){if(t<2)return [];return TRACKS.map(name=>({track:name,source:t-offsets[name],lag:offsets[name],cost:distance(tape[t]?.core,tape[t-offsets[name]]?.[name])}));}
 function greedyLinks(tape,t,threshold=0.058){if(t<2)return [];return TRACKS.flatMap(name=>{const c=candidates(tape,t,name).sort((a,b)=>a.cost-b.cost)[0];return c.cost<=threshold?[{track:name,source:c.index,lag:c.lag,cost:c.cost}]:[];});}
 return {Zipper,Seam,distance,candidates,fixedLinks,greedyLinks,TRACKS};
});
