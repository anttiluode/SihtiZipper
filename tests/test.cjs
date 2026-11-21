'use strict';
const assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path');
const E=require('../engine.js'),W=require('../world.js');
const test=(label,f)=>{f();console.log('PASS:',label);};
const item=(n,noise=0)=>({key:[n+noise,0,0,0],payload:[.1,-1,3]});
test('reversible seam stores raw vectors as immutable, reconstructable originals',()=>{
 const z=new E.Zipper();const sent=[];
 for(let t=0;t<12;t++){const f={core:item(t),mid:item(t+1),fast:item(t)};sent.push(structuredClone(f));z.append(f);f.core.key[0]=-9999;}
 z.detach('mid');assert.deepEqual(z.export().tape,sent);
 assert.deepEqual(z.original(7,'mid'),sent[7].mid);
 assert.ok(z.seams.length>0);assert.ok(z.seams[0].links.every(l=>l.source>=0));
});
test('unzip / rezip release only the seam, restoring original links exactly',()=>{
 const z=new E.Zipper({threshold:.001});
 for(let t=0;t<22;t++)z.append({core:item(t),mid:item(t),fast:item(t)});
 const before=z.export(),beforeLinks=z.activeLinks(17);
 assert.equal(beforeLinks.length,2);
 assert.equal(z.window(15,18,'mid',true),4);
 assert.equal(z.activeLinks(17).length,1);
 assert.equal(z.activeLinks(17)[0].track,'fast');
 assert.equal(z.window(15,18,'mid',false),4);
 assert.deepEqual(z.activeLinks(17),beforeLinks);
 assert.deepEqual(z.export().tape,before.tape);
 assert.deepEqual(z.export().seams,before.seams);
});
test('causality: future changes cannot alter earlier seam or classification',()=>{
 const w=W.make(62,100),a=new E.Zipper(),b=new E.Zipper();
 const before=[];
 for(let t=0;t<60;t++){a.append(w.frames[t]);b.append(w.frames[t]);}
 const rec=JSON.stringify(a.export());w.frames[71].mid.key=[333,444,555,666];
 for(let t=60;t<100;t++)b.append(w.frames[t]);
 assert.equal(JSON.stringify(a.export()),rec);
 for(let t=0;t<a.seams.length;t++)assert.deepEqual(a.seams[t],b.seams[t]);
});
test('known one-tick phase slip locks and switches without sampling the future',()=>{
 const z=new E.Zipper({threshold:0.001});
 const values=Array.from({length:25},(_,i)=>[Math.sin(i*19),Math.cos(i*11),Math.sin(i*7),Math.cos(i*3)]);
 const history=[];
 for(let t=0;t<20;t++){
  const lag=t<9?1:2;
  const f={core:{key:values[Math.max(0,t-2)],payload:[t]},mid:{key:values[Math.max(0,t-2+lag)],payload:[t]},fast:{key:values[Math.max(0,t-2)],payload:[t]}};
  history.push(z.append(f));
 }
 assert.ok(history.some(x=>x.actions?.some(a=>a.action==='slip')));
 assert.equal(history[18].links.find(x=>x.track==='mid').lag,2);
 assert.equal(history[18].links.find(x=>x.track==='fast').lag,0);
});
test('a broken tooth opens, never forces a false link, then recovers',()=>{
 const z=new E.Zipper({threshold:0.001});
 for(let t=0;t<6;t++)z.append({core:item(t),mid:item(t),fast:item(t)});
 let out;
 for(let t=6;t<10;t++){out=z.append({core:item(t),mid:item(99),fast:item(t)});assert.equal(out.links.some(x=>x.track==='mid'),false);}
 assert.ok(z.stats.jams>0);assert.ok(z.sliders.mid.detaches>0);
 out=z.append({core:item(10),mid:item(10),fast:item(10)});
 assert.ok(out.links.some(x=>x.track==='mid'));
 assert.deepEqual(z.original(6,'mid').key,item(99).key);
});
test('oracle truth is not copied to model tape',()=>{
 const w=W.make(9,8),z=new E.Zipper();
 z.append({...w.frames[0],oracleAnswer:'secret',truth:w.truth[0]});
 assert.equal(Object.hasOwn(z.tape[0],'truth'),false);
 assert.equal(Object.hasOwn(z.tape[0],'oracleAnswer'),false);
});
test('deterministic replay and reproducible ledger',()=>{
 const w=W.make(42,390,{noise:.13});const a=W.evaluate(w),b=W.evaluate(W.make(42,390,{noise:.13}));
 assert.deepEqual(a.counts,b.counts);assert.deepEqual(a.mechanics,b.mechanics);
 assert.equal(a.total,770);
});
const results=[];
for(const noise of [.08,.13,.18,.24]){
 const aggregate={};for(const k of ['zip','fixed','greedy'])aggregate[k]={correct:0,wrong:0,open:0};
 let slips=0,jams=0;
 for(let seed=1;seed<=40;seed++){
  const run=W.evaluate(W.make(seed,390,{noise}));
  slips+=run.mechanics.slips;jams+=run.mechanics.jams;
  for(const k of Object.keys(aggregate))for(const m of ['correct','wrong','open'])aggregate[k][m]+=run.counts[k][m];
 }
 const total=40*770;
 const row={noise,seeds:40,total,methods:{},meanSlips:slips/40,meanJams:jams/40};
 for(const [name,c] of Object.entries(aggregate))row.methods[name]={...c,correctRate:c.correct/total,falseJoinRate:c.wrong/total,coverage:(c.correct+c.wrong)/total,utility:(c.correct-2*c.wrong)/total};
 results.push(row);
 console.log('BENCH',noise,'zip',row.methods.zip.correctRate.toFixed(3),'fixed',row.methods.fixed.correctRate.toFixed(3),'greedy',row.methods.greedy.correctRate.toFixed(3));
}
test('zipper improves correct binding over fixed-offset when tracks slip (40 held-out seeds)',()=>{
 for(const row of results.slice(0,3))assert.ok(row.methods.zip.correctRate>row.methods.fixed.correctRate+.15);
});
test('seam continuity slightly reduces false joins versus nearest-per-frame in noisy controls',()=>{
 const r=results[2];assert.ok(r.methods.zip.falseJoinRate<r.methods.greedy.falseJoinRate);
});
const receipt={name:'SihtiZipper controlled switchyard',version:1,date:'2026-11-21',protocol:{world:'3 synthetic 4D keyed streams plus private vector payloads',trackCount:3,lookbacks:[0,1,2],score:'source-tick identity for bound pairs; test-only oracle not passed to reader',seedSet:'1..40 fixed',windows:390,threshold:.058,slipMargin:.008,falseJoinPenalty:2,note:'No supervised training. All methods see the same causal observations. Fixed baseline has no abstention. Greedy and zipper use identical match threshold; zipper adds seam continuity. No real-sensor claims.'},results};
const out=path.join(__dirname,'../results/benchmark.json');fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify(receipt,null,2)+'\n');console.log('SAVED',out);
