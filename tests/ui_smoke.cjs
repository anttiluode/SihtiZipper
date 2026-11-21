'use strict';
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const IDs=['play','step','reset','seed','noise','noise-text','rate','fault','slip','unzip','rezip','export','tracks','history','s-t','s-zip','s-fixed','s-greedy','s-events','ledger','readout','bench-button','bench'];
const nodes={};
const dummyCanvas=()=>({clearRect(){},fillRect(){},beginPath(){},moveTo(){},lineTo(){},stroke(){},fillText(){},arc(){},fill(){},bezierCurveTo(){},setLineDash(){}});
for(const id of IDs){const canvas=['tracks','history'].includes(id);nodes[id]={id,value:id==='seed'?'7':id==='noise'?'0.13':id==='rate'?'14':'',textContent:'',innerHTML:'',listeners:{},childNodes:[],getContext:dummyCanvas,addEventListener(type,fn){this.listeners[type]=fn;},replaceChildren(){this.childNodes=[]},prepend(node){this.childNodes.unshift(node);},get lastChild(){return this.childNodes[this.childNodes.length-1]},remove(){},width:canvas?1200:0,height:id==='tracks'?310:id==='history'?168:0};}
const doc={getElementById(id){assert.ok(nodes[id],id);return nodes[id];},createElement(){return{className:'',childNodes:[],append(...x){this.childNodes.push(...x)},click(){},remove(){}};},createTextNode(x){return{value:x}}};
const sandbox={document:doc,console,structuredClone,globalThis:null,setInterval(){return 99},clearInterval(){},setTimeout(){return 0},URL:{createObjectURL(){return 'blob:x'},revokeObjectURL(){}},Blob:class{constructor(parts){this.parts=parts}}};sandbox.globalThis=sandbox;
for(const file of ['engine.js','world.js','app.js'])vm.runInNewContext(fs.readFileSync(path.join(__dirname,'..',file),'utf8'),sandbox,{filename:file});
function click(id){assert.ok(nodes[id].listeners.click,'no click '+id);nodes[id].listeners.click();}
assert.match(nodes['s-t'].textContent,/0 \/ 390/);
for(let i=0;i<100;i++)click('step');
assert.match(nodes['s-t'].textContent,/100 \/ 390/);
assert.ok(!nodes['s-zip'].textContent.includes('NaN'));
click('fault');click('slip');for(let i=0;i<70;i++)click('step');click('unzip');click('rezip');click('export');
click('bench-button');assert.match(nodes.bench.textContent,/10 independent seeds/);
click('play');click('play');click('reset');assert.match(nodes['s-t'].textContent,/0 \/ 390/);
console.log('PASS: offline UI controls, tick stepping, future faults, phase interventions, baselines, benchmark and reset');
