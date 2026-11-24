/* A single same-origin quantized/float ONNX model; never uploads images.
   This worker will only run if models/model-manifest.json and the ONNX weights exist. */
'use strict';
let ortSession=null,manifest=null,loadPromise=null,serial=Promise.resolve();
const reply=(id,type,fields={})=>self.postMessage({id,type,...fields});
async function load(info){
 if(ortSession)return true;if(loadPromise)return loadPromise;
 loadPromise=(async()=>{
  importScripts('../vendor/ort/ort.min.js');
  if(!self.ort)throw Error('로컬 ONNX 라이브러리를 찾을 수 없습니다');
  ort.env.wasm.wasmPaths=new URL('../vendor/ort/',self.location.href).href;
  ort.env.wasm.numThreads=1;
  ort.env.wasm.proxy=false;
  manifest=info;
  if(!/^[a-f0-9]{64}$/.test(info.sha256||''))throw Error('전용 모델 해시 누락');
  const response=await fetch(new URL('../models/object-model.onnx',self.location.href),{cache:'no-cache'});
  if(!response.ok)throw Error('전용 모델 파일 누락');
  const weights=await response.arrayBuffer();
  const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',weights)),x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==info.sha256)throw Error('전용 모델 파일 검증 실패');
  ortSession=await ort.InferenceSession.create(weights,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
  if(!ortSession.inputNames.includes('input')||!ortSession.outputNames.includes('logits'))throw Error('ONNX 입력/출력 형식 불일치');
  await classify(new Float32Array(3*224*224).buffer);
  return true;
 })();
 try{return await loadPromise}catch(e){ortSession=null;throw e}finally{loadPromise=null}
}
async function classify(buffer){
 if(!ortSession)throw Error('전용 ONNX 모델 미준비');
 const data=new Float32Array(buffer);
 if(data.length!==3*224*224)throw Error('전용 모델 입력 픽셀 수 불일치');
 const output=await ortSession.run({input:new ort.Tensor('float32',data,[1,3,224,224])});
 const logits=output.logits?.data;if(!logits||logits.length!==manifest.labels.length||!Array.from(logits).every(Number.isFinite))throw Error('출력 라벨 개수 불일치');
 const m=Math.max(...logits),exps=Array.from(logits,x=>Math.exp(x-m)),sum=exps.reduce((a,b)=>a+b,0);
 return manifest.labels.map((key,i)=>({key,score:exps[i]/sum})).sort((a,b)=>b.score-a.score).slice(0,5);
}
self.onmessage=e=>{
 const {id,type,manifest:incoming,buffer}=e.data||{};
 if(!Number.isInteger(id))return;
 if(type==='init'){load(incoming).then(result=>reply(id,'result',{result}),err=>reply(id,'error',{message:String(err.message||err)}));return;}
 if(type==='detect'){
  const task=serial.catch(()=>{}).then(()=>classify(buffer));serial=task;
  task.then(result=>reply(id,'result',{result}),err=>reply(id,'error',{message:String(err.message||err)}));
 }
};
