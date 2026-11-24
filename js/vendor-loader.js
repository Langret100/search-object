/* Loads TF.js model libraries on demand, with ordered CDN fallbacks.
 * All script sources below are public third-party hosts; no camera data is uploaded by this loader.
 * Model weights also load separately from the model authors' hosting infrastructure. */
(function(){
 'use strict';
 const libs={
  tf:{global:'tf',version:'@tensorflow/tfjs@4.22.0/dist/tf.min.js'},
  coco:{global:'cocoSsd',version:'@tensorflow-models/coco-ssd@2.2.3/dist/coco-ssd.min.js'},
  mobilenet:{global:'mobilenet',version:'@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js'},
  knn:{global:'knnClassifier',version:'@tensorflow-models/knn-classifier@1.2.6/dist/knn-classifier.min.js'}
 };
 const inFlight=new Map();
 const hosts=['https://cdn.jsdelivr.net/npm/','https://unpkg.com/'];
 function script(url,timeout=18000){
  return new Promise((resolve,reject)=>{
   const tag=document.createElement('script');let done=false;
   const timer=setTimeout(()=>finish(new Error('CDN 연결 시간 초과')),timeout);
   function finish(error){if(done)return;done=true;clearTimeout(timer);tag.onload=tag.onerror=null;if(error){tag.remove();reject(error)}else resolve()}
   tag.async=true;tag.crossOrigin='anonymous';tag.src=url;
   tag.onload=()=>finish();tag.onerror=()=>finish(new Error('스크립트 다운로드 실패'));
   document.head.appendChild(tag);
  });
 }
 async function requireLib(name,onProgress){
  if(!libs[name])throw Error('알 수 없는 AI 라이브러리: '+name);
  const spec=libs[name];if(window[spec.global])return window[spec.global];
  if(inFlight.has(name))return inFlight.get(name);
  const task=(async()=>{
   const errors=[];
   for(const host of hosts){
    const address=host+spec.version;
    onProgress?.(`${name} 다운로드 중 (${new URL(host).hostname})`);
    try{await script(address);if(window[spec.global])return window[spec.global];throw Error('스크립트 실행 결과 없음')}
    catch(e){errors.push(`${new URL(host).hostname}: ${e.message}`)}
   }
   throw Error(`${name} 다운로드 실패. 인터넷/학교망 CDN 차단 확인. ${errors.join(' / ')}`);
  })();
  inFlight.set(name,task);
  try{return await task}finally{inFlight.delete(name)}
 }
 window.DetectiveVendors={
  requireLib,
  async ensure(names,onProgress){for(const n of names)await requireLib(n,onProgress)},
  status(){return Object.fromEntries(Object.entries(libs).map(([k,s])=>[k,Boolean(window[s.global])]))}
 };
})();
