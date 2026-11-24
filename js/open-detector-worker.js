/* Lightweight zero-shot IMAGE CLASSIFIER in a Web Worker.
 * This is not an object detector: classify the item placed in the camera centre.
 * MobileCLIP-S0 uses q8 text+vision weights (~55 MB total), not 155-200 MB DINO/OWL.
 * Text labels are embedded ONCE, then the text model is disposed. Video frames only
 * pass through the much smaller vision encoder. Nothing is sent to a remote inference API.
 */
'use strict';
const LIBRARY='https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';
const MODEL='Xenova/mobileclip_s0';
const LABELS=[
 {key:'pencil',phrases:['a photo of a pencil','a photo of a wooden pencil']},
 {key:'pen',phrases:['a photo of a ballpoint pen','a photo of a writing pen']},
 {key:'eraser',phrases:['a photo of a rubber eraser','a photo of an eraser for pencils']},
 {key:'a4paper',phrases:['a photo of a blank sheet of white paper','a photo of an A4 white paper sheet']},
 {key:'box',phrases:['a photo of a cardboard box','a photo of an empty paper carton']},
 {key:'jump_rope',phrases:['a photo of a skipping rope','a photo of a jump rope']},
 {key:'basket',phrases:['a photo of a plastic basket','a photo of a shopping basket']},
 {key:'shuttlecock',phrases:['a photo of a badminton shuttlecock','a photo of a badminton birdie']},
 {key:'traffic_cone',phrases:['a photo of an orange traffic cone','a photo of a road safety cone']},
 {key:'__none__',phrases:['a photo of an empty desk','a photo of a bare wall','a photo of a human hand','a photo of a human face','a photo of a classroom interior','a photo of a book on a table']}
];
let visionModel=null,processor=null,embeddings=null,RawImageReader=null,loading=null,busy=Promise.resolve();
const errorMessage=e=>String(e?.message||e||'알 수 없는 오류').slice(0,900);
function reply(id,type,payload={}){self.postMessage({id,type,...payload})}
const norm=v=>{let length=Math.hypot(...v)||1;return Array.from(v,x=>x/length)};
function dot(a,b){let sum=0;for(let i=0;i<a.length;i++)sum+=a[i]*b[i];return sum}
function classifyVector(vec){
 const img=norm(vec);
 const scores=embeddings.map(({key,vector})=>({key,sim:dot(img,vector)}));
 const grouped=new Map();for(const {key,sim} of scores){grouped.set(key,Math.max(sim,grouped.get(key)??-1))}
 const ranked=[...grouped].map(([key,sim])=>({key,sim})).sort((a,b)=>b.sim-a.sim);
 // Relative softmax is a ranking strength, NOT a calibrated probability of correctness.
 const exp=ranked.map(r=>Math.exp(30*(r.sim-ranked[0].sim))),denom=exp.reduce((a,b)=>a+b,0);
 return ranked.map((r,i)=>({label:r.key,score:exp[i]/denom,similarity:r.sim}));
}
async function initialize(modelId){
 if(modelId!==MODEL)throw Error('허용되지 않은 경량 AI 모델입니다.');
 if(visionModel&&embeddings)return true;
 if(loading)return loading;
 loading=(async()=>{
  self.postMessage({type:'progress',status:'initiate',file:'MobileCLIP-S0 q8'});
  const {AutoTokenizer,CLIPTextModelWithProjection,AutoProcessor,CLIPVisionModelWithProjection,RawImage,env}=await import(LIBRARY);
  if(!AutoTokenizer||!CLIPTextModelWithProjection||!CLIPVisionModelWithProjection||!RawImage)throw Error('Transformers.js 모델 구성요소를 가져오지 못했습니다.');
  RawImageReader=RawImage;
  env.useBrowserCache=true;
  // Optional SAME-ORIGIN mirror: upload models/Xenova/mobileclip_s0/* to GitHub Pages.
  // If present, avoid all Hugging Face weight requests on school networks.
  const localRoot=new URL('../models/',self.location.href).href;
  let locallyHosted=false;
  try{
   const res=await fetch(new URL('Xenova/mobileclip_s0/config.json',localRoot),{method:'HEAD',cache:'force-cache'});
   locallyHosted=res.ok&&/json/i.test(res.headers.get('content-type')||'');
  }catch{}
  env.allowLocalModels=locallyHosted;env.allowRemoteModels=!locallyHosted;
  if(locallyHosted)env.localModelPath=localRoot;
  self.postMessage({type:'progress',status:'host',file:locallyHosted?'GitHub Pages 자체 모델 파일 사용':'Hugging Face 모델 저장소 사용'});
  // GitHub Pages is not cross-origin isolated. Keep ONNX on a single WASM thread.
  if(env.backends?.onnx?.wasm){env.backends.onnx.wasm.numThreads=1;env.backends.onnx.wasm.wasmPaths=`${LIBRARY}/dist/`}
  let lastProgress=0;
  const progress_callback=e=>{let t=Date.now();if(e.status==='progress'&&t-lastProgress<350)return;lastProgress=t;self.postMessage({type:'progress',status:e.status,file:e.file||'',loaded:e.loaded,total:e.total})};
  // Encode all object/negative prompts once; no text inference in the camera loop.
  const tokenizer=await AutoTokenizer.from_pretrained(MODEL,{progress_callback});
  const textModel=await CLIPTextModelWithProjection.from_pretrained(MODEL,{device:'wasm',dtype:'q8',progress_callback});
  let textEmbeds;
  try{
   const prompts=LABELS.flatMap(v=>v.phrases);
   const textInputs=tokenizer(prompts,{padding:'max_length',truncation:true});
   const outputs=await textModel(textInputs);
   textEmbeds=outputs.text_embeds.tolist();
   if(textEmbeds.length!==prompts.length)throw Error('문장 특징 벡터의 개수가 맞지 않습니다.');
   embeddings=[];let i=0;
   for(const o of LABELS)for(const phrase of o.phrases)embeddings.push({key:o.key,vector:norm(textEmbeds[i++])});
  } finally {await textModel.dispose?.()}
  processor=await AutoProcessor.from_pretrained(MODEL,{progress_callback});
  visionModel=await CLIPVisionModelWithProjection.from_pretrained(MODEL,{device:'wasm',dtype:'q8',progress_callback});
  self.postMessage({type:'progress',status:'ready',file:'MobileCLIP-S0'});
  return true;
 })();
 try{return await loading}catch(e){embeddings=null;visionModel=null;processor=null;throw e}finally{loading=null}
}
self.addEventListener('message',event=>{
 const d=event.data||{},id=d.id;
 if(!Number.isSafeInteger(id))return;
 if(d.type==='init'){initialize(d.modelId).then(()=>reply(id,'result',{result:true}),e=>reply(id,'error',{message:errorMessage(e)}));return}
 if(d.type==='detect'){
  const task=busy.catch(()=>{}).then(async()=>{
   if(!visionModel||!embeddings||!processor)throw Error('경량 AI 모델 초기화가 완료되지 않았습니다.');
   if(typeof d.image!=='string'||!/^data:image\/(jpeg|png|webp);base64,/.test(d.image))throw Error('카메라 영상 형식이 올바르지 않습니다.');
   // The same set of labels is scored for every mission; avoid forced positives.
   const input=await processor(await RawImageReader.read(d.image));
   const output=await visionModel(input);
   const vec=output.image_embeds.tolist()?.[0];
   if(!Array.isArray(vec)||vec.length!==embeddings[0].vector.length)throw Error('화면 특징 벡터 생성에 실패했습니다.');
   return classifyVector(vec);
  });
  busy=task;
  task.then(result=>reply(id,'result',{result}),e=>reply(id,'error',{message:errorMessage(e)}));
 }
});
