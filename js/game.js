'use strict';
/* AI 사물 탐정: GitHub Pages static application. No user camera uploads. */
const $=id=>document.getElementById(id);
const LOCAL_ONLY=window.DETECTIVE_LOCAL_ONLY===true;
// Private browsing / embedded previews may disallow localStorage access.
const storage=(()=>{try{const s=window.localStorage;const k='__ai_detective_storage_test__';s.setItem(k,'1');s.removeItem(k);return s}catch(e){console.warn('Local storage unavailable; settings are temporary',e);const m=new Map();return {getItem:k=>m.get(k)||null,setItem:(k,v)=>m.set(k,String(v)),removeItem:k=>m.delete(k)}}})();
const COCO=[
['sports ball','공(축구공·농구공)','축구공이나 농구공 같은 공을 보여줘!','⚽','체육'],['cup','컵(머그컵 포함)','컵이나 머그컵을 찾아봐! AI는 컵 종류를 구분하지 않아.','☕','생활'],['bottle','물병·물통','물을 담는 병이나 물통을 찾아봐!','🧴','생활'],['book','책','읽을 수 있는 책을 찾아봐!','📕','학습'],['scissors','가위','종이를 자를 때 사용하는 가위야!','✂️','학습'],['cell phone','휴대폰','전화도 할 수 있는 휴대폰을 찾아봐!','📱','전자'],['laptop','노트북','접고 펼치는 노트북 컴퓨터를 찾아봐!','💻','전자'],['keyboard','키보드','글자를 입력하는 키보드야!','⌨️','전자'],['mouse','마우스','컴퓨터를 조작할 때 사용하는 마우스!','🖱️','전자'],['remote','리모컨','멀리서 기기를 조작하는 리모컨이야!','🎛️','전자'],['spoon','숟가락','음식을 떠먹을 때 사용하는 도구야!','🥄','생활'],['bowl','그릇','음식을 담는 오목한 그릇을 찾아봐!','🥣','생활'],['toothbrush','칫솔','이를 닦을 때 사용하는 칫솔이야!','🪥','생활'],['clock','시계','시간을 알려주는 시계를 찾아봐!','⏰','생활'],['backpack','배낭','어깨에 메는 가방을 찾아봐!','🎒','생활'],['handbag','손가방','손에 들고 다니는 가방을 찾아봐!','👜','생활'],['umbrella','우산','비가 올 때 쓰는 우산이야!','☂️','생활'],['teddy bear','곰 인형','귀여운 곰 인형을 찾아봐!','🧸','장난감'],['potted plant','화분','식물이 자라는 화분이야!','🪴','생활'],['vase','꽃병','꽃을 꽂아 두는 꽃병을 찾아봐!','🏺','생활'],['apple','사과','빨갛거나 초록색인 사과를 찾아봐!','🍎','음식'],['banana','바나나','길고 노란 바나나를 찾아봐!','🍌','음식'],['orange','오렌지','둥근 오렌지를 찾아봐!','🍊','음식'],['carrot','당근','주황색 채소 당근이야!','🥕','음식'],['broccoli','브로콜리','초록색 나무처럼 생긴 브로콜리야!','🥦','음식'],['donut','도넛','가운데 구멍이 있는 달콤한 도넛!','🍩','음식'],['sandwich','샌드위치','빵 사이에 재료를 넣은 샌드위치야!','🥪','음식'],['pizza','피자','치즈가 올라간 피자야!','🍕','음식'],['cake','케이크','달콤한 케이크를 찾아봐!','🍰','음식']
].map(a=>({key:a[0],name:a[1],desc:a[2],emoji:a[3],group:a[4],kind:'coco'}));
// Extra categories currently use full-frame MobileCLIP similarity comparison; replacement training pipeline is in /training.
const OPEN=[
 ['pencil','연필','연필을 찾아서 보여줘!','✏️','a wooden pencil'],
 ['pen','볼펜','볼펜을 찾아서 보여줘!','🖊️','a ballpoint pen'],
 ['eraser','지우개','지우개를 찾아서 보여줘!','🩹','a pencil eraser'],
 ['a4paper','A4용지','흰 종이를 찾아서 보여줘! (AI는 A4 규격까지 측정하지 못해.)','📄','a sheet of white paper'],
 ['box','박스','종이 상자를 찾아서 보여줘!','📦','a cardboard box'],
 ['jump_rope','줄넘기','줄넘기 줄을 찾아서 보여줘!','➰','a skipping rope'],
 ['basket','바구니','바구니를 찾아서 보여줘!','🧺','a plastic basket'],
 ['shuttlecock','배드민턴 공','셔틀콕을 찾아서 보여줘!','🏸','a badminton shuttlecock'],
 ['traffic_cone','라바콘','주황색 삼각형 안전콘을 찾아서 보여줘!','🔺','an orange traffic cone'],
].map(a=>({key:a[0],name:a[1],desc:a[2],emoji:a[3],prompt:a[4],group:'확장 탐지 (추가 학습 없음)',kind:'open'}));
const CLIP=OPEN; // Preserve older selection paths while removing CLIP inference.
const RECOMMENDED=['pencil','pen','eraser','a4paper','bottle','box','jump_rope','basket','sports ball','shuttlecock','traffic_cone'];
const DEFAULT={mode:'general',clipMode:'on',openModel:'mobileclip',duration:60,hold:0.65,rerollCount:1,tts:true,sfx:true,sfxVolume:0.68,autoReturn:true,enabled:RECOMMENDED};
const OPEN_DETECTOR_MODELS={mobileclip:'Xenova/mobileclip_s0'};
const OPEN_MODEL_LABELS={mobileclip:'MobileCLIP-S0 (경량)'};
const MIN_SAMPLES=5; // require several distinct views and a negative/background class
// When a validated, locally bundled ONNX model exists, it takes precedence over all legacy AI engines.
// No remote model host is accessed by this path. The ZIP does NOT fabricate missing weights.
let localReady=false,localLabels=new Set(),localPreload=null,localWorker=null,localJobs=new Map(),localSeq=0;
let localModelManifest=null,localInputCanvas=null;
function localRequest(type,data={},ms=35000){
 return new Promise((resolve,reject)=>{
  if(!localWorker){reject(Error('전용 모델 Worker가 준비되지 않았습니다'));return;}
  const id=++localSeq,timer=setTimeout(()=>{localJobs.delete(id);reject(Error('전용 모델 응답 시간 초과'))},ms);
  localJobs.set(id,{resolve,reject,timer});
  try{localWorker.postMessage({id,type,...data},data.buffer?[data.buffer]:[])}
  catch(e){clearTimeout(timer);localJobs.delete(id);reject(e)}
 });
}
async function initializeLocalModel(){
 try{
  const response=await fetch('models/model-manifest.json',{cache:'no-cache'});
  if(!response.ok){if(LOCAL_ONLY)throw Error('전용 모델 명세가 없습니다');return false;}
  const manifest=await response.json();
  if(manifest.format!=='onnx-mobilenet-v3-small-v1'||!Array.isArray(manifest.labels)||manifest.labels.length<3)throw Error('전용 AI 모델 목록이 잘못되었습니다');
  if(new Set(manifest.labels).size!==manifest.labels.length||manifest.input_size!==224||!Number.isFinite(manifest.min_score)||manifest.min_score<.68||manifest.min_score>1||!Number.isFinite(manifest.min_margin)||manifest.min_margin<.15||manifest.min_margin>1)throw Error('전용 AI 판정 규격이 잘못되었습니다');
  if(!manifest.labels.includes('__background__'))throw Error('전용 AI에 배경 거부 클래스가 없습니다');
  if(LOCAL_ONLY&&manifest.labels.length!==COCO.length+OPEN.length+1)throw Error('38종 전용 모델이 필요합니다');
  if(!manifest.labels.every(k=>k==='__background__'||[...COCO,...OPEN].some(o=>o.key===k)))throw Error('등록되지 않은 물품의 학습 데이터입니다');
  const worker=new Worker('js/local-object-worker.js?v=1',{name:'AI 사물탐정 전용 모델'});
  localWorker=worker;
  worker.onmessage=ev=>{const {id,type,result,message}=ev.data||{};const job=localJobs.get(id);if(!job)return;localJobs.delete(id);clearTimeout(job.timer);if(type==='error')job.reject(Error(message));else job.resolve(result)};
  worker.onerror=e=>{for(const job of localJobs.values()){clearTimeout(job.timer);job.reject(Error('로컬 모델 실행 실패: '+(e.message||'')))}localJobs.clear()};
  await localRequest('init',{manifest},90000);
  localModelManifest=manifest;localLabels=new Set(manifest.labels);localReady=true;
  showOpenStatus(`전용 AI 준비 완료 · 물품 ${manifest.labels.length-1}종 · 모델 재사용`,'ready');
  $('engineStatus').textContent='내장 학습 AI 준비 완료';
  if(phase==='selection'&&chosen&&!localLabels.has(chosen.key)){chosen=null;phase='idle';renderMission();updateButtons()}
  return true;
 }catch(e){console.warn('전용 모델 사용 불가:',e);if(LOCAL_ONLY)showOpenStatus('전용 AI 준비 실패 · 파일 확인 필요','error');localReady=false;localWorker?.terminate();localWorker=null;for(const job of localJobs.values()){clearTimeout(job.timer);job.reject(e)}localJobs.clear();return false}
}
function captureLocalFrame(video){
 const size=224;
 if(!localInputCanvas){localInputCanvas=document.createElement('canvas');localInputCanvas.width=size;localInputCanvas.height=size}
 const c=localInputCanvas,ctx=c.getContext('2d',{willReadFrequently:true});
 // Letterbox rather than centre crop: thin jump ropes and long stationery remain visible.
 ctx.fillStyle='#808080';ctx.fillRect(0,0,size,size);
 const w=video.videoWidth||video.width,h=video.videoHeight||video.height,scale=Math.min(size/w,size/h),dw=w*scale,dh=h*scale;
 ctx.drawImage(video,(size-dw)/2,(size-dh)/2,dw,dh);
 const rgba=ctx.getImageData(0,0,size,size).data;
 const out=new Float32Array(3*size*size),plane=size*size;
 for(let i=0,j=0;i<plane;i++,j+=4){out[i]=(rgba[j]/255-.485)/.229;out[i+plane]=(rgba[j+1]/255-.456)/.224;out[i+2*plane]=(rgba[j+2]/255-.406)/.225}
 return out.buffer;
}
async function detectLocal(video){return localRequest('detect',{buffer:captureLocalFrame(video)},18000)}
const PREDICTION_STALE_MS=1600;
const OPEN_CONFIRM_MAX_GAP_MS=4500; // two DISTINCT analyzed frames of the same object
const CAMERA_FRAME_TIMEOUT_MS=6500; // Stop a mission when a camera feed freezes
const INFERENCE_TIMEOUT_MS=18000; // Recover from a stuck inference; model initialization happens before timer start

let config={...DEFAULT,...safeParse(storage.getItem('detective_config'),{})};
// Persisted user data is untrusted. A malformed old config must not crash the game.
if(!Array.isArray(config.enabled))config.enabled=[...RECOMMENDED];
config.enabled=config.enabled.filter(x=>typeof x==='string');
if(!['general','custom','hybrid'].includes(config.mode))config.mode='general';
if(!['on','off'].includes(config.clipMode))config.clipMode='on';
config.sfx=config.sfx!==false;config.sfxVolume=Math.min(1,Math.max(0,Number.isFinite(+config.sfxVolume)?+config.sfxVolume:.68));
if(config.catalogVersion!==31){config.enabled=[...RECOMMENDED];config.clipMode='on';config.catalogVersion=31;saveConfig();}
if(!Object.keys(OPEN_DETECTOR_MODELS).includes(config.openModel))config.openModel='mobileclip';
config.duration=Number.isFinite(+config.duration)?Math.min(180,Math.max(15,+config.duration)):60;
config.hold=Number.isFinite(+config.hold)?Math.min(3,Math.max(.2,+config.hold)):.65;
config.rerollCount=Number.isFinite(+config.rerollCount)?Math.min(3,Math.max(0,Math.floor(+config.rerollCount))):1;
const MODEL_READY_TIMEOUT_MS=90000; // first-time model download only
function withDeadline(task,ms,label){let timer;return Promise.race([task,new Promise((_,reject)=>{timer=setTimeout(()=>reject(Error(label+' 시간 초과. 인터넷 및 외부 모델 서버 접속을 확인하세요.')),ms)})]).finally(()=>clearTimeout(timer))}

let classes=safeParse(storage.getItem('detective_classes'),[]);if(!Array.isArray(classes))classes=[];classes=classes.filter(c=>c&&typeof c.id==='string'&&typeof c.name==='string');
let phase='idle',chosen=null,previous=null,remainingRerolls=1,stream=null,camReady=false,model=null,modelPromise=null,mobile=null,knn=null,customPromise=null,clip=null,clipPromise=null;
let lastPred=[],lastCustom=[],lastClip=[],lastLocal=[],lastInferenceAt=0,detectorReady=false,timerDeadline=0,timerTask=null,detectTask=null,holdStart=0,holdKey='',lastSpeech=0,drawFrame=0,returnTimeout=null,spinTimeout=null,aiBusy=false,loopActive=false,animToken=0,frameCounter=0,confirmedFrames=0;
let spinTimers=[],spinVersion=0,robotSpeechTimer=null,cameraSession=0,inferenceFailures=0,lastVideoTime=-1,lastPredictedVideoTime=-1,lastFrameHealthyAt=0,activeInferenceSince=0,latestPredictionId=0,lastTrainingFrameAt=0;
let cameraPromise=null, inferenceJob=null, inferenceStuck=false, modelUnderlying=null;
let openWorker=null,openPending=new Map(),openSeq=0,openRequestedId='',openFailure=null,openState='idle',openGeneration=0;
let lastPositiveAt=0; // A long gap cannot count as continuous recognition
let renderFps=0,frameSample=0,frameSampleAt=performance.now(),lastOpenInferenceMs=0,openPredictionCount=0;
function recordRenderFps(t){frameSample++;if(t-frameSampleAt>=1200){renderFps=Math.round(frameSample*1000/(t-frameSampleAt));frameSample=0;frameSampleAt=t}requestAnimationFrame(recordRenderFps)}
requestAnimationFrame(recordRenderFps);
let timeWarning30=false,timeWarning10=false;
const isLiveCamera=()=>Boolean(camReady&&stream?.getVideoTracks()?.some(t=>t.readyState==='live')&&$('video').readyState>=2);
function invalidatePredictions(){++latestPredictionId;lastPred=[];lastCustom=[];lastClip=[];lastLocal=[];lastInferenceAt=0;holdStart=0;holdKey='';confirmedFrames=0;lastPositiveAt=0;}

// Locally bundled original sound effects: no CDN, autoplay only after a player gesture.
const SOUND_NAMES=['tap','spin_start','spin_tick','reroll','select','camera','recognize','warning30','warning10','error','fail','success','start'];
const soundBank=new Map();
for(const name of SOUND_NAMES){const a=new Audio(`assets/sfx/${name}.wav`);a.preload='auto';soundBank.set(name,a)}
let soundUnlocked=false;let lastSfxAt=0;
function unlockSound(){if(soundUnlocked)return;soundUnlocked=true;const a=soundBank.get('tap');if(a){const wasMuted=a.muted;a.muted=true;try{const p=a.play();if(p&&p.then)p.then(()=>{a.pause();a.currentTime=0;a.muted=wasMuted}).catch(()=>{a.muted=wasMuted})}catch{a.muted=wasMuted}}}
function playSfx(name,volume=1){
 if(!config.sfx)return;
 const source=soundBank.get(name);if(!source)return;
 try{const audio=source.cloneNode();audio.volume=Math.min(1,Math.max(0,(Number(config.sfxVolume)||0)*volume));const p=audio.play();if(p?.catch)p.catch(()=>{});audio.onended=()=>{audio.src='';audio.load()}}catch(e){console.warn('SFX',name,e)}
}
function refreshSoundUI(){$('soundBtn').textContent=(config.tts||config.sfx)?'🔊':'🔇';$('soundBtn').setAttribute('aria-label',(config.tts||config.sfx)?'음소거':'소리 켜기')}

const sleep=t=>new Promise(r=>setTimeout(r,t));
function safeParse(s,d){try{return JSON.parse(s)||d}catch{return d}}
function saveConfig(){config.enabled=[...new Set(Array.isArray(config.enabled)?config.enabled:[])];storage.setItem('detective_config',JSON.stringify(config));}
function setSpeech(s,force=false){
 $('speech').textContent=s;
 if(robotSpeechTimer)clearTimeout(robotSpeechTimer);
 if(config.tts&&(force||Date.now()-lastSpeech>2500)&&'speechSynthesis'in window){
  try{speechSynthesis.cancel();const u=new SpeechSynthesisUtterance(s);u.lang='ko-KR';u.rate=1.03;u.pitch=1.12;const voices=speechSynthesis.getVoices();u.voice=voices.find(v=>v.lang==='ko-KR')||null;
   const previous=$('robot').dataset.state;
   if(phase!=='playing'&&phase!=='spinning'&&previous!=='success'&&previous!=='fail')robot('talk');
   u.onend=()=>{if(['selection','idle'].includes(phase)&&$('robot').dataset.state==='talk')robot('idle')};
   speechSynthesis.speak(u);lastSpeech=Date.now();
  }catch(e){console.warn('TTS',e)}
 }
}
function robot(state){
 const el=$('robot');
 const allowed=new Set(['idle','talk','think','search','success','fail']);
 const pose=allowed.has(state)?state:'idle';
 el.dataset.state=pose;
 // WebP atlas: switch CSS background-position, no individual downloads.
 el.dataset.state=pose;
}

const ITEM_ICON_ASSETS={traffic_cone:'assets/traffic_cone.svg',jump_rope:'assets/jump_rope.svg',shuttlecock:'assets/shuttlecock.svg',eraser:'assets/eraser.svg',bottle:'assets/bottle.svg',remote:'assets/remote.svg'};
function reelGraphic(o, mount){
 if(!mount)return;
 mount.replaceChildren();
 if(!o){mount.textContent='✦';return}
 const icon=ITEM_ICON_ASSETS[o.key];
 if(icon){const pic=document.createElement('img');pic.src=icon;pic.alt=o.name;pic.decoding='async';mount.append(pic);return;}
 mount.textContent=o.emoji||'🔎';
}
function renderMission(){
 const o=chosen,p=pool(),idx=p.findIndex(x=>x.key===o?.key),n=p.length;
 reelGraphic(idx>=0?p[(idx-1+n)%n]:null,$('slotPrev'));
 reelGraphic(idx>=0?p[(idx+1)%n]:null,$('slotNext'));
 $('targetName').textContent=o?o.name:'미션을 뽑아 주세요';
 $('targetDesc').textContent=o?o.desc:'로봇이 물건을 골라 줄 거야!';
 reelGraphic(o,$('targetEmoji'));
 $('itemName').textContent=o?o.name:'미션 준비!';
 reelGraphic(o,$('itemGraphic'));
}

function pool(){
  if(localReady)return [...COCO,...OPEN].filter(o=>config.enabled.includes(o.key)&&localLabels.has(o.key));
  const regular=config.mode==='custom'?[]:COCO.filter(o=>config.enabled.includes(o.key));
  const extensions=config.mode==='custom'||config.clipMode!=='on'||openFailure?[]:OPEN.filter(o=>config.enabled.includes(o.key));
  const backgrounds=knn?.getClassExampleCount()?.['__background__']||0;
  const trained=config.mode==='general'?[]:classes.filter(c=>c.enabled!==false&&(knn?.getClassExampleCount()?.[c.id]||0)>=MIN_SAMPLES&&backgrounds>=MIN_SAMPLES).map(c=>({...c,key:c.id,kind:'custom'}));
  return [...regular,...extensions,...trained];
}
function updateButtons(){for(let id of ['startBtn','rerollBtn','confirmBtn','resetBtn'])$(id).classList.add('hidden');if(phase==='idle'||phase==='finished')$('startBtn').classList.remove('hidden');if(phase==='selection'){$('confirmBtn').classList.remove('hidden');if(remainingRerolls>0&&pool().length>1){$('rerollBtn').classList.remove('hidden');$('rerollBtn').textContent=`↻ 다시? (${remainingRerolls}/${config.rerollCount})`}}if(phase==='playing')$('resetBtn').classList.remove('hidden')}
function pick(exclude){let choices=pool().filter(o=>o.key!==exclude);if(!choices.length)choices=pool();if(!choices.length){setSpeech('관리자 설정에서 미션 물품을 골라줘!',true);$('settings').showModal();switchTab('objects');return}chosen=choices[Math.floor(Math.random()*choices.length)];renderMission();}
function cancelSpin(){
 ++spinVersion;
 for(const id of spinTimers)clearTimeout(id);
 spinTimers=[];
 if(spinTimeout)clearTimeout(spinTimeout);
 spinTimeout=null;
 $('slot').classList.remove('spinning');
 $('slot').classList.remove('slot-landed');
}
function showReelItem(o){
 const p=pool(),n=p.length,index=p.findIndex(x=>x.key===o.key);
 reelGraphic(p[(index-1+n)%n],$('slotPrev'));
 reelGraphic(o,$('itemGraphic'));
 reelGraphic(p[(index+1)%n],$('slotNext'));
 $('itemName').textContent=o.name;
}

function spin(exclude){
 cancelSpin();const candidates=pool().filter(o=>o.key!==exclude);
 if(!candidates.length){phase='idle';updateButtons();setSpeech('설정에서 사용할 물건을 두 개 이상 선택해 줘!',true);return}
 const target=candidates[Math.floor(Math.random()*candidates.length)];
 const catalogue=pool();const ticket=spinVersion;
 phase='spinning';playSfx(exclude?'reroll':'spin_start');updateButtons();robot('search');setSpeech('이번 미션을 뽑아 볼까?',true);
 $('slot').classList.add('spinning');
 // A contiguous reel advances in catalogue order, not unrelated random jumps.
 // Full rows are separate DOM elements; the bezel and camera never get rasterized into the reel.
 let elapsed=0;const totalSteps=19;
 const targetIndex=catalogue.findIndex(o=>o.key===target.key);
 const initialIndex=(targetIndex-totalSteps+1+catalogue.length*10)%catalogue.length;
 for(let i=0;i<totalSteps;i++){
  const t=i/(totalSteps-1);
  const delay=43+Math.round(245*t*t*t);
  elapsed+=delay;
  const id=setTimeout(()=>{
   if(ticket!==spinVersion||phase!=='spinning')return;
   const item=catalogue[(initialIndex+i)%catalogue.length];
   showReelItem(item);if(i%2===0)playSfx('spin_tick',.35);
   const win=$('slot-window');
   if(win&&win.animate&&!matchMedia('(prefers-reduced-motion: reduce)').matches){
    win.getAnimations().forEach(a=>a.cancel());
    win.animate([
      {transform:'translateY(-14px)',opacity:.77},
      {transform:'translateY(0)',opacity:1}
    ],{duration:Math.min(235,delay+65),easing:'cubic-bezier(.2,.75,.3,1)'});
   }
  },elapsed);
  spinTimers.push(id);
 }

 spinTimeout=setTimeout(()=>{
  if(ticket!==spinVersion||phase!=='spinning')return;
  if(!pool().some(o=>o.key===target.key)){phase='idle';updateButtons();$('feedback').textContent='확장 AI 연결에 실패해 해당 물건은 이번 미션에서 제외됐어요. 다시 뽑아 주세요.';return;}
  chosen=target;renderMission();phase='selection';playSfx('select');
  $('slot').classList.remove('spinning');$('slot').classList.add('slot-landed');
  robot('talk');setSpeech(`이번 미션은 ${chosen.name}! ${chosen.desc}`,true);
  updateButtons();
 },elapsed+150);
}
async function startSelection(){
  if(localPreload){$('feedback').textContent='AI 준비 확인 중…';await localPreload;}
  if(!['idle','finished'].includes(phase))return;
  if(LOCAL_ONLY&&!localReady){$('feedback').textContent='전용 AI 파일이 누락되었거나 손상되었어요. 관리자에게 알려 주세요.';return;}
  stopRound();const selectionToken=spinVersion;
  phase='loading-selection';updateButtons();
  remainingRerolls=Math.max(0,config.rerollCount);chosen=null;renderMission();
  try{
    if(!localReady&&config.mode!=='general'&&!knn&&(classes.length || await idbGet('training').then(saved=>Boolean(saved?.dataset&&Object.keys(saved.dataset).length)).catch(()=>false))){
      $('feedback').textContent='직접 학습 자료를 불러오는 중…';
      try{await loadCustom()}catch(e){modelError('직접 학습 AI',e);if(config.mode==='custom')throw e}
    }
    // Expanded detector is already preloaded at startup; never load per reel item.
    if(selectionToken!==spinVersion||phase!=='loading-selection')return;
    phase='idle';spin(null);
  }catch(e){
    if(phase==='loading-selection'){
      phase='idle';updateButtons();
      $('feedback').textContent='AI 준비 실패: '+String(e?.message||e).slice(0,120);
    }
  }
}
async function confirm(){
  if(phase!=='selection'||!chosen)return;
  phase='preparing'; updateButtons();
  $('feedback').textContent='카메라를 준비하는 중이야…';
  if(inferenceStuck){$('feedback').textContent='이전 AI 분석이 아직 종료되지 않았어요. 잠시 뒤 다시 시도하거나 페이지를 새로고침해 주세요.';phase='selection';updateButtons();return}
  const ready=isLiveCamera()||await startCamera();
  if(phase!=='preparing')return;
  if(!ready){phase='selection';updateButtons();$('feedback').textContent='카메라 권한을 확인한 뒤 다시 시도해 줘!';return}
  // A network/model failure must not consume the child's allotted time.
  try {
    $('feedback').textContent='AI 모델을 준비하는 중이야…';
    if(localReady&&localLabels.has(chosen.key)){}
    else if(chosen.kind==='coco')await loadCoco();
    else if(chosen.kind==='custom'){await loadCustom();if(knn.getNumClasses()<2)throw Error('학습 자료가 부족합니다')}
    else if(chosen.kind==='open')await loadClip();
  } catch(e){modelError('미션 AI',e);phase='selection';updateButtons();$('feedback').textContent='AI 준비 실패: '+String(e.message||e).slice(0,110)+' · 설정의 AI 테스트에서 재시도해 줘.';return}
  if(phase!=='preparing')return;
  if(!isLiveCamera()){phase='selection';updateButtons();$('feedback').textContent='카메라가 준비되지 않았습니다. 다시 연결해 주세요.';return}
  phase='playing';playSfx('start');inferenceFailures=0;lastVideoTime=-1;lastPredictedVideoTime=-1;lastFrameHealthyAt=performance.now();holdStart=0;holdKey='';confirmedFrames=0;lastInferenceAt=0;lastPred=[];lastCustom=[];lastClip=[];lastLocal=[];lastPositiveAt=0;timerDeadline=performance.now()+config.duration*1000;
  robot('search');setSpeech(`찾을 물건은 ${chosen.name}! 카메라에 보여줘! 시작!`,true);
  $('feedback').textContent=`찾을 물건: ${chosen.name} · ${chosen.kind==='open'?'화면 중앙에 크게 보여줘!':'카메라에 보여줘!'}`;startTimer();updateButtons();
}
function startTimer(){stopTimer();timeWarning30=false;timeWarning10=false;function tick(){if(phase!=='playing')return;if(activeInferenceSince&&performance.now()-activeInferenceSince>INFERENCE_TIMEOUT_MS){pauseForInferenceError('AI 응답이 너무 오래 걸려 미션을 중단했어. 다시 시도해 줘.');return}const t=Math.max(0,Math.ceil((timerDeadline-performance.now())/1000));$('clock').textContent=`${String(Math.floor(t/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`;$('clock').parentElement.classList.toggle('urgent',t<=10);
 if(t>0&&t<=30&&!timeWarning30&&config.duration>35){timeWarning30=true;playSfx('warning30');$('feedback').textContent='⏳ 30초 남았어! 물건을 찾아줘!';setSpeech('30초 남았어! 주변을 잘 찾아봐!',true)}
 if(t>0&&t<=10&&!timeWarning10){timeWarning10=true;playSfx('warning10');$('feedback').textContent='🚨 10초 남았어! 물건을 보여줘!';setSpeech('이제 10초! 서둘러서 물건을 보여줘!',true)}
 if(t===0)finish(false)}tick();if(phase==='playing')timerTask=setInterval(tick,120)}
function stopTimer(){if(timerTask){clearInterval(timerTask);timerTask=null}}
function pauseForInferenceError(message){
 latestPredictionId++;
 if(phase!=='playing')return;
 stopTimer();playSfx('error');activeInferenceSince=0;phase='selection';lastPred=[];lastCustom=[];lastClip=[];
 holdStart=0;confirmedFrames=0;holdKey='';lastInferenceAt=0;
 $('progressFill').style.width='0%';
 $('feedback').textContent=message;
 $('engineStatus').textContent='인식 중단 · 재시도 가능';
 robot('think');updateButtons();
}

function stopRound(){invalidatePredictions();lastVideoTime=-1;lastPredictedVideoTime=-1;activeInferenceSince=0;lastInferenceAt=0;lastPred=[];lastCustom=[];lastClip=[];cancelSpin();stopTimer();if(returnTimeout){clearTimeout(returnTimeout);returnTimeout=null}holdStart=0;holdKey='';confirmedFrames=0;$('progressFill').style.width='0%'}
function confetti(){let el=$('confetti');el.replaceChildren();const colors=['#ffce19','#46e6ff','#fd51bd','#76fb86','#fff'];for(let i=0;i<125;i++){let p=document.createElement('div');p.className='particle';p.style.setProperty('--x',Math.random()*100+'%');p.style.setProperty('--dx',(Math.random()-.5)*300+'px');p.style.setProperty('--rot',Math.random()*180+'deg');p.style.setProperty('--dur',(1.6+Math.random()*2.4)+'s');p.style.setProperty('--color',colors[Math.floor(Math.random()*colors.length)]);p.style.animationDelay=(Math.random()*1.3)+'s';el.appendChild(p)}}
function finish(win){if(phase!=='playing')return;phase='finished';stopRound();robot(win?'success':'fail');playSfx(win?'success':'fail');$('resultMedal').textContent=win?'🏆':'⏰';$('resultTitle').textContent=win?'성공!':'시간 초과!';$('resultDetail').textContent=win?`${chosen.name} 찾기 성공! 멋진 탐정이야!`:'괜찮아! 다시 도전해 보자!';$('result').classList.remove('hidden');if(win)confetti();setSpeech(win?'성공! 정답이야! 정말 잘 찾았어!':'아쉽다! 시간이 끝났어. 괜찮아, 다음엔 꼭 찾아보자!',true);updateButtons();if(config.autoReturn)returnTimeout=setTimeout(resetGame,7000)}
function resetGame(){stopRound();$('detected').textContent='아직 없음';showCandidates([],'relative');$('engineStatus').textContent=camReady?'카메라 연결됨':'카메라 대기';phase='idle';chosen=null;remainingRerolls=config.rerollCount;renderMission();$('result').classList.add('hidden');$('clock').textContent=fmt(config.duration);$('clock').parentElement.classList.remove('urgent');$('feedback').textContent='미션을 뽑고 물건을 카메라에 보여줘!';robot('idle');setSpeech('다음 탐정 준비! 미션을 뽑아 보자!',true);updateButtons()}
function fmt(t){return `${String(Math.floor(t/60)).padStart(2,'0')}:${String(t%60).padStart(2,'0')}`}
function modelError(where,e){console.error(where,e);$('engineStatus').textContent=`${where} 불러오기 실패`;if(phase==='playing')$('feedback').textContent='AI 모델을 불러오지 못했어요. 인터넷 연결을 확인해 주세요.'}
async function loadCoco(){
 if(LOCAL_ONLY)throw Error('전용 AI 배포판에서는 외부 COCO 모델을 사용하지 않습니다');
 if(model)return model;
 if(!modelUnderlying){
  modelUnderlying=(async()=>{
   if(!window.DetectiveVendors)throw Error('AI 로더 파일이 없습니다. js/vendor-loader.js 포함 여부를 확인하세요');
   await DetectiveVendors.ensure(['tf','coco'],text=>$('engineStatus').textContent=text);
   await tf.ready();
   $('engineStatus').textContent='기본 AI 가중치 다운로드 중…';
   const result=await cocoSsd.load({base:'lite_mobilenet_v2'});
   if(!result||typeof result.detect!=='function')throw Error('모델 초기화 결과가 올바르지 않습니다');
   model=result;$('engineStatus').textContent='기본 AI 준비 완료';return result;
  })();
  modelUnderlying.then(()=>{if(!model)modelUnderlying=null},()=>{modelUnderlying=null});
 }
 // Timeout only limits UI waiting; it does not launch a duplicate model download.
 return withDeadline(modelUnderlying,MODEL_READY_TIMEOUT_MS,'기본 AI 모델 다운로드');
}
async function loadCustom(){
 if(LOCAL_ONLY)throw Error('전용 AI 배포판에서는 외부 직접 학습 모델을 사용하지 않습니다');
 if(mobile&&knn)return true;
 if(customPromise)return customPromise;
 customPromise=(async()=>{
  await DetectiveVendors.ensure(['tf','mobilenet','knn'],text=>$('engineStatus').textContent=text);
  await tf.ready();$('engineStatus').textContent='직접 학습 모델 준비 중…';
  const loadedMobile=await withDeadline(mobilenet.load({version:2,alpha:0.5}),MODEL_READY_TIMEOUT_MS,'직접 학습 모델 다운로드');
  const loadedKnn=knnClassifier.create();
  try{
   const backup=await idbGet('training');
   // IndexedDB stores the actual trained tensors AND their matching class metadata.
   // If localStorage was cleared, recover trained class names from this canonical backup.
   if(backup?.dataset && Array.isArray(backup.classes)){
    const persisted=backup.classes.filter(c=>c&&typeof c.id==='string'&&c.id.startsWith('custom_')&&typeof c.name==='string');
    const ids=new Set(persisted.map(c=>c.id));
    const untrained=classes.filter(c=>c&&typeof c.id==='string'&&!ids.has(c.id)&&Number(c.count||0)===0);
    classes=[...persisted,...untrained];
    persistClasses();
   }
   if(backup?.dataset){
    const dataset={};
    try{
     for(const [label,obj] of Object.entries(backup.dataset)){
      if(!obj||!Array.isArray(obj.shape)||!Array.isArray(obj.data)||obj.shape.length!==2||obj.shape[0]*obj.shape[1]!==obj.data.length||obj.data.length>5e6)throw Error('손상되거나 너무 큰 학습 데이터');
      dataset[label]=tf.tensor2d(obj.data,obj.shape);
     }
     if(Object.keys(dataset).length)loadedKnn.setClassifierDataset(dataset);
    }catch(e){for(const t of Object.values(dataset))t.dispose();throw e}
   }
   mobile=loadedMobile;knn=loadedKnn;
   const counts=loadedKnn.getClassExampleCount();
   for(const c of classes)c.count=counts[c.id]||0;
   $('engineStatus').textContent='직접 학습 AI 준비';return true;
  }catch(e){loadedKnn.dispose();loadedMobile.model?.dispose?.();throw e}
 })();
 try{return await customPromise}catch(e){customPromise=null;mobile=null;knn=null;throw e}
}
// All expanded objects share ONE model, initialized at page startup rather than per mission.
// Heavy ONNX downloads + inference run inside a same-origin Web Worker to keep the UI responsive.
let openLoadedModelId='';
function showOpenStatus(message,state='loading'){
 openState=state;
 const el=$('openStatus');
 if(el){el.textContent=message;el.dataset.status=state;el.title=message;}
}
function stopOpenWorker(){
 // Invalidate every previous load, even when the next load uses the same model ID.
 // Otherwise an old rejected init can destroy a freshly started Worker.
 ++openGeneration;
 const w=openWorker;openWorker=null;
 // Prevent obsolete responses from old workers and never leave a pending call unresolved.
 for(const rec of openPending.values()){clearTimeout(rec.timer);rec.reject(Error('확장 탐지 엔진이 종료되었습니다.'))}
 openPending.clear();w?.terminate();
 clip=null;clipPromise=null;openLoadedModelId='';openRequestedId='';
}
function startOpenWorker(){
 if(openWorker)return openWorker;
 if(typeof Worker==='undefined')throw Error('브라우저가 백그라운드 AI 실행을 지원하지 않습니다.');
 const w=new Worker('js/open-detector-worker.js?v=whalebook2',{type:'module',name:'AI 사물탐정 확장탐지'});
 openWorker=w;
 const rejectAll=error=>{
  if(openWorker!==w)return;
  for(const rec of openPending.values()){clearTimeout(rec.timer);rec.reject(error)}
  openPending.clear();
 };
 w.addEventListener('message',event=>{
  if(openWorker!==w)return;
  const data=event.data||{};
  if(data.type==='progress'){
   const pct=Number.isFinite(data.loaded)&&Number.isFinite(data.total)&&data.total>0?` ${Math.min(100,Math.round(data.loaded/data.total*100))}%`:'';
   const stateText=data.status==='host'?data.file:['download','progress'].includes(data.status)?`모델 가중치 다운로드${pct}`:data.status==='initiate'?'모델 파일 연결 중':data.status==='done'?'모델 파일 준비됨':'AI 초기화 중';
   showOpenStatus(`${OPEN_MODEL_LABELS[config.openModel]||'확장 AI'} · ${stateText}`);
   return;
  }
  const rec=openPending.get(data.id);if(!rec)return;
  openPending.delete(data.id);clearTimeout(rec.timer);
  if(data.type==='error')rec.reject(Error(data.message||'확장 모델 처리 실패'));
  else if(data.type==='result')rec.resolve(data.result);
 });
 w.addEventListener('error',event=>{
  if(openWorker!==w)return;
  console.error('open-detector-worker',event.message,event);
  rejectAll(Error(`확장 탐지 Worker 실행 실패: ${event.message||'모듈 네트워크/CORS 오류'}`));
 });
 w.addEventListener('messageerror',()=>rejectAll(Error('확장 탐지 Worker 메시지 처리 오류')));
 return w;
}
function callOpenWorker(type,payload,timeout){
 const w=startOpenWorker(),id=++openSeq;
 return new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>{
   openPending.delete(id);
   reject(Error(type==='init'?'모델 다운로드/초기화 시간 초과 (외부 모델 서버나 연결 확인)':'AI 분석 시간 초과'));
  },timeout);
  openPending.set(id,{resolve,reject,timer});
  try{w.postMessage({type,id,...payload})}catch(e){clearTimeout(timer);openPending.delete(id);reject(e)}
 });
}
async function loadClip(){
 if(LOCAL_ONLY)throw Error('전용 AI 배포판에서는 외부 확장 모델을 사용하지 않습니다');
 const modelId=OPEN_DETECTOR_MODELS.mobileclip;
 if(clip&&openLoadedModelId===modelId)return clip;
 if(clipPromise&&openRequestedId===modelId)return clipPromise;
 if(openWorker)stopOpenWorker();
 const generation=++openGeneration;
 openFailure=null;openRequestedId=modelId;
 showOpenStatus(`${OPEN_MODEL_LABELS[config.openModel]} · 준비 중…`);
 const task=Promise.resolve().then(()=>callOpenWorker('init',{modelId},120000)).then(()=>{
  if(openGeneration!==generation||openRequestedId!==modelId)throw Error('이전 확장 AI 초기화가 취소되었습니다');
  openLoadedModelId=modelId;
  clip=(image,labels,options)=>callOpenWorker('detect',{image,labels,options},INFERENCE_TIMEOUT_MS);
  showOpenStatus(`${OPEN_MODEL_LABELS[config.openModel]} · 준비 완료`,'ready');
  return clip;
 }).catch(e=>{
  // No impossible missions: if the remote model cannot load, remove only expanded missions.
  if(openGeneration===generation&&openRequestedId===modelId){stopOpenWorker();openFailure=e;
   showOpenStatus(`확장 AI 실패: ${String(e.message||e).slice(0,90)} · 설정에서 재시도`,'failed');
   if(chosen?.kind==='open'&&phase==='selection'){phase='idle';chosen=null;updateButtons();$('feedback').textContent='확장 AI 연결에 실패해 해당 물건을 제외했어요. 다시 미션을 뽑아 주세요.';}
   console.error('확장 탐지 모델 초기화 실패:',e);
   if(phase==='idle'||phase==='selection')renderMission();
  }
  throw e;
 });
 clipPromise=task;
 return task;
}
function warmupAI(){if(localReady||LOCAL_ONLY)return;
 // Load sequentially on low-memory Chromebooks: no simultaneous model compilation spike.
 // Preload happens when the page starts, NEVER when a particular roulette item appears.
 if(config.mode==='custom'){showOpenStatus('확장 AI 사용 안 함','off');return}
 const base=loadCoco().catch(e=>console.warn('기본 AI 사전 로딩 실패:',e));
 if(config.clipMode==='on'&&OPEN.some(o=>config.enabled.includes(o.key))){
  showOpenStatus('경량 확장 AI 대기 · 기본 AI 준비 후 시작','loading');
  base.then(()=>{if(config.mode!=='custom'&&config.clipMode==='on'&&OPEN.some(o=>config.enabled.includes(o.key)))return loadClip()})
   .catch(e=>console.warn('경량 확장 AI 사전 로딩 실패:',e));
 }else showOpenStatus('확장 AI 사용 안 함','off');
}
// MobileCLIP classifies the item held at the camera centre, rather than predicting bounding boxes.
// Every request compares ALL classes, including empty-background examples, not only the chosen item.
function openLabelToObject(label){return OPEN.find(o=>o.key===label)}
function takeFrame(source=null){
 const v=source||$('video');const w=v.videoWidth||v.width,h=v.videoHeight||v.height;
 if(!w||!h)throw Error('카메라 영상이 준비되지 않았습니다.');
 const c=document.createElement('canvas');c.width=256;c.height=256;
 const side=Math.min(w,h)*.85;
 c.getContext('2d').drawImage(v,(w-side)/2,(h-side)/2,side,side,0,0,256,256);
 return c;
}
async function detectOpenObjects(sourceCanvas=null,target=chosen,all=false){
 const requestedModel=OPEN_DETECTOR_MODELS.mobileclip;
 const detector=await loadClip();
 if(openLoadedModelId!==requestedModel)throw Error('확장 AI 설정이 변경되었습니다. 다시 시도하세요.');
 const image=takeFrame(sourceCanvas).toDataURL('image/jpeg',.78);
 const begin=performance.now();
 const result=await detector(image,[],{});
 lastOpenInferenceMs=Math.round(performance.now()-begin);openPredictionCount++;
 // Model scores are relative ranking values, not calibrated probabilities.
 const normalized=(Array.isArray(result)?result:[]).filter(r=>r&&Number.isFinite(r.score)&&Number.isFinite(r.similarity)).sort((a,b)=>b.score-a.score);
 const top=normalized[0],runner=normalized[1];
 const confident=top&&top.label!=='__none__'&&top.similarity>=.18&&top.score>=.38&&top.score-(runner?.score||0)>=.08;
 return normalized.filter(r=>r.label!=='__none__').map(r=>({
  key:r.label,label:r.label,score:r.score,
  similarity:r.similarity,verified:confident&&r.label===top.label,bbox:null
 }));
}
async function startCamera(){
 if(isLiveCamera())return true;
 if(cameraPromise)return cameraPromise;
 const session=++cameraSession;
 const task=(async()=>{
  let acquired=null;
  try{
   if(!navigator.mediaDevices?.getUserMedia)throw Error('HTTPS 또는 localhost에서 카메라를 실행해 주세요');
   acquired=await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment',width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false});
   if(session!==cameraSession){acquired.getTracks().forEach(t=>t.stop());return false}
   stream=acquired;
   $('video').srcObject=acquired;$('trainPreview').srcObject=acquired;
   await $('video').play();await $('trainPreview').play().catch(()=>{});
   if(session!==cameraSession){acquired.getTracks().forEach(t=>t.stop());if(stream===acquired)stream=null;return false}
   if(!$('video').videoWidth)throw Error('카메라에서 영상이 들어오지 않습니다');
   camReady=true;lastVideoTime=-1;lastPredictedVideoTime=-1;lastFrameHealthyAt=performance.now();
   for(const track of acquired.getVideoTracks())track.addEventListener('ended',()=>{if(stream===acquired)stopCamera()},{once:true});
   $('camEmpty').classList.add('hidden');$('camBtn').textContent='⏹ 카메라 끄기';$('engineStatus').textContent='카메라 연결됨';
   runDetection();return true;
  }catch(e){
   console.error('camera',e);acquired?.getTracks().forEach(t=>t.stop());
   if(session===cameraSession){if(stream===acquired)stream=null;camReady=false;$('video').srcObject=null;$('trainPreview').srcObject=null;$('engineStatus').textContent='카메라 실패';$('camEmpty').replaceChildren();const hint=document.createElement('small');hint.textContent=e.message||'카메라 권한을 확인해 주세요';$('camEmpty').append('📷',hint)}
   return false;
  }
 })();
 cameraPromise=task;
 try{return await task}finally{if(cameraPromise===task)cameraPromise=null}
}
function stopCamera(){
 ++cameraSession;invalidatePredictions();lastVideoTime=-1;lastPredictedVideoTime=-1;
 if(phase==='playing'||phase==='preparing'){phase='selection';stopTimer();$('feedback').textContent='카메라가 꺼졌어. 다시 켜고 미션을 시작해 줘!';updateButtons()}
 const old=stream;stream=null;camReady=false;old?.getTracks().forEach(t=>t.stop());
 $('video').srcObject=null;$('trainPreview').srcObject=null;
 $('camBtn').textContent='📷 카메라 켜기';$('camEmpty').classList.remove('hidden');$('engineStatus').textContent='카메라 꺼짐';
 const c=$('overlay');c.getContext('2d').clearRect(0,0,c.width,c.height);
}
function drawBoxes(preds){const cn=$('overlay'),v=$('video');if(!v.videoWidth)return;if(cn.width!==v.videoWidth)cn.width=v.videoWidth;if(cn.height!==v.videoHeight)cn.height=v.videoHeight;const c=cn.getContext('2d');c.clearRect(0,0,cn.width,cn.height);c.lineWidth=4;c.font=`bold ${Math.max(16,cn.width/45)}px system-ui`;for(const p of preds.filter(p=>p.score>.38)){let [x,y,w,h]=p.bbox;const isHit=(chosen?.kind==='coco'&&p.class===chosen.key)||(chosen?.kind==='open'&&p.class===chosen.name);c.strokeStyle=isHit?'#42ff96':'#5deeff';c.fillStyle=isHit?'#42ff96':'#5deeff';c.strokeRect(x,y,w,h);let label=`${p.class} ${Math.round(p.score*100)}%`;c.fillText(label,x+5,Math.max(28,y-8))}}
const ALL_ITEMS=[...COCO,...OPEN];
const displayName=key=>ALL_ITEMS.find(o=>o.key===key)?.name||classes.find(o=>o.id===key)?.name||key;
function showCandidates(candidates,kind){
 const el=$('candidateList');if(!el)return;
 const valid=(candidates||[]).filter(c=>Number.isFinite(c.score)&&c.score>=0).slice(0,5);
 el.replaceChildren();
 if(!valid.length){el.textContent='AI 분석 대기 중';return;}
 for(const c of valid){
  const line=document.createElement('div');line.className='candidate-row';
  const name=document.createElement('span');name.textContent=displayName(c.key);
  const track=document.createElement('div');track.className='candidate-track';
  const fill=document.createElement('i');fill.style.width=`${Math.min(100,Math.round(c.score*100))}%`;track.append(fill);
  const pct=document.createElement('strong');pct.textContent=`${(c.score*100).toFixed(1)}%`;
  if(c.key===chosen?.key)line.classList.add('candidate-target');
  line.append(name,track,pct);el.append(line);
 }
 $('candidateHint').textContent=kind==='coco'?'COCO 탐지 신뢰도 · 실제 정답 확률과 다를 수 있어요':'AI 상대 비교 점수 · 실제 정답 확률이 아니에요';
}
function localAccept(results,key,manifest=localModelManifest){
 const p=results[0],runner=results[1];
 return Boolean(p&&p.key===key&&key!=='__background__'&&p.score>=(manifest?.min_score??.68)&&p.score-(runner?.score??0)>=(manifest?.min_margin??.15));
}
function readTarget(){if(!chosen||phase!=='playing'||performance.now()-lastInferenceAt>PREDICTION_STALE_MS)return false;let now=performance.now(),ok=false,confidence=0,label='';if(localReady&&localLabels.has(chosen.key)){const p=lastLocal[0],runner=lastLocal[1];if(p&&p.key===chosen.key&&p.key!=='__background__'){ok=localAccept(lastLocal,chosen.key);confidence=p.score;label=chosen.name}}else if(chosen.kind==='coco'&&config.mode!=='custom'){const m=lastPred.filter(p=>p.class===chosen.key).sort((a,b)=>b.score-a.score)[0];if(m){ok=m.score>=.58;confidence=m.score;label=m.class}}if(chosen.kind==='custom'&&config.mode!=='general'){const p=lastCustom[0];if(p&&p.classIndex===chosen.id&&(knn?.getClassExampleCount()?.[chosen.id]||0)>=MIN_SAMPLES&&(knn?.getClassExampleCount()?.['__background__']||0)>=MIN_SAMPLES){const rival=lastCustom[1]?.confidence||0;ok=p.confidence>=.78 && p.confidence-rival>=.20;confidence=p.confidence;label=chosen.name}}if(chosen.kind==='open'&&config.clipMode==='on'){const p=lastClip[0];if(p&&p.key===chosen.key&&p.verified){ok=true;confidence=p.score;label=chosen.name}}if(ok){if(holdKey!==chosen.key||(lastPositiveAt&&now-lastPositiveAt>((localReady||chosen.kind==='open')?OPEN_CONFIRM_MAX_GAP_MS:PREDICTION_STALE_MS))){holdKey=chosen.key;holdStart=now;confirmedFrames=0}lastPositiveAt=now;confirmedFrames++;const elapsed=now-holdStart;$('progressFill').style.width=Math.min(100,100*elapsed/(config.hold*1000))+'%';$('feedback').textContent=`${chosen.name} 판별 중… ${confirmedFrames}회 확인`;if(elapsed>=config.hold*1000&&confirmedFrames>=2)finish(true)}else{holdStart=0;holdKey='';confirmedFrames=0;lastPositiveAt=0;$('progressFill').style.width='0%';$('feedback').textContent=`찾을 물건: ${chosen.name} · ${chosen.kind==='open'?'화면 중앙에 크게 비춰줘!':'카메라에 보여줘!'}`}return ok}
async function doPredict(){
 const predictionId=++latestPredictionId;
 if(!camReady||!$('video').videoWidth||$('video').readyState<2)throw Error('카메라에서 영상 프레임을 받지 못했습니다');
 const activeStream=stream, activeSession=cameraSession, target=phase==='playing'?chosen:null;
 const testing=phase!=='playing';
 const useLocal=localReady&&(testing||Boolean(target&&localLabels.has(target.key)));
 const useCoco=!useLocal&&config.mode!=='custom'&&(testing||target?.kind==='coco');
 const useCustom=!useLocal&&config.mode!=='general'&&(testing||target?.kind==='custom');
 const useClip=!useLocal&&config.mode!=='custom'&&config.clipMode==='on'&&clip&&(testing||target?.kind==='open');
 let msgs=[],preds=[],custom=[],clips=[],locals=[];
 if(useLocal){locals=await detectLocal($('video'));msgs.push(...locals.slice(0,3).filter(x=>x.key!=='__background__').map(x=>`${displayName(x.key)}: 상대점수 ${(x.score*100).toFixed(1)}%`))}
 if(useCoco){try{const m=await loadCoco();preds=await m.detect($('video'),12,.28);msgs.push(...preds.slice(0,5).map(p=>`${p.class}: ${Math.round(p.score*100)}%`))}catch(e){modelError('기본 AI',e);if(target?.kind==='coco')throw e}}
 if(useCustom){try{await loadCustom();if(knn.getNumClasses()>=2){const emb=tf.tidy(()=>mobile.infer($('video'),true));try{const pred=await knn.predictClass(emb,3);custom=Object.entries(pred.confidences).map(([classIndex,confidence])=>({classIndex,confidence})).sort((a,b)=>b.confidence-a.confidence);msgs.push(...custom.slice(0,2).map(p=>`${classes.find(c=>c.id===p.classIndex)?.name||p.classIndex}: ${Math.round(p.confidence*100)}%`))}finally{emb.dispose()}}}catch(e){modelError('학습 AI',e);if(target?.kind==='custom')throw e}}
 if(useClip){try{
   clips=await detectOpenObjects();
   msgs.push(...clips.slice(0,3).map(p=>`${openLabelToObject(p.label)?.name||p.label}: 상대점수 ${Math.round(p.score*100)}%${p.verified?' ✓':''}`));
  }catch(e){modelError('확장 탐지',e);if(target?.kind==='open')throw e}}
 // Never attribute a stale inference to a different stream or a different mission.
 if(predictionId!==latestPredictionId||!camReady||cameraSession!==activeSession||stream!==activeStream||(target&&(chosen!==target||phase!=='playing'))||!['playing','selection','idle'].includes(phase))return;
 lastPred=preds;lastCustom=custom;lastClip=clips;lastLocal=locals;
 drawBoxes(target?.kind==='open'||useLocal?[]:preds);
 lastInferenceAt=performance.now();$('detected').textContent=msgs.length?msgs[0]:'인식 물품 없음';
 const mode=target?.kind||chosen?.kind;
 if(useLocal)showCandidates(locals,'relative');
 else if(mode==='open'&&clips.length)showCandidates(clips.map(x=>({key:x.key,score:x.score})),'relative');
 else if(mode==='custom'&&custom.length)showCandidates(custom.map(x=>({key:x.classIndex,score:x.confidence})),'relative');
 else if(preds.length)showCandidates(preds.map(x=>({key:x.class,score:x.score})).sort((a,b)=>b.score-a.score),'coco');
 else showCandidates([],'relative');
 if(phase==='playing')readTarget();
}
async function runDetection(){
 // One inference loop per camera generation. If a replacement camera arrives while
 // the old loop is winding down, the finally block starts a fresh loop for it.
 if(loopActive)return;
 loopActive=true;
 const detectionSession=cameraSession;
 try{
  while(camReady && cameraSession===detectionSession){
   if(phase==='playing'){
    try{
     const currentTime=$('video').currentTime;
     const track=stream?.getVideoTracks()?.[0];
     if(!track||track.readyState!=='live'||!camReady)throw Error('카메라 연결이 종료되었습니다');
     // Video may be visible while the camera is frozen. Check media timestamps before each inference.
     if($('video').readyState<2||!$('video').videoWidth){
      if(performance.now()-lastFrameHealthyAt>CAMERA_FRAME_TIMEOUT_MS)throw Error('카메라 영상이 준비되지 않습니다');
      await sleep(250);continue;
     }
     if(currentTime!==lastVideoTime){lastVideoTime=currentTime;lastFrameHealthyAt=performance.now()}
     else if(performance.now()-lastFrameHealthyAt>CAMERA_FRAME_TIMEOUT_MS)throw Error('카메라 영상이 멈췄습니다');
     // Only classify fresh video frames; repeated processing of one frozen frame must not count twice.
     if(currentTime===lastPredictedVideoTime){await sleep(90);continue}
     lastPredictedVideoTime=currentTime;
     activeInferenceSince=performance.now();
     // A never-settling model call must not deadlock the detection loop.
     let inferenceTimeoutId;
     try {
      if(inferenceStuck)throw Error('이전 AI 추론이 종료되지 않았습니다');
      const job=doPredict();inferenceJob=job;
      await Promise.race([job,new Promise((_,reject)=>{inferenceTimeoutId=setTimeout(()=>reject(new Error('AI 추론 시간 초과')),INFERENCE_TIMEOUT_MS)})]);
     } catch(e){
      // Ignore failures from an obsolete camera generation.
      if(cameraSession!==detectionSession)break;
      if(String(e.message||e).includes('시간 초과')){
       inferenceStuck=true;invalidatePredictions();
       const old=inferenceJob;
       old?.finally(()=>{if(inferenceJob===old){inferenceJob=null;inferenceStuck=false}}).catch(()=>{});
       pauseForInferenceError('AI 분석이 멈췄습니다. 분석이 끝나면 다시 시도하거나 페이지를 새로고침해 주세요.');
      }
      throw e;
     } finally {if(inferenceTimeoutId)clearTimeout(inferenceTimeoutId);if(!inferenceStuck)inferenceJob=null} 
     if(phase==='playing')inferenceFailures=0;
    }
    catch(e){
     if(cameraSession!==detectionSession)break;
     console.error('inference',e);
     inferenceFailures++;
     if(inferenceFailures>=2&&phase==='playing')pauseForInferenceError('AI 또는 카메라 인식 오류! 연결을 확인하고 다시 시작해 줘.');
    }finally{activeInferenceSince=0}
   }
   await sleep(260);
  }
 }finally{
  loopActive=false;
  if(camReady && cameraSession!==detectionSession){
   // Do not strand a new stream when its initial runDetection call hit loopActive.
   queueMicrotask(()=>{if(camReady && !loopActive)runDetection()});
  }
 }
}
function switchTab(id){for(const el of document.querySelectorAll('.tab-body'))el.classList.toggle('hidden',el.id!==id);for(const b of document.querySelectorAll('.tabs button'))b.classList.toggle('active',b.dataset.tab===id);if(id==='training')renderTraining()}
function renderObjects(){let groups={};for(const o of [...COCO,...OPEN]){let g=o.group+(o.kind==='open'?' (화면 중앙 물건 분류)':'');(groups[g]??=[]).push(o)}$('objectList').replaceChildren();for(const [group,objects] of Object.entries(groups)){let h=document.createElement('strong');h.style.gridColumn='1/-1';h.textContent=group;$('objectList').appendChild(h);for(const o of objects){let l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.checked=config.enabled.includes(o.key);i.onchange=()=>{if(i.checked)config.enabled.push(o.key);else config.enabled=config.enabled.filter(x=>x!==o.key);saveConfig();if(o.kind==='open'&&!localReady&&!LOCAL_ONLY){if(i.checked)warmupAI();else if(!OPEN.some(item=>config.enabled.includes(item.key))){stopOpenWorker();openFailure=null;showOpenStatus('확장 AI 사용 안 함','off')}}};l.append(i);if(ITEM_ICON_ASSETS[o.key]){const pic=document.createElement('img');pic.src=ITEM_ICON_ASSETS[o.key];pic.className='object-list-icon';pic.alt='';l.append(pic)}else l.append(document.createTextNode(o.emoji+' '));l.append(document.createTextNode(o.name));$('objectList').appendChild(l)}}}
function renderTraining(){let container=$('trainingList');container.replaceChildren();for(const c of classes){let row=document.createElement('div');row.className='train-row';let title=document.createElement('strong');title.textContent=`${c.name} (${c.count||0}장 · ${c.count>=MIN_SAMPLES?'예시 충족':'최소 '+MIN_SAMPLES+'장 필요'})`;let add=document.createElement('button');add.className='btn cyan';add.textContent='📷 현재 화면 학습';add.onclick=()=>trainSample(c.id);let toggle=document.createElement('button');toggle.className='btn secondary';toggle.textContent=c.enabled===false?'미션에 포함':'미션에서 제외';toggle.onclick=()=>{c.enabled=c.enabled===false;persistClasses();renderTraining()};let del=document.createElement('button');del.className='btn secondary';del.textContent='삭제';del.onclick=async()=>{
 if(!window.confirm('이 물품의 학습을 삭제할까요?'))return;
 if(!knn && !Number(c.count||0)){
  try{
   const backup=await idbGet('training').catch(()=>null);
   if(!backup?.dataset?.[c.id]){
    const updated=classes.filter(x=>x.id!==c.id);
    if(backup?.dataset)await idbSet('training',{...backup,classes:updated});
    classes=updated;persistClasses();renderTraining();return;
   }
  }catch(e){alert('삭제 저장 실패: '+e.message);return}
 }
 let previous=null,oldClasses=classes;
 try{
  await loadCustom();const old=knn.getClassifierDataset()[c.id];if(old)previous=old.clone();
  if(old)knn.clearClass(c.id);classes=classes.filter(x=>x.id!==c.id);
  await persistTraining();renderTraining();
 }catch(e){
  classes=oldClasses;
  if(previous){knn.setClassifierDataset({...knn.getClassifierDataset(),[c.id]:previous});previous=null}
  renderTraining();alert('삭제 저장 실패(기존 자료 보존): '+e.message);
 }finally{if(previous)previous.dispose()}
};row.append(title,add,toggle,del);container.appendChild(row)}let bg=document.createElement('div');bg.className='train-row';let lab=document.createElement('strong');lab.textContent='배경 / 다른 물건 (필수 권장)';let b=document.createElement('button');b.className='btn cyan';b.textContent='📷 배경 학습';b.onclick=()=>trainSample('__background__');bg.append(lab,b);const n=document.createElement('small');n.textContent=`${knn?.getClassExampleCount()?.['__background__']||0}장 / 최소 ${MIN_SAMPLES}장`;bg.append(n);container.appendChild(bg)}
async function trainSample(id){
 if(!camReady){alert('먼저 게임 화면에서 카메라를 켜 주세요.');return}
 if(performance.now()-lastTrainingFrameAt<400){$('saved').textContent='각도를 바꾸고 잠깐 뒤 다시 촬영해 주세요.';return}
 let previous=null,previousCount=0,committed=false,added=false;
 try{
  await loadCustom();
  const c=classes.find(x=>x.id===id);
  if(id!=='__background__'&&!c)throw Error('등록되지 않은 물품입니다');
  previousCount=c?.count||0;
  const old=knn.getClassifierDataset()[id];
  if(old)previous=old.clone();
  const embedding=tf.tidy(()=>mobile.infer($('video'),true));
  try{knn.addExample(embedding,id);added=true}finally{embedding.dispose()}
  if(c)c.count=knn.getClassExampleCount()[id]||0;
  // Never report success until the entire dataset has been committed to IndexedDB.
  await persistTraining();committed=true;lastTrainingFrameAt=performance.now();
  renderTraining();$('saved').textContent='학습 예시 저장 완료 · 여러 각도와 배경 예시를 학습해 주세요.';
 }catch(e){
  // An IndexedDB quota/security failure must not leave a ghost training sample in memory.
  if(added){
   knn.clearClass(id);
   if(previous){knn.setClassifierDataset({...knn.getClassifierDataset(),[id]:previous});previous=null}
  }
  const c=classes.find(x=>x.id===id);if(c)c.count=previousCount;
  renderTraining();alert('학습 저장 실패(기존 상태 보존): '+e.message);
 }finally{if(previous)previous.dispose()}
}
async function persistTraining(){if(!knn){persistClasses();return}const dataset=knn.getClassifierDataset();const save={dataset:{},classes:classes.map(c=>({...c}))};for(const [key,t] of Object.entries(dataset)){const data=await t.data();save.dataset[key]={shape:[...t.shape],data:Array.from(data)}}await idbSet('training',save);persistClasses()}
function persistClasses(){storage.setItem('detective_classes',JSON.stringify(classes))}
function openDb(){return new Promise((resolve,reject)=>{if(!('indexedDB' in window))return reject(Error('브라우저가 IndexedDB 저장소를 지원하지 않습니다'));let req=indexedDB.open('ai-object-detective',1);req.onupgradeneeded=()=>req.result.createObjectStore('data');req.onsuccess=()=>resolve(req.result);req.onerror=()=>reject(req.error)})}
async function idbGet(key){const db=await openDb();return new Promise((res,rej)=>{const tx=db.transaction('data','readonly');const r=tx.objectStore('data').get(key);r.onsuccess=()=>res(r.result||null);r.onerror=()=>rej(r.error);tx.oncomplete=()=>db.close()})}
async function idbSet(key,value){const db=await openDb();return new Promise((res,rej)=>{const tx=db.transaction('data','readwrite');tx.objectStore('data').put(value,key);tx.oncomplete=()=>{db.close();res()};tx.onerror=()=>rej(tx.error)})}
function updateFields(){for(let id of ['mode','clipMode','openModel','duration','hold','rerollCount'])$(id).value=config[id];$('tts').checked=config.tts;$('autoReturn').checked=config.autoReturn}
function commitFields(){if(phase==='playing'||phase==='preparing'){pauseForInferenceError('설정이 변경되어 미션을 중단했어요. 다시 시작해 주세요.')}invalidatePredictions();const formerModel=config.openModel;for(let id of ['mode','clipMode','openModel'])config[id]=$(id).value;if(config.openModel!==formerModel||config.mode==='custom'||config.clipMode!=='on'){stopOpenWorker();openFailure=null;}config.duration=Math.min(180,Math.max(15,+$('duration').value||60));config.hold=Math.min(3,Math.max(.2,+$('hold').value||.65));config.rerollCount=Math.min(3,Math.max(0,+$('rerollCount').value||0));config.tts=$('tts').checked;config.autoReturn=$('autoReturn').checked;saveConfig();remainingRerolls=config.rerollCount;$('saved').textContent='설정 저장 완료';$('clock').textContent=fmt(config.duration);// Settings affecting the detector trigger a single new preload; no per-object downloads.
 if(!localReady)warmupAI();updateButtons()}
function diagnosticReport(){
 const v=$('video'),track=stream?.getVideoTracks()?.[0];
 const d=[
  ['보안 연결',window.isSecureContext?'정상':'HTTPS 필요'],
  ['카메라 API',navigator.mediaDevices?.getUserMedia?'지원':'지원되지 않음'],
  ['웹캠 연결',camReady?'연결됨':'연결 전'],
  ['영상 크기',v.videoWidth?`${v.videoWidth}×${v.videoHeight}`:'없음'],
  ['영상 상태',track?`${track.readyState} / ${track.muted?'음소거됨':'정상'}`:'없음'],
  ['TensorFlow.js',window.tf?`로드됨${tf.getBackend?.()?' / '+tf.getBackend():''}`:'미로드 (미션 시작 시 다운로드)'],
  ['COCO-SSD 스크립트',window.cocoSsd?'로드됨':'미로드'],
  ['전용 모델',localReady?`준비됨 (${localModelManifest?.labels?.length-1}종 / 자체 호스팅)`:'전용 가중치 미포함 · 기존 엔진 사용'],
  ['COCO 모델',model?'준비됨':'아직 준비 안 됨'],
  ['MobileNet·KNN',window.mobilenet&&window.knnClassifier?'스크립트 로드됨':'스크립트 미로드'],
  ['직접학습',knn?`${knn.getNumClasses()}개 분류 준비됨`:'아직 준비 안 됨'],
  ['확장 분류 ('+(OPEN_MODEL_LABELS[config.openModel]||'')+')',clip?'준비됨':openFailure?'실패: '+openFailure.message:openState==='loading'?'다운로드·초기화 중':'아직 준비 안 됨'],
  ['확장 분류 Worker',openWorker?'연결됨':'없음'],
  ['확장 라이브러리','jsDelivr Transformers.js 3.8.1 / MobileCLIP-S0 q8 / 약 55MB' ],
  ['렌더링 FPS(최근 약 1초)',renderFps||'측정 중'],
  ['경량 확장 분류 시간',openPredictionCount?lastOpenInferenceMs+'ms / 마지막 1회':'분석 전'],
  ['최근 인식',lastInferenceAt?`${Math.round((performance.now()-lastInferenceAt)/1000)}초 전`:'없음'],
  ['현재 미션',chosen?.name||'없음']
 ];
 return `AI 사물 탐정 진단\n${new Date().toLocaleString('ko-KR')}\n`+d.map(([k,val])=>`${k}: ${val}`).join('\n');
}
$('diagnoseBtn').onclick=()=>{$('diagResult').textContent=diagnosticReport()};
$('prepareOpenBtn').onclick=async()=>{
 const btn=$('prepareOpenBtn');btn.disabled=true;
 try{if(localPreload)await localPreload;if(localReady){$('testResult').textContent='전용 AI 준비 완료 · 외부 확장 모델 다운로드 불필요';return}stopOpenWorker();openFailure=null;showOpenStatus('확장 AI 재시도 중…');await loadClip();$('testResult').textContent='✅ 경량 확장 AI 준비 완료. 카메라 중앙의 물건을 9종 중에서 분류합니다.';}
 catch(e){$('testResult').textContent='❌ 확장 AI 준비 실패: '+e.message+'\n관리자 설정 → 환경 점검에서 원인을 확인하세요.';}
 finally{btn.disabled=false}
};
$('prepareAiBtn').onclick=async()=>{
 const btn=$('prepareAiBtn');btn.disabled=true;
 $('testResult').textContent='기본 AI를 준비하는 중입니다. 최초에는 모델 다운로드가 필요합니다…';
 try{if(localPreload)await localPreload;if(localReady){$('testResult').textContent='전용 AI 준비 완료 · 외부 기본 모델 다운로드 불필요';return}await loadCoco();$('testResult').textContent='✅ 기본 AI 준비 완료! 카메라를 켜고 인식 테스트를 진행하세요.'}
 catch(e){modelError('기본 AI',e);$('testResult').textContent='❌ '+e.message+'\n네트워크를 확인한 후 다시 눌러주세요.'}
 finally{btn.disabled=false}
};
// End-to-end verification runs the *real* downloaded model against *real* webcam frames.
// No generated/mock labels are used, and camera frames are never uploaded by this code.
let liveCheckBusy=false;
async function verifyRealCameraAndModel(){
 const btn=$('liveCheckBtn'),out=$('liveCheckResult');
 if(liveCheckBusy||aiBusy)return;
 if(phase==='playing'||phase==='preparing'||phase==='spinning'){out.textContent='진행 중인 미션이 끝난 뒤 검사해 주세요.';return}
 liveCheckBusy=true;btn.disabled=true;
 const lines=[],write=s=>{lines.push(s);out.textContent=lines.join('\n')};
 try{
  write('1/4 · 보안 연결: '+(window.isSecureContext?'정상':'HTTPS 필요'));
  if(!window.isSecureContext)throw Error('GitHub Pages HTTPS에서 접속해 주세요.');
  if(!navigator.mediaDevices?.getUserMedia)throw Error('브라우저에서 카메라 API를 사용할 수 없습니다.');
  write('2/4 · 실제 카메라 권한 및 영상 확인 중…');
  if(!camReady){const ok=await startCamera();if(!ok)throw Error('카메라 권한 또는 연결 실패')}
  const video=$('video');
  if(!video.videoWidth||video.readyState<2)throw Error('카메라 영상 프레임 없음');
  write(`카메라: ${video.videoWidth}×${video.videoHeight} / ${stream?.getVideoTracks()?.[0]?.readyState||'unknown'}`);
  write('3/4 · 실제 모델 가중치 확인 중…');
  if(localPreload)await localPreload;const useLocal=localReady;const m=useLocal?null:await loadCoco();if(!useLocal&&typeof m.detect!=='function')throw Error('AI 모델이 로드됐으나 detect 함수가 없습니다');
  write('AI 모델: 실제 로딩 완료 / '+(useLocal?'내장 ONNX WASM':(tf.getBackend?.()||'unknown')));
  write('4/4 · 실제 영상으로 3회 추론 중…');
  const results=[];
  for(let i=0;i<3;i++){
   if(!camReady||!stream?.getVideoTracks()?.some(t=>t.readyState==='live'))throw Error('검사 도중 카메라 연결이 끊어졌습니다');
   await sleep(300);
   const begin=performance.now();
   const predictions=useLocal?(await detectLocal(video)).map(p=>({class:displayName(p.key),score:p.score})):await m.detect(video,12,.25);
   results.push({ms:Math.round(performance.now()-begin),items:predictions.slice(0,5).map(p=>`${p.class} ${(p.score*100).toFixed(0)}%`)});
  }
  for(let i=0;i<results.length;i++)write(`${i+1}차 ${results[i].ms}ms: ${results[i].items.join(', ')||'감지된 사물 없음'}`);
  write('✅ 카메라 + 실제 AI 가중치 + 영상 추론 성공');
  write('※ 사물이 감지되지 않으면 물건을 카메라 가까이 비춘 뒤 다시 검사하세요. 이 결과는 모든 물품의 인식 정확도를 보장하지 않습니다.');
 }catch(e){
  write('❌ 실제 기기 검사 실패: '+String(e?.message||e));
  write('카메라 권한, HTTPS 주소, 인터넷 연결 및 학교망의 AI 모델/CDN 차단을 확인하세요.');
 }finally{liveCheckBusy=false;btn.disabled=false}
}
// Admin-only real model test on a local photo; no camera permission needed.
// This is intentionally a live model invocation, not a simulated confidence score.
$('photoTestBtn').onclick=async()=>{
 const file=$('photoInput').files?.[0];const key=$('photoTarget').value;
 if(!file){$('photoTestResult').textContent='사진을 선택해 주세요.';return}
 if(phase==='playing'||phase==='preparing'||inferenceJob||aiBusy||liveCheckBusy){$('photoTestResult').textContent='게임 또는 다른 분석이 끝난 뒤 시험해 주세요.';return}
 const item=[...COCO,...OPEN].find(o=>o.key===key);
 if(!item){$('photoTestResult').textContent='목표 물품을 선택해 주세요.';return}
 aiBusy=true;$('photoTestBtn').disabled=true;
 try{
  $('photoTestResult').textContent='이미지 읽기·모델 준비 중…';
  if(localPreload)await localPreload;
  const bitmap=await createImageBitmap(file);
  const scale=Math.min(1,640/Math.max(bitmap.width,bitmap.height));
  const c=document.createElement('canvas');c.width=Math.round(bitmap.width*scale);c.height=Math.round(bitmap.height*scale);
  c.getContext('2d').drawImage(bitmap,0,0,c.width,c.height);bitmap.close();
  const t=performance.now();let found=[];
  if(localReady&&localLabels.has(item.key)){const results=await detectLocal(c);const accepted=localAccept(results,item.key);found=results.map(o=>({key:o.key,label:o.key,score:o.score,verified:o.key===item.key&&accepted}));}
  else if(item.kind==='open')found=await detectOpenObjects(c,item,false);
  else{
   const m=await loadCoco();
   found=(await m.detect(c,12,.18)).map(r=>({key:r.class,label:r.class,score:r.score,bbox:r.bbox}));
  }
  const ms=Math.round(performance.now()-t),hit=found.find(r=>r.key===item.key);
  const others=found.filter(r=>r.key!==item.key).slice(0,4);
  const status=hit?(((localReady&&localLabels.has(item.key))||item.kind==='open'?hit.verified:hit.score>=.58)?'정답 후보 판별':'목표 순위 낮음 (재촬영 필요)'):'목표 물품 판별 안 됨';
  $('photoTestResult').textContent=`${status}\n목표: ${item.name}\n처리 시간: ${ms}ms\n목표 점수: ${hit?(hit.score*100).toFixed(1)+'%':'없음'}\n다른 감지: ${others.map(r=>(openLabelToObject(r.label)?.name||r.label)+' '+(r.score*100).toFixed(0)+'%').join(', ')||'없음'}\n※ 확률은 정확도 보증이 아닙니다. 같은 장소에서 다른 물건 사진도 비교하세요.`;
 }catch(e){$('photoTestResult').textContent='실제 모델 검사 실패: '+String(e?.message||e)}
 finally{aiBusy=false;$('photoTestBtn').disabled=false}
};
$('liveCheckBtn').onclick=verifyRealCameraAndModel;
$('copyDiagBtn').onclick=async()=>{const report=diagnosticReport();try{await navigator.clipboard.writeText(report);$('diagResult').textContent=report+'\n\n클립보드 복사 완료'}catch{$('diagResult').textContent=report+'\n\n복사 불가: 위 내용을 직접 복사하세요'}};
$('startBtn').onclick=()=>{unlockSound();return startSelection()};$('rerollBtn').onclick=()=>{unlockSound();if(phase==='selection'&&remainingRerolls>0&&pool().length>1){remainingRerolls--;const before=chosen.key;spin(before)}};$('confirmBtn').onclick=()=>{unlockSound();playSfx('tap',.65);return confirm()};$('resetBtn').onclick=resetGame;$('resultAgain').onclick=()=>{playSfx('tap');resetGame()};
$('camBtn').onclick=()=>{playSfx('camera',.55);return camReady?stopCamera():startCamera()};$('trainCamBtn').onclick=()=>camReady?stopCamera():startCamera();$('settingsBtn').onclick=()=>{updateFields();renderObjects();renderTraining();$('settings').showModal()};$('closeSettings').onclick=()=>$('settings').close();$('saveBtn').onclick=()=>{commitFields();$('settings').close()};document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>switchTab(b.dataset.tab));$('allOff').onclick=()=>{config.enabled=[];saveConfig();if(!localReady&&!LOCAL_ONLY){stopOpenWorker();openFailure=null;showOpenStatus('확장 AI 사용 안 함','off')}renderObjects()};$('allOn').onclick=()=>{config.enabled=[...RECOMMENDED];saveConfig();renderObjects();if(!localReady)warmupAI()};$('soundBtn').onclick=()=>{unlockSound();const newState=!(config.tts||config.sfx);config.tts=newState;config.sfx=newState;saveConfig();refreshSoundUI();if(!config.tts&&'speechSynthesis'in window)speechSynthesis.cancel();if(newState)playSfx('tap')};$('fullBtn').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen?.().catch(()=>{})};
$('addClass').onclick=()=>{const name=$('trainName').value.trim();if(!name)return alert('물건 이름을 입력해 주세요');if(classes.some(c=>c.name===name))return alert('같은 이름이 이미 있어요');const id='custom_'+Date.now()+'_'+Math.random().toString(36).slice(2,5);classes.push({id,key:id,name,desc:$('trainDesc').value.trim()||`찾아야 할 물건: ${name}`,emoji:'📦',count:0,enabled:true});persistClasses();$('trainName').value='';$('trainDesc').value='';renderTraining()};
$('testBtn').onclick=async()=>{if(aiBusy)return;if(phase==='playing'||phase==='preparing'||inferenceJob){$('testResult').textContent='미션 또는 다른 AI 분석이 끝난 후 다시 시험해 주세요.';return}if(!camReady){$('testResult').textContent='카메라를 먼저 켜 주세요.';return}try{aiBusy=true;$('testResult').textContent='경량 AI 모델을 준비하는 중…';if(!localReady){if(config.mode!=='custom')await loadCoco();if(config.mode!=='general')await loadCustom();if(config.clipMode==='on'&&config.mode!=='custom')await loadClip();}await doPredict();let lines=[localReady?'전용 학습 AI:':'기존 AI:'];lines.push(...lastLocal.slice(0,6).map(p=>`• ${displayName(p.key)}: 상대점수 ${(p.score*100).toFixed(1)}%`));lines.push(...lastPred.slice(0,8).map(p=>`• ${p.class}: ${(p.score*100).toFixed(1)}%`));lines.push('직접 학습 AI:');lines.push(...lastCustom.slice(0,5).map(p=>`• ${classes.find(c=>c.id===p.classIndex)?.name||p.classIndex}: ${(p.confidence*100).toFixed(1)}%`));lines.push('확장 객체 탐지:');lines.push(...lastClip.slice(0,5).map(p=>`• ${p.label}: ${(p.score*100).toFixed(1)}%`));$('testResult').textContent=lines.join('\n')}catch(e){$('testResult').textContent=e.message}finally{aiBusy=false}};
$('exportData').onclick=async()=>{try{const data={app:'ai-object-detective',version:1,classes,training:await idbGet('training')};const blob=new Blob([JSON.stringify(data)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='AI_사물탐정_학습백업.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000)}catch(e){alert(e.message)}};
$('importData').onchange=async e=>{let f=e.target.files?.[0];if(!f)return;try{const data=JSON.parse(await f.text());if(data.app!=='ai-object-detective'||!Array.isArray(data.classes))throw Error('올바른 학습 백업 파일이 아닙니다');if(!window.confirm('기존 학습 데이터를 새 백업으로 대체할까요?'))return;if(!data.training||typeof data.training!=='object'||!data.training.dataset||typeof data.training.dataset!=='object')throw Error('학습 데이터가 없는 백업입니다');
 const importedClasses=data.classes.filter(c=>c&&typeof c.id==='string'&&c.id.startsWith('custom_')&&typeof c.name==='string'&&c.name.length<=40&&typeof c.desc==='string'&&c.desc.length<=200);
 if(importedClasses.length!==data.classes.length||importedClasses.length>200||new Set(importedClasses.map(c=>c.id)).size!==importedClasses.length)throw Error('학습 물품 정보가 손상됐습니다');
 const importedDataset=data.training.dataset;let total=0;for(const [key,obj] of Object.entries(importedDataset)){if(key!=='__background__'&&!importedClasses.some(c=>c.id===key))throw Error('학습 데이터에 알 수 없는 물건이 있습니다');if(!obj||!Array.isArray(obj.shape)||obj.shape.length!==2||!obj.shape.every(n=>Number.isInteger(n)&&n>0)||!Array.isArray(obj.data)||obj.shape[0]*obj.shape[1]!==obj.data.length||obj.data.length>5e6||!obj.data.every(Number.isFinite))throw Error('손상된 학습 데이터입니다');total+=obj.data.length;if(total>6e6)throw Error('학습 데이터 용량이 너무 큽니다')}
 const prior=await idbGet('training');await idbSet('training',{dataset:importedDataset,classes:importedClasses});
 try{classes=importedClasses;persistClasses()}catch(e){await idbSet('training',prior);throw e}
 if(knn){knn.dispose();knn=null;mobile=null;customPromise=null}renderTraining();alert('가져오기 완료')}catch(err){alert('가져오기 실패: '+err.message)}e.target.value=''};
$('clearData').onclick=async()=>{if(!window.confirm('모든 직접 학습 물건과 사진 특징 데이터를 삭제할까요?'))return;try{await idbSet('training',null);classes=[];persistClasses();if(knn){knn.dispose();knn=null;mobile=null;customPromise=null}renderTraining()}catch(e){alert('학습 자료 삭제 실패: '+e.message+' · 기존 학습 정보를 보존했어요.')}};
$('sfx').checked=config.sfx;$('sfxVolume').value=config.sfxVolume;
$('sfx').onchange=()=>{config.sfx=$('sfx').checked;saveConfig();refreshSoundUI();if(config.sfx)playSfx('tap')};
$('sfxVolume').oninput=()=>{config.sfxVolume=Number($('sfxVolume').value);saveConfig()};
$('sfxTest').onclick=()=>{unlockSound();playSfx('success')};
document.addEventListener('pointerdown',unlockSound,{once:true,passive:true});
$('photoTarget').replaceChildren(...[...COCO,...OPEN].map(o=>new Option(o.name,o.key)));$('photoTarget').value='pencil';
if(LOCAL_ONLY){
 for(const id of ['mode','clipMode','openModel'])$(id).closest('label').hidden=true;
 document.querySelector('[data-tab=training]').hidden=true;
 $('prepareOpenBtn').hidden=true;$('prepareAiBtn').hidden=true;
 $('general').querySelector('.hint').textContent='내장 전용 AI는 같은 사이트에서 한 번만 모델을 로드합니다. 외부 AI 서비스로 영상이나 사진을 보내지 않습니다. 카메라는 HTTPS 또는 localhost에서 허용하세요.';
 $('objects').querySelector('p').textContent='전용 AI가 지원하는 전체 38종 중 행사장에 준비된 물건만 활성화하세요.';
 $('objects').querySelector('.hint').textContent='물건을 화면 중앙에 크게 보여주세요. 실제 물품으로 정확도를 먼저 시험하세요.';
 $('tests').querySelector('p').textContent='카메라 또는 사진으로 내장 전용 AI의 상대 비교 점수를 확인할 수 있습니다.';
 $('tests').lastElementChild.textContent='같은 사이트에 포함된 전용 AI만 사용합니다. 외부 AI 모델로 전환하지 않습니다.';
}
resetGame();refreshSoundUI();localPreload=initializeLocalModel().then(ok=>{if(!ok)warmupAI();return ok});window.addEventListener('beforeunload',()=>{stopOpenWorker();stopCamera();if('speechSynthesis'in window)speechSynthesis.cancel()});
