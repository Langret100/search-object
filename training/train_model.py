"""Train/validate MobileNetV3-Small head on reviewed per-class photos; export ONNX.

The output is intentionally blocked when training images are missing or when
holdout recall fails. No pretrained weights or photos are fabricated by this script.
"""
from __future__ import annotations
import argparse,hashlib,json,random,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageEnhance
import torch
from torch import nn
from torchvision import models,transforms as T

HERE=Path(__file__).parent
LABELS=[]  # selected catalogue populated at CLI entrypoint
SIZE=224
MIN_SCORE=.68
MIN_MARGIN=.15

def transform(augment=False):
 def base(im):
  im=ImageOps.exif_transpose(im).convert('RGB')
  im=ImageOps.pad(im,(SIZE,SIZE),method=Image.Resampling.BILINEAR,color=(128,128,128),centering=(.5,.5))
  return im
 ops=[T.Lambda(base)]
 if augment:ops+=[T.RandomHorizontalFlip(),T.RandomAffine(degrees=9,translate=(.04,.04)),T.ColorJitter(brightness=.18,contrast=.22,saturation=.16)]
 ops+=[T.ToTensor(),T.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])]
 return T.Compose(ops)

@torch.inference_mode()
def get_features(model,paths,aug=0,batch=24):
 pre=transform(aug!=0);features=[]
 for start in range(0,len(paths),batch):
  sub=paths[start:start+batch]
  images=[]
  for path in sub:
   with Image.open(path) as img:images.append(pre(img))
  inputs=torch.stack(images)
  output=model.features(inputs)
  output=model.avgpool(output).flatten(1)
  output=model.classifier[:-1](output)
  features.append(output.cpu())
 return torch.cat(features)

def split_files(data,minimum,seed=17):
 rng=random.Random(seed);train=[];test=[];seen_global={}
 for label in LABELS:
  files=sorted(p for p in (data/label).glob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'} and p.stem.lower()!='contact_sheet')
  # Duplicate identical photos must never appear in training & validation.
  unique={}
  for f in files:
   with Image.open(f) as img:
    pixels=ImageOps.exif_transpose(img).convert('RGB').resize((64,64),Image.Resampling.BILINEAR).tobytes()
   digest=hashlib.sha256(pixels).digest()
   if digest in seen_global and seen_global[digest]!=label:raise RuntimeError(f'같은 사진에 서로 다른 분류: {seen_global[digest]} / {label}')
   seen_global[digest]=label;unique[digest]=f
  files=list(unique.values());rng.shuffle(files)
  if len(files)<minimum:raise RuntimeError(f'[{label}] 사진 {len(files)}장; 최소 {minimum}장 (검수된 실제 사진 필요)')
  cutoff=max(5,int(len(files)*.2))
  for f in files[:cutoff]:test.append((f,LABELS.index(label)))
  for f in files[cutoff:]:train.append((f,LABELS.index(label)))
 return train,test

def independent_files(data,holdout,minimum,min_held):
 # Perceptual dHash catches resized copies and many near-identical photographs.
 # Different object instances/sessions still require operator review.
 if data.resolve()==holdout.resolve():raise RuntimeError('학습과 검증 폴더는 별개여야 합니다')
 seen={};seen_hashes=[];partitions=[]
 for root,split,limit in [(data,'train',minimum),(holdout,'holdout',min_held)]:
  rows=[]
  for label in LABELS:
   unique=[]
   for f in sorted((root/label).glob('*')):
    if not f.is_file() or f.suffix.lower() not in {'.jpg','.jpeg','.png','.webp'} or f.stem.lower()=='contact_sheet':continue
    with Image.open(f) as image:
     image=ImageOps.exif_transpose(image).convert('RGB')
     digest=hashlib.sha256(image.resize((64,64),Image.Resampling.BILINEAR).tobytes()).hexdigest()
     a=np.asarray(image.convert('L').resize((9,8),Image.Resampling.BILINEAR));bits=(a[:,1:]>a[:,:-1]).ravel();ph=int.from_bytes(np.packbits(bits).tobytes(),'big')
    if digest in seen:
     oldsplit,oldlabel=seen[digest]
     if oldsplit!=split or oldlabel!=label:raise RuntimeError(f'중복 사진 {f}: {oldsplit}/{oldlabel} → {split}/{label}')
     continue
    if split=='holdout' and any(oldsplit=='train' and (ph^oldph).bit_count()<=3 for oldph,oldsplit in seen_hashes):raise RuntimeError(f'학습 사진과 매우 유사한 검증 사진: {f}')
    seen[digest]=(split,label);seen_hashes.append((ph,split));unique.append((f,LABELS.index(label)))
   if len(unique)<limit:raise RuntimeError(f'{split}/{label}: {len(unique)}장, 최소 {limit}장 필요')
   rows.extend(unique)
  partitions.append(rows)
 return tuple(partitions)

def evaluate(feats,ys,fc):
 with torch.inference_mode():
  probs=fc(feats).softmax(1);scores,indices=probs.topk(2,dim=1)
  pred=indices[:,0];accepted=(scores[:,0]>=MIN_SCORE)&(scores[:,0]-scores[:,1]>=MIN_MARGIN)
  background=LABELS.index('__background__')
  correct=(pred==ys)&accepted
  # A background image is safe if the game would reject every real object.
  correct[ys==background]=((pred==background)|(~accepted))[ys==background]
 recalls={LABELS[i]:round(correct[ys==i].float().mean().item(),3) for i in range(len(LABELS))}
 accuracy=round(correct.float().mean().item(),3)
 return accuracy,recalls

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--catalog',type=Path,default=HERE/'catalog_all_38.json');parser.add_argument('--data',type=Path,default=HERE/'data');parser.add_argument('--output',type=Path,default=HERE.parent/'models');parser.add_argument('--minimum',type=int,default=20);parser.add_argument('--holdout',type=Path,default=HERE/'holdout');parser.add_argument('--minimum-holdout',type=int,default=10);parser.add_argument('--epoch',type=int,default=55);parser.add_argument('--smoke-test',action='store_true');a=parser.parse_args()
 global LABELS
 LABELS=list(json.loads(a.catalog.read_text()).keys())
 if a.minimum<20 or a.epoch<1 or '__background__' not in LABELS or len(LABELS)<3:raise ValueError('minimum >= 20, epoch >= 1, background and at least two objects required')
 random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.set_num_threads(3)
 # Release requires a physically separate photo collection, not random frames from one capture.
 if a.smoke_test:
  samples,held=split_files(a.data,a.minimum)
 else:
  samples,held=independent_files(a.data,a.holdout,a.minimum,max(10,a.minimum_holdout))
 print('Training samples:',len(samples),'held-out different files:',len(held),flush=True)
 model=models.mobilenet_v3_small(weights=None if a.smoke_test else models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
 if a.smoke_test:print('SMOKE TEST ONLY: random backbone; do not deploy')
 for p in model.parameters():p.requires_grad=False
 model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,len(LABELS))
 model.eval()
 trainx=[];trainy=[];testx=[];testy=[]
 for aug in range(3):
  features=get_features(model,[x[0] for x in samples],aug)
  trainx.append(features);trainy.append(torch.tensor([y for _,y in samples]))
 for features in [get_features(model,[x[0] for x in held])]:testx.append(features);testy.append(torch.tensor([y for _,y in held]))
 trainx=torch.cat(trainx).clone();trainy=torch.cat(trainy);testx=torch.cat(testx).clone();testy=torch.cat(testy)
 classifier=model.classifier[-1];classifier.train()
 optimizer=torch.optim.AdamW(classifier.parameters(),lr=.004,weight_decay=.015)
 lossfn=nn.CrossEntropyLoss(label_smoothing=.07)
 for epoch in range(a.epoch):
  perm=torch.randperm(len(trainx))
  for indices in perm.split(64):
   optimizer.zero_grad();loss=lossfn(classifier(trainx[indices]),trainy[indices]);loss.backward();optimizer.step()
 classifier.eval();acc,recall=evaluate(testx,testy,classifier)
 report={'kind':'mobilenet_v3_small_head','trained_on':'smoke test only' if a.smoke_test else 'operator supplied reviewed photos','train_images':len(samples),
   'validation_images':len(held),'min_score':MIN_SCORE,'min_margin':MIN_MARGIN,'holdout_accuracy':acc,'per_class_recall':recall,'class_order':LABELS}
 # Critical: reject models that would be unreliable at a school fair.
 a.output.mkdir(parents=True,exist_ok=True)
 (a.output/'training-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
 if a.smoke_test:print('SMOKE TEST: intentionally no deployable weights');return
 weak=[k for k,v in recall.items() if v<(.9 if k=='__background__' else .8 if k=='jump_rope' else .7)]
 if acc<.80 or weak:
  raise RuntimeError(f'검증 실패: 전체 {acc:.1%}, 미흡한 물품 {weak}, 배경 거부율 {recall["__background__"]:.1%}; 모델을 배포하지 않습니다')
 model.eval()
 fake=torch.randn(1,3,SIZE,SIZE)
 out=a.output/'object-model.pending.onnx'
 torch.onnx.export(model,fake,str(out),input_names=['input'],output_names=['logits'],opset_version=17,dynamo=False)
 import onnx,onnxruntime as ort
 onnx.checker.check_model(str(out))
 session=ort.InferenceSession(str(out),providers=['CPUExecutionProvider'])
 scores=session.run(None,{'input':fake.numpy()})[0]
 with torch.inference_mode():torch_scores=model(fake).numpy()
 diff=float(np.max(np.abs(scores-torch_scores)))
 if diff>.003:out.unlink();raise RuntimeError(f'ONNX 정확도 불일치: {diff}')
 (a.output/'model-manifest.pending.json').write_text(json.dumps({'format':'onnx-mobilenet-v3-small-v1','labels':LABELS,'input_size':224,
   'min_score':MIN_SCORE,'min_margin':MIN_MARGIN,'validation_accuracy':acc,'report_sha256':hashlib.sha256((a.output/'training-report.json').read_bytes()).hexdigest(),'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
 size=out.stat().st_size
 out.replace(a.output/'object-model.onnx')
 (a.output/'model-manifest.pending.json').replace(a.output/'model-manifest.json')
 print(f'PASS, saved {size/1024/1024:.2f}MB ONNX / maximum absolute diff {diff:.5f}')

if __name__=='__main__':main()
