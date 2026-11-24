"""Collect Creative Commons/PD photographs as UNREVIEWED candidate training images.

Photos are not shipped or silently treated as ground-truth. Review contact sheets
and LICENSES.jsonl before promotion. External HTTP requires network access.
"""
from __future__ import annotations
import argparse, hashlib, json, io, random, re, time
from pathlib import Path
import requests
from PIL import Image, ImageOps, ImageDraw

BASE='https://commons.wikimedia.org/w/api.php'
HERE=Path(__file__).parent
UA='AI-Object-Detective-school-event-dataset/0.1 (educational photograph review; no bulk redistribution)'

def accepted(meta):
    cc=(meta.get('LicenseShortName') or {}).get('value','')
    cc=re.sub(r'<[^>]*>','',cc).strip().lower()
    return any(w in cc for w in ['cc by','cc0','public domain','pd-']) and not any(w in cc for w in ['nc','nd','non-commercial'])

def collect(cat,queries,dest,limit,session):
    saved=0;seen=set();records=[]
    dest.mkdir(parents=True,exist_ok=True)
    saved=sum(1 for f in dest.glob('*.jpg') if f.name!='CONTACT_SHEET.jpg')
    for q in queries:
      if saved>=limit:break
      params={'action':'query','format':'json','generator':'search','gsrsearch':q+' filetype:bitmap','gsrnamespace':6,'gsrlimit':50,'prop':'imageinfo','iiprop':'url|size|mime|extmetadata','iiurlwidth':600}
      try:
       response=session.get(BASE,params=params,timeout=22);response.raise_for_status();data=response.json()
       if 'error' in data: raise RuntimeError(data['error'])
      except Exception as ex:
       print(f'[{cat}] search failed: {ex}');continue
      for page in data.get('query',{}).get('pages',{}).values():
       if saved>=limit:break
       item=(page.get('imageinfo') or [{}])[0]
       if item.get('mime') not in ('image/jpeg','image/png','image/webp'):continue
       if not accepted(item.get('extmetadata') or {}):continue
       if min(item.get('width',0),item.get('height',0))<320:continue
       if item['width']/item['height']>3.5 or item['height']/item['width']>3.5:continue
       url=item.get('thumburl') or item.get('url')
       if not url or url in seen:continue
       seen.add(url)
       try:
        response=session.get(url,timeout=25);response.raise_for_status();data=response.content
        if len(data)>5_000_000 or len(data)<2000:continue
        im=Image.open(io.BytesIO(data));im=ImageOps.exif_transpose(im).convert('RGB')
        if min(im.size)<220:continue
        # Avoid storing a person's location data in extracted metadata.
        digest=hashlib.sha256(im.resize((50,50)).tobytes()).hexdigest()[:16]
        filename=f'{digest}.jpg'; target=dest/filename
        if target.exists():continue
        im.thumbnail((512,512));im.save(target,'JPEG',quality=85,optimize=True)
        m=item.get('extmetadata') or {}
        records.append({'file':filename,'source':item.get('descriptionurl',url),'author':m.get('Artist',{}).get('value',''),
           'license':m.get('LicenseShortName',{}).get('value',''), 'license_url':m.get('LicenseUrl',{}).get('value',''),
           'source_file':item.get('url'),'search':q,'status':'UNREVIEWED'})
        saved+=1
       except Exception as ex: print(f'[{cat}] image failed: {str(ex)[:100]}')
      time.sleep(.5)
    if records:
      with (dest/'ATTRIBUTIONS.jsonl').open('a') as file:
       for rec in records:file.write(json.dumps(rec,ensure_ascii=False)+'\n')
    print(f'{cat}: {saved} candidate photos; review required')

def sheet(folder):
    photos=sorted(p for p in folder.glob('*.jpg') if p.name!='CONTACT_SHEET.jpg')[:80]
    if not photos:return
    cols=6;cell=(140,134);canvas=Image.new('RGB',(cols*cell[0], ((len(photos)+cols-1)//cols)*cell[1]),'white')
    draw=ImageDraw.Draw(canvas)
    for i,path in enumerate(photos):
      img=Image.open(path).convert('RGB');img.thumbnail((132,110));x=(i%cols)*cell[0];y=(i//cols)*cell[1]
      canvas.paste(img,(x+(cell[0]-img.width)//2,y));draw.text((x+4,y+112),path.stem[:16],fill='black')
    canvas.save(folder/'CONTACT_SHEET.jpg',quality=88)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(HERE/'candidates'));p.add_argument('--per-class',type=int,default=60);p.add_argument('--catalog',type=Path,default=HERE/'catalog_all_38.json');a=p.parse_args()
    catalog=json.loads(a.catalog.read_text());root=Path(a.data);root.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.headers['User-Agent']=UA
    for key,terms in catalog.items():collect(key,terms,root/key,a.per_class,session);sheet(root/key)
    print('IMPORTANT: Results are candidate photos, not validated labels. Review and copy accepted images into training/data/<key>/ before training.')
