"""Only package a complete, validated 38-class model. Excludes source training photos."""
import json,hashlib,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent.parent
models=root/'models'; manifest=json.loads((models/'model-manifest.json').read_text()); report=json.loads((models/'training-report.json').read_text())
expected=set(json.loads((root/'training/catalog_all_38.json').read_text()))
weights=models/'object-model.onnx'
assert len(expected)==39 and set(manifest['labels'])==expected and len(manifest['labels'])==39, '38 classes + background required'
assert report['class_order']==manifest['labels'] and report['trained_on']=='operator supplied reviewed photos', 'real-photo validation required'
assert report['validation_images']>=390 and report['train_images']>=780
assert manifest['sha256']==hashlib.sha256(weights.read_bytes()).hexdigest()
assert manifest['report_sha256']==hashlib.sha256((models/'training-report.json').read_bytes()).hexdigest()
assert manifest['validation_accuracy']==report['holdout_accuracy'] and report['holdout_accuracy']>=.80
assert set(report['per_class_recall'])==expected
assert all(v >= (.9 if k=='__background__' else .8 if k=='jump_rope' else .7) for k,v in report['per_class_recall'].items())
assert manifest['min_score']==report['min_score']==.68 and manifest['min_margin']==report['min_margin']==.15
runtime=root/'vendor/ort';assert (runtime/'ort.min.js').is_file() and list(runtime.glob('*.wasm')) and list(runtime.glob('*.mjs')), 'local runtime incomplete'
out=root/'AI_Object_Detective_validated_38.zip'
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 html=(root/'index.html').read_text().replace('<script src="js/vendor-loader.js', '<script>window.DETECTIVE_LOCAL_ONLY=true;</script><script src="js/vendor-loader.js',1)
 assert 'DETECTIVE_LOCAL_ONLY=true' in html
 z.writestr('index.html',html)
 for name in ('style.css','atlas.json'):z.write(root/name,name)
 for folder in ('assets','js','vendor/ort'):
  for file in (root/folder).rglob('*'):
   if file.is_file() and file.suffix!='.txt':z.write(file,file.relative_to(root))
 for name in ('object-model.onnx','model-manifest.json','training-report.json'):z.write(models/name,'models/'+name)
print(out)
