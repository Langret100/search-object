#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p vendor/ort
npm install --no-save --no-package-lock --prefix training/ort-package onnxruntime-web@1.20.1
cp training/ort-package/node_modules/onnxruntime-web/dist/ort.min.js vendor/ort/ort.min.js
cp training/ort-package/node_modules/onnxruntime-web/dist/*.wasm vendor/ort/
cp training/ort-package/node_modules/onnxruntime-web/dist/*.mjs vendor/ort/
# Model and WASM now load exclusively from GitHub Pages (same origin).
