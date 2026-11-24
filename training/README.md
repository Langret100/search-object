# AI 사물탐정 v3.9 — 전용 38종 모델 제작

**현재 소스 ZIP에는 실사 학습 데이터, ONNX 가중치 및 WASM 런타임이 없습니다.** 이 개발 환경에서는 외부 사진/사전학습 가중치 다운로드가 차단되어 있습니다. 기존 게임은 계속 사용할 수 있으나 전용 엔진 완성판은 아닙니다.

## 실제 사진으로 학습하기
1. 인터넷이 되는 PC에서 Python 3.11, Node 22를 준비합니다. `python -m pip install -r training/requirements.txt`.
2. `python training/collect_commons.py --per-class 80`은 38종 + 배경의 공개 라이선스 사진 후보를 수집합니다. GitHub Actions의 Collect candidate photos 워크플로로도 실행 가능합니다. 후보는 검수 전 상태이며 수집 성공률은 네트워크와 검색 결과에 따라 달라집니다. 사진의 실제 물품, 라이선스, 출처를 확인하고 필요한 경우 현장 사진을 추가합니다. `ATTRIBUTIONS.jsonl` 및 라이선스 의무는 별도 보관하세요. 사진 원본은 배포 ZIP에 넣지 않습니다.
3. 물품 키마다 `training/data/<key>/`에 학습 사진 최소 20장, `training/holdout/<key>/`에 **다른 물품 실체·촬영 장소/세션의 독립 검증 사진** 최소 10장을 넣습니다. 총 39클래스 기준 최소 780장 학습 + 390장 검증입니다. 이 최소치는 높은 정확도를 보증하지 않습니다. 줄넘기는 다른 손잡이, 말린 형태, 늘어진 형태를 포함하고 배경에는 전선, 신발끈, 로프, 빈 교실, 빈 손 등을 넣으세요. 인물 식별이 가능한 사진은 제외하거나 비식별 처리하세요.
4. `python training/train_model.py --data training/data --holdout training/holdout --output models` 실행. 기본 목록은 `catalog_all_38.json`입니다. ImageNet 사전학습 MobileNetV3-Small에 분류층을 전이학습합니다. 검증 사진은 학습에 사용하지 않습니다. 이미지 해시와 유사도 검사로 일부 중복 유입을 막지만 모든 연속 촬영·유사 사진을 검출하지는 않습니다.
5. 게임의 판정 점수 0.68 및 2위와 차이 0.15 기준으로 독립 검증합니다. 전체 판정 성공률 80% 이상, 물품별 70% 이상, 줄넘기 80% 이상, 배경 거부 90% 이상을 통과해야 ONNX 출력합니다. ONNX 수치 일치도도 검사합니다. 현장 카메라 검증은 별도입니다.
6. `bash training/bundle_ort.sh` 후 `python training/package_release.py`. 38종 전체, 검증 리포트/모델 해시 일치 및 런타임 파일 존재를 확인하고 `AI_Object_Detective_validated_38.zip`을 생성합니다. 이 ZIP만 외부 엔진을 차단하는 전용 배포판입니다. ZIP을 풀어 GitHub Pages에 올립니다.

GitHub Actions의 Train and package validated 38-object model 워크플로는 4~6을 수행합니다. 검수된 사진이 저장소에 있어야 합니다. Actions에 업로드된 사진의 개인정보 및 라이선스, 저장소 공개 범위를 확인하세요. 워크플로는 자동 커밋/배포하지 않습니다. 실패하면 배포 ZIP을 만들지 않습니다.

## 검증 범위
- 실제 PyTorch 연산을 무작위 테스트 픽셀 3개 클래스에 대해 수행: 학습 절차 실행 확인, 의도적으로 배포 모델 미생성.
- 38종 목록 일치, 독립 데이터 분리 및 확대 복사 중복 거부, 가중치 누락 시 배포 차단 확인.
- Chromium 인라인 HTML/JS 가상 실행(모델 요청 404 모의): FHD 및 390x844에서 설정/38종 목록과 오류 안내, 외부 HTTP 요청 미발생 확인. 실제 호스팅, 웹캠, ONNX 실행 및 FPS는 미검증.
- 같은 사이트에 포함된 모델은 Worker에서 SHA-256 검증 후 초기 추론 1회를 실행하고 준비 상태로 전환합니다. 전용 배포판에는 외부 엔진 자동 전환을 허용하지 않습니다.
