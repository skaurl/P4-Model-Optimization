# P4-Model-Optimization

# 전체 개요 설명

최근들어 분야를 막론하고 인공지능 기술은 사람을 뛰어넘은 엄청난 성능을 보여주고 있고, 때문에 여러 산업에서 인공지능을 이용해 그동안 해결하지 못 했던 문제들을 풀려는 노력을 하고 있습니다.
대표적인 예로는 수퍼빈의 수퍼큐브가 있습니다. 수퍼큐브는 수퍼빈에서 만든 인공지능 분리수거 기계로 사람이 기계에 캔과 페트병을 넣으면 내부에서 인공지능을 통해 재활용이 가능한 쓰레기인지를 판단해 보관해주는 방식입니다. 간단한 인공지능을 이용해 그동안 힘들었던 분리수거 문제를 해결한 것입니다. 그렇다면 수퍼큐브를 만들기 위해 필요한 인공지능은 무엇일까요? 당연히 들어온 쓰레기를 분류하는 작업일 것입니다. 하지만 분류만 잘 한다고 해서 사용할 수 있는 것은 아닙니다. 로봇 내부 시스템에 탑재되어 즉각적으로 쓰레기를 분류할 수 있어야만 실제로 사용이 될 수 있습니다.
이번 프로젝트를 통해서는 분리수거 로봇에 가장 기초 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 작고 계산량이 적은 모델을 만들어볼 예정입니다.
https://gscaltexmediahub.com/story/energy-plus-hub-supercube/

# 평가 방법

모델 최적화의 최종 목표는 어느 정도 이상의 성능을 유지하며 크기가 작은 모델과 빠른 추론 속도를 갖는 모델을 만드는 것입니다. 때문에 이상적인 최종 score는 모델의 성능, 크기, 속도를 모두 아우르는 score일 것입니다. 하지만 대회 특성상 모델의 크기와 속도를 측정하는 데에는 한계가 매우 크다고 느껴져서 이러한 요소들을 score에 반영하지 못 하게 되었습니다. 대신 모델의 연산량을 계산하는 MACs를 통해 모델 최적화 부분의 성능을 반영하고자 합니다.
따라서 최종 score는 모델의 f1-score와 모델의 MACs를 통해 계산될 예정입니다. 본 대회는 모델 최적화 대회이기 때문에 모델의 f1 score보다는 MACs를 낮추는 데에 조금 더 비중을 두려고 합니다. 때문에 모델의 f1 score가 기준 모델의 f1 score를 넘어섰다면 그 이후로는 f1 score의 비중을 줄여 MACs를 통해 좋은 점수를 받을 수 있도록 score를 설계하였습니다. 모델 평가를 위한 식과 평가 예시는 아래 첨부된 이미지를 통해 더 쉽게 이해하실 수 있습니다.
- f1-score: imbalanced data에 대해서 많이 사용되는 분류기 성능 지표
    - (0과 1사이의 값, 높을수록 좋음)
    - 자세한 설명: https://nittaku.tistory.com/295
- MACs: 모델의 (덧셈 + 곱셈)의 계산 수 (낮을수록 좋음)
f1-score를 통해 이미지 분류 성능을 측정하고 MACs을 통해 모델의 계산량을 측정할 예정입니다.
f1-score가 기준 모델의 f1-score를 넘을 시 f1-score의 비중을 반으로 낮춰 MAC의 중요도를 높일 예정입니다.
때문에 어느 정도의 f1-score 이상 부터는 MACs로 인해 score 변동이 크도록 설계하였습니다.

추가사항:  f1-score 계산에 성능적 하한을 두어서 채점된 f1-score가 0.5 미만일 경우, f1 항의 score는 1.0으로 고정됩니다.즉 f1-score를 0.0으로 반영합니다. 너무 낮은 f1-score를 방지하기 위함입니다. 이는 public, private 채점 시 모두 반영되므로, public에서는 0.5 이상이었던 f1-score가 private에서는 0.5보다 작은 경우가 있을 수 있으니, 주의해주세요!

<p align="center"><img src="https://user-images.githubusercontent.com/55614265/119596464-28ed4e80-be1a-11eb-9656-ac62d18fa909.png"/></p>
<p align="center"><img src="https://user-images.githubusercontent.com/55614265/119596466-2b4fa880-be1a-11eb-9255-d42687d0cfb1.png"/></p>

예시에서 낮은 score가 높은 등수를 갖게 됩니다.
즉, 예시모델 1이 가장 좋은 모델을 만들었다고 판단됩니다.
기준 대비 MACs가 낮고, f1 score가 높으면 음수의 score도 가능합니다. 낮은 score가 높은 성적임에는 변함이 없다는 점 참고해주세요!

추가 내용(5/27일 업데이트)
1. 상위권 팀들에 대해서는 코드를 검증하고 학습을 재수행하여 현재 제공된 데이터 만으로 제출 성능이 재현가능한지 확인할 예정이니 참고 바랍니다. 따라서 pretrained에 활용하는 dataset 보관 및 전체과정을 재현가능하도록 코드 관리에 유의 부탁드립니다.
2. 모델 관련 외부 라이브러리 사용 금지
    - MACs 측정 라이브러리의 제약으로 인해 내려진 결정입니다.
    - 이미지 변환(PIL, opencv)등 다른 라이브러리는 사용가능합니다.
    - torchvision model의 pretrained는 사용가능합니다.
    - module을 직접 구현하고, pretrained의 weight를 copy하여 사용하는 것은 허용합니다.
3. MACs를 측정하는 라이브러리에 hook이 걸리지 않는 형태의 구현을 모델에 포함하는 것을 제한합니다.
    - nn.Conv2d를 상속받고, torch.functional.conv2d를 사용하여 Conv 연산 자체를 재정의하는 모듈 사용을 제한합니다.(그밖의 functional.linear 등등 모두 동일하게 제한합니다.)
    - 허용 가능 예시
        - https://github.com/pytorch/vision/blob/master/torchvision/models/squeezenet.py#L15 // nn.Modules 와 nn.Conv2d로 Layer 모듈을 구성함
        - https://github.com/rwightman/pytorch-image-models/blob/23c18a33e4168dc7cb11439c1f9acd38dc8e9824/timm/models/ghostnet.py#L46
    - 허용 불가 예시
        - https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/std_conv.py#L14 // nn.Conv2d를 상속받고, Function conv2d를 사용
4. 모델 경량화라는 토픽의 특성상 기존에 수행해온 대회와 약간의 결이 다른 점이 있어 추가 공지드립니다.이번 대회는 기존과 다르게, 기업 간 또는 기업 내 프로젝트를 수행하는 관점으로 대회가 설계되었습니다.“특정 성능을 만족하는 선에서 연산 횟수가 최소인 경량 모델을 찾자“의 프로젝트를 모사하고자 하였는데, 이러한 관점에서 아래의 항목을 명확히하고자 합니다.
    - 1회의 inference로 생성된 결과만을 제출(앙상블 등의 여러번 inference를 수행 취합하는 방법을 제한, 즉 단일 모델 1회 추론 결과를 제출)
    - Test셋은 psuedo labeling 등 모든 직, 간접적인 활용을 제한(아예 제공되지 않았다고 가정해주시면 감사하겠습니다.)

# 학습 데이터 개요

- Segmentation / Object detecion task 를 풀기 위해 제작된 COCO format의 재활용 쓰레기 데이터인 TACO 데이터를 사용합니다.
- 단순한 Classification 문제로 설정하기 위해 TACO 데이터셋의 Bounding box를 crop 한 데이터를 사용했습니다.

```
├─train
│  ├─Battery
│  ├─Clothing
│  ├─Glass
│  ├─Metal
│  ├─Paper
│  ├─Paperpack
│  ├─Plastic
│  ├─Plasticbag
│  └─Styrofoam
└─val
    ├─Battery
    ├─Clothing
    ├─Glass
    ├─Metal
    ├─Paper
    ├─Paperpack
    ├─Plastic
    ├─Plasticbag
    └─Styrofoam
```

train 폴더와 val 폴더에 들어있는 데이터는 모두 train 데이터로 필요에 따라 섞어서 사용하셔도 됩니다. train data는 9개의 카테고리로 분류되어 있고 총 32,599장의 .jpg format 이미지를 보유하고 있습니다. EDA 과정을 통해 알 수 있겠지만 현재 class별 데이터 수 차이가 상당히 크기 때문에 class imbalance문제를 잘 해결하는 것도 하나의 좋은 점수를 얻는 방법으로 볼 수 있습니다. 또한 이미지의 width와 height가 모두 제각각이기 때문에 data tranform 작업에서 각기 다른 크기의 이미지를 어떻게 처리할 지 고민하시는 작업도 점수에 유의미한 영향을 줄 것 같습니다.
class 별 데이터 수와 sample image에 대한 정보는 아래 이미지를 통해 확인하실 수 있습니다.

![pasted image 0](https://user-images.githubusercontent.com/55614265/119746787-3bc05b80-becc-11eb-94f6-59eb3bf044f2.png)

train data를 무작위로 샘플링해서 그린 결과입니다.

![pasted image 0 (1)](https://user-images.githubusercontent.com/55614265/119746800-41b63c80-becc-11eb-9188-674a00a2fa9d.png)

# 평가 데이터 개요

```
├─test
│  └─NoLabel
```

평가 데이터는 test/NoLabel 폴더 안에 있고 label 정보가 존재하지 않습니다. 
평가데이터의 수는 총 8,159개로 그 중 4083개를 private으로, 4076개를 public으로 사용합니다.

대회 제출 관련 주의사항
- 비공개적으로 다른 참가자 팀과 코드 혹은 데이터를 공유하는 것은 허용되지 않습니다. (Private Sharing 금지)
- 다만 공개적으로 토론 게시판 등을 통해 공유하는 것은 허용되니 참고 부탁드립니다.
- 또한 상위권은 제출한 코드 검수가 진행될 예정입니다.
