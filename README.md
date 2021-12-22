# 🙌 PM ( Personal Mobility ) 위법행위 감지 시스템   

## ❓ 등장 배경
> 최근 3년간 도시의 거리에 생긴 가장 큰 변화는 공유 킥보드를 필두로 한 다양한 이동수단의 등장입니다.</br>
실제로 공유 킥보드 서비스가 도입된 뒤, 서비스 이용 건수가 2020년 상반기 대비 하반기에 60%가 증가 할 정도로 PM ( Personal Molbility ) 시장은 급격도로 확산하고 있습니다. 하지만 이에 따른 여러 문제들도 같이 발생하고 있습니다. 이용객이 증가한 만큼 그에 따른 사고들도 많이 발생하고 있으며, "킥라니( 킥보드 + 고라니 )" 라는 신조어 까지 등장할 정도로 운전자 혹은 보행자에게 위협이 되고 있습니다. </br>
그래서 정부는 이와 관련된 법과 제도들을 만들고 시행하고 있지만 잘 지켜지지 않아 아직도 많은 사고들이 잇따르고 있습니다. 그래서 저희 팀은 **<u>킥보드 위법행위 ( 헬멧 미착용, 동승 ) 탐지를 자동화</u>** 하여 사용자들이 좀 더 안전한 전동 킥보드 사용을 도모하려고 합니다.
</br>

## 🙄문제 정의
> 이미지, 동영상, 실시간 영상에서 Object Detection을 이용하여 **<u>사용자의 킥보드 사용 위법행위 ( 헬멧 미착용, 동승 )를 탐지하여 이용자의 안전한 주행을 유도한다.</u>**
</br>

## 👩‍🏫개발 환경
- **GPU환경** : V100 서버, Google Cloud Platform(GCP) 서버
- **팀 협력 Tools** : Notion, Weights&Biases, Github, Google Drive
- **개발 Tools** : VScode, Jupyter Lab
- **라이브러리 및 프레임워크** : PyTorch, Streamlit, FastAPI
- **Annotation Tools** : Supervisely
</br>

## 👨‍🏫평가 Metric
<img src="https://user-images.githubusercontent.com/64246382/137627632-404ecf72-6244-4128-ae3c-607e8df2a314.PNG" width="600" height="300"> 
</br>

## 🛴데이터셋
![라벨링](https://user-images.githubusercontent.com/64246382/147076690-caf366bb-8c75-43e3-aefd-c87746b4199d.PNG)
### 데이터 버전 관리
| Version | 데이터 수 | Train 데이터 수 | Val 데이터 수 | Test 데이터 수 | 세부사항 |
| --- | --- | --- | --- | --- | --- |
| 1.0 | 1128 | 790 | 225 | 113 | 초기 데이터셋 |
| 2.0 | 1321 | 960 | 240 | 121 | Class 불균형 해소를 위한 이미지 추가 |
| 2.1 | 1364 | 1003 | 240 | 121 | Edge case 해결을 위해 합성 이미지 추가 |
| 2.2 | 1364 | 995 | 245 | 124 | 데이터셋(Train, Val, Test) 재조정 |
| 3.0 | 1447 | 1078 | 245 | 124 | Box 없는 배경 이미지 추가 |

### Input
- PM으로 주행중인 사람이 담긴 이미지 or 영상 데이터
- 크롤링, 직접 촬영으로 모은 데이터
- 이미지의 다양성, 저작권, 초상권 고려
- Supervisely 툴을 이용하여 직접 Annotation
- 라벨 : 0번(헬멧O, 혼자O), 1번(헬멧X, 혼자O), 2번(헬멧O,혼자X), 3번(헬멧X,혼자X)
- 태그 : Orientation, Blur, Out of Focus, Low quality
- 형식 : JSON, YOLO

### Output
- 이용자의 헬멧 착용여부와 동승여부 분류
- PM과 이용자를 함께 detection하고 분류
- 휴대폰 사용여부, 연령까지 확장 가능
