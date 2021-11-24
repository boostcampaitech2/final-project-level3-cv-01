# YOLOR

##Version 0.1


data 에 새로운 yaml 파일을 만들고, 

train: /opt/ml/yolor/yolor/img/train/images  
val: /opt/ml/yolor/yolor/img/val/images  
test: /opt/ml/yolor/yolor/img/test/images

처럼 이미지가 있는 디렉터리의 이름을 지정해 줍니다.

train 방법으로는

```
python train.py --batch-size <원하는 배치 사이즈 수> --data <yaml 파일 위치> --weights <yolor-d6.pt 파일 위치> --device 0 --name <runs/train에 저장될 디렉터리 이름> --hyp data/hyp.scratch.1280.yaml --epochs <원하는 에폭 수>
```


예시
```
python train.py --batch-size 8 --img 512 512 --data data/yolor_test.yaml --weights pretrained/yolor-d6.pt --device 0 --name yolor_d6 --hyp data/hyp.scratch.1280.yaml --epochs 7
```
