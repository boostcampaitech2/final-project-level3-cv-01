# YOLOR

### Installation
```
pip install -r requirements.txt
```

### Usage

data 에 새로운 yaml 파일을 만들고, 

train: /opt/ml/yolor/yolor/img/train/images  
val: /opt/ml/yolor/yolor/img/val/images  
test: /opt/ml/yolor/yolor/img/test/images

처럼 이미지가 있는 디렉터리의 이름을 지정해 줍니다.

### Train

- Train

```
python train.py --batch-size <원하는 배치 사이즈 수> --img <원하는 이미지 크기> --data <yaml 파일 위치> --cfg <d6의 cfg 위치->yaml파일 입니다>--weights <yolor-d6.pt 파일 위치> --device 0 --name <runs/train에 저장될 디렉터리 이름> --hyp data/hyp.scratch.1280.yaml --epochs <원하는 에폭 수> --hyp <hyp 파일 위치>
```


- Example
```
python train.py --batch-size 8 --img 512 512 --data data/yolor_test.yaml --cfg models/yolor-d6.yaml --weights pretrained/yolor-d6.pt --device 0 --name yolor-d6 --hyp hyp.scratch.1280.yaml --epochs 100
```

### Test

- Test

```
python test.py --img <이미지 크기> --conf 0.001 --iou <iou 설정> --batch <배치 수> --device 0 --data <yaml 파일 위치> --weights <train에서 나온 pt파일> --name <runs/test에 저장될 이름> --task <val or test>
```

- Example
```
python test.py --img 512 --conf 0.001 --iou 0.5 --batch 32 --device 0 --data data/yolor_test.yaml --weights runs/train/yolor_d63/weights/best_ap.pt --name yolor-d6_val --task test
```
