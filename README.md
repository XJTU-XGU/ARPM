# Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation

Code for paper "Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu, Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation".

This is the extended version of the conference paper ["Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu, Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation. NeurIPS. 2021"](https://github.com/XJTU-XGU/Adversarial-Reweighting-for-Partial-Domain-Adaptation). The extended version is more stable and more effective.
## Prerequisites:
python==3.9.12 <br>
torch==2.0.1 <br>
torchaudio==2.0.2 <br>
torchvision==0.15.2 <br>
numpy==1.25.2 <br>
cvxpy==1.3.2 <br>
tqdm==4.66.1 <br>
Pillow==10.0.0 <br>
scikit-learn==1.3.0 <br>
torch_ema==0.3

## Datasets:
Download the datasets of <br>
[VisDA-2017](http://ai.bu.edu/visda-2017/) <br>
[DomainNet](http://ai.bu.edu/M3SDA/) <br>
[Office-Home](https://www.hemanthdv.org/officeHomeDataset.html) <br>
[Office](https://www.cc.gatech.edu/~judy/domainadapt/) <br>
[ImageNet](https://www.image-net.org/) <br>
[Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/) <br>
and put them into the folder "./data/" and modify the path of images in each '.txt' under the folder './data/'. Note the full list of ImageNet (imagenet.txt) is too big. Please download it [here](https://drive.google.com/file/d/1aZGNVO4-6yl7L0ulinDPxo11-RDozeBP/view?usp=sharing) and put it into './data/imagenet_caltech/'. 

## Domain ID:
**VisDA-2017**: train (synthetic), validation (real) ==> 0,1 <br>
**DomainNet**: clipart, painting, real, sketch ==> 0,1,2,3 <br>
**Office-Home**: art, clipart, product, real_world ==> 0,1,2,3 <br>
**Office**: amazon, dslr, webcam  ==> 0,1,2 <br>
**ImageNet-Caltech**: imagenet, caltech ==> 0,1 <br>
## Training
VisDA-2017:
```
python main.py --dset visda-2017 --s 0 --t 1
```
DomainNet:
```
python main.py --dset domainnet --s 0 --t 1
```
Office-Home:
```
python main.py --dset office_home --s 0 --t 1
```
Office:
```
python main.py --dset office --s 0 --t 1
```
ImageNet-Caltech:
```
python main.py --dset imagenet_caltech --s 0 --t 1
```

## Results
We run the code on a single Tesla V-100 GPU. The results are as follows.

Office:

| seed    | A2D      | A2W      | D2A      | D2W      | W2A      | W2D       | Avg      |
| ------- | -------- | -------- | -------- | -------- | -------- | --------- | -------- |
| 2019    | 99.4     | 99.3     | 96.7     | 100.0    | 96.8     | 100.0     | **98.7** |
| 2021    | 100.0    | 99.7     | 96.4     | 100.0    | 96.6     | 100.0     | **98.8** |
| 2023    | 99.4     | 99.3     | 96.6     | 99.7     | 96.9     | 100.0     | **98.7** |
| **Avg** | **99.6** | **99.4** | **96.6** | **99.9** | **96.8** | **100.0** | **98.7** |

Office-Home:

| seed    | A2C      | A2P      | A2R      | C2A      | C2P      | C2R      | P2A      | P2C      | P2R      | R2A      | R2C      | R2P      | Avg      |
| ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 2019    | 67.9     | 90.1     | 92.8     | 78.4     | 81.2     | 85.0     | 80.0     | 67.5     | 89.5     | 87.3     | 72.5     | 89.2     | **81.8** |
| 2021    | 67.4     | 85.7     | 90.8     | 76.2     | 85.8     | 88.5     | 81.5     | 71.0     | 88.8     | 85.3     | 69.2     | 88.9     | **81.6** |
| 2023    | 69.5     | 87.7     | 93.0     | 78.8     | 86.8     | 85.4     | 81.7     | 69.1     | 90.2     | 86.1     | 68.2     | 89.2     | **82.1** |
| **Avg** | **68.3** | **87.8** | **92.2** | **77.8** | **84.6** | **86.3** | **81.1** | **69.2** | **89.5** | **86.2** | **70.0** | **89.1** | **81.8** |

VisDA-2017:

| seed    | 2019 | 2021 | 2023 | Avg      |
| ------- | ---- | ---- | ---- | -------- |
| **S2R** | 92.2 | 93.9 | 93.6 | **93.2** |

ImageNet-Caltech:

| seed    | I2C      | C2I      | Avg      |
| ------- | -------- | -------- | -------- |
| 2019    | 84.3     | 87.1     | **85.7** |
| 2021    | 84.6     | 87.0     | **85.8** |
| 2023    | 85.0     | 87.2     | **86.1** |
| **Avg** | **84.6** | **87.1** | **85.9** |

DomainNet:

| seed    | C2P      | C2R      | C2S      | P2C      | P2R      | P2S      | R2C      | R2P      | R2S      | S2C      | S2P      | S2R      | Avg      |
| ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 2019    | 65.3     | 77.0     | 66.5     | 78.3     | 84.3     | 81.9     | 86.7     | 77.9     | 78.5     | 66.8     | 63.8     | 70.9     | **74.8** |
| 2021    | 69.1     | 81.0     | 65.9     | 77.4     | 84.4     | 82.2     | 86.5     | 78.1     | 77.7     | 60.1     | 65.7     | 72.5     | **75.1** |
| 2023    | 69.2     | 81.4     | 66.6     | 79.4     | 83.7     | 81.5     | 86.2     | 78.0     | 79.6     | 60.7     | 65.0     | 71.8     | **75.3** |
| **Avg** | **67.9** | **79.8** | **66.3** | **78.4** | **84.1** | **81.9** | **86.5** | **78.0** | **78.6** | **62.5** | **64.8** | **71.7** | **75.0** |

## Reference code:

https://github.com/thuml/CDAN <br>
https://github.com/tim-learn/BA3US <br>
https://github.com/XJTU-XGU/RSDA

## Contact：
If you have any problem, feel free to contect xianggu@stu.xjtu.edu.cn.
