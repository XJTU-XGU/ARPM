# Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation: OSDA Experiments

Code for paper "Xiang Gu, Xi Yu, Yan Yang, Jian Sun, Zongben Xu, Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation".

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

## Domain ID:

**Office-Home**: art, clipart, product, real_world ==> 0,1,2,3 <br>

## Training

Office-Home:

```shell
python train.py --dset office_home --s 0 --t 1
```

## Results

We run the code on a single Tesla V-100 GPU. The results are as follows.

Office-Home:

| seed    | AC       | AP       | AR       | CA       | CP       | CR       | PA       | PC       | PR       | RA       | RC       | RP       | Avg      |
| ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 2019    | 63.4     | 75.6     | 80.9     | 67.5     | 71.4     | 74.9     | 67.2     | 62.1     | 77.1     | 73.9     | 65.4     | 81.6     | **71.7** |
| 2021    | 64.0     | 76.6     | 81.1     | 67.5     | 73.1     | 73.7     | 68.2     | 62.2     | 76.6     | 74.0     | 65.7     | 80.5     | **71.9** |
| 2023    | 64.1     | 75.9     | 79.9     | 67.0     | 71.7     | 75.3     | 67.6     | 60.9     | 76.3     | 73.5     | 65.1     | 81.4     | **71.6** |
| **Avg** | **63.8** | **76.0** | **80.6** | **67.3** | **72.1** | **74.6** | **67.7** | **61.7** | **76.7** | **73.8** | **65.4** | **81.2** | **71.7** |


## Contact：

If you have any problem, feel free to contect xianggu@stu.xjtu.edu.cn.