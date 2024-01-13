# Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation: UniDA Experiments

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
python main.py --dset office_home --s 0 --t 1
```

## Results

We run the code on a single Tesla V-100 GPU. The results are as follows.

Office-Home:

| seed    | AC       | AP       | AR       | CA       | CP       | CR       | PA       | PC       | PR       | RA       | RC       | RP       | Avg      |
| ------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- |
| 2019    | 64.9     | 81.2     | 89.7     | 73.6     | 72.2     | 84.3     | 73.3     | 67.2     | 85.1     | 79.7     | 70.3     | 85.7     | **77.3** |
| 2021    | 65.7     | 81.6     | 88.9     | 73.2     | 74.5     | 83.4     | 77.1     | 67.6     | 84.9     | 78.5     | 69.5     | 84.8     | **77.5** |
| 2023    | 65.1     | 80.8     | 89.5     | 72.7     | 73.6     | 84.1     | 74.2     | 67.1     | 84.5     | 78.6     | 71.2     | 84.9     | **77.2** |
| **Avg** | **65.2** | **81.2** | **89.4** | **73.2** | **73.4** | **83.9** | **74.9** | **67.3** | **84.8** | **78.9** | **70.3** | **85.1** | **77.3** |


## Contact：

If you have any problem, feel free to contect xianggu@stu.xjtu.edu.cn.