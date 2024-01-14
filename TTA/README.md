# Adversarial Reweighting with α-Power Maximization for Partial Domain Adaptation: TTA Experiments

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

## Evaluation

Download the [ImageNet-R](https://github.com/hendrycks/imagenet-r) dataset first.
For test entropy minimization (tent), run

```shell
python main.py --method tent --data_dir root/to/imagenet-r 
```

For test $\alpha$-power maximization (tpm), run

```shell
python main.py --method tpm --data_dir root/to/imagenet-r
```

## Results

We run the code on a single Tesla V-100 GPU. The results are as follows.

|                          | tent | tpm  |
| ------------------------ | ---- | ---- |
| Average error of 10 runs | 63.2 | 61.7 |


## Contact：

If you have any problem, feel free to contect xianggu@stu.xjtu.edu.cn.