from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
import method
import evaluate
import copy
import torch
import numpy as np
import argparse
import os
import logging
import time

def main(args):
    NORM_IMGNET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    te_transforms_imgnet = transforms.Compose([transforms.Resize(256),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize(*NORM_IMGNET)
                                               ])
    dataset = ImageFolder(args.data_dir,te_transforms_imgnet)
    data_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=4)

    net = resnet18(pretrained=True).cuda()
    logging.info("Model loaded")

    # res = evaluate.test(data_loader, net)[0] * 100
    # logging.info(f'Error Before Adaptation: {res:.1f}')

    logging.info(f"\n\nBegin {args.method}")
    results = []
    for i in range(args.num_runs):
        logging.info(f"\nThe {i}-th run")
        model = copy.deepcopy(net)
        model = method.configure_model(model)
        params, param_names = method.collect_params(model)
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9)

        # model adaptation
        '''The best coeff for tent is 10, and for tpm is 100.'''
        TTA_model = method.TTA(model, optimizer, steps=1,method=args.method,coeff=10 if args.method=="tent" else 100)

        adaptation_result = evaluate.test(data_loader, TTA_model)[0] * 100
        logging.info(f'Error after Adaptation: {adaptation_result:.1f}')
        results.append(adaptation_result)
    logging.info(f"\nmean error of {args.num_runs} runs: {np.mean(results):.1f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default="tent",choices=["tpm","tent"])
    parser.add_argument('--data_dir', default='../imagenet-r',help="root to imagenet-r")
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--lr', default=1e-4,type=float)
    parser.add_argument('--num_runs', default=10)
    parser.add_argument('--gpu', default="0")

    args =parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    os.makedirs(f"log/{args.method}",exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(message)s')

    fh = logging.FileHandler(f'log/{args.method}/{time.time()}.txt')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    for arg in vars(args):
        logger.info(f'{arg}: {getattr(args, arg)}')

    main(args)





