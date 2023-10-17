import argparse
import copy
import os
import random
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import data_list
import get_weight_net
import loss
import lr_schedule
import network
import utils
from utils import get_features


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def image_test_ten(resize_size=256, crop_size=224,tencrop=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
    return transforms.Compose([
            transforms.Resize((resize_size,resize_size)),
            transforms.TenCrop(crop_size),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),]
            )


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train(args):
    ## prepare data
    train_bs, test_bs = args.batch_size, args.batch_size * 2

    if args.sampler == "subset_sampler":
        source_base_dataset_train = data_list.ImageList(open(args.s_dset_path).readlines(),
                                                        transform=image_train(), root=args.root)
        source_base_dataset_test = data_list.ImageList(open(args.s_dset_path).readlines(),
                                                       transform=image_test(), root=args.root)

    dsets = {}
    dsets["source"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_train(),return_index=True,root=args.root)
    dsets["target"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_train(),return_index=True,root=args.root)
    dsets["test"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test(),root=args.root)
    dsets["source_val"] = data_list.ImageList(open(args.s_dset_path).readlines(), transform=image_test(),root=args.root)
    dsets["test_ten"] = data_list.ImageList(open(args.t_dset_path).readlines(), transform=image_test_ten(), root=args.root)

    dset_loaders = {}
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs, shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, shuffle=False, num_workers=args.worker)
    dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=test_bs, shuffle=False, num_workers=args.worker)
    dset_loaders["test_ten"] = DataLoader(dsets["test_ten"], batch_size=10, shuffle=False, num_workers=args.worker)

    ## prepare model
    if "ResNet" in args.net:
        params = {"resnet_name": args.net, "bottleneck_dim": args.bottleneck_dim,
                  'class_num': args.class_num,"radius":args.radius,"normalize_classifier":args.normalize_classifier}
        base_network = network.ResNetFc(**params)

    base_network = base_network.cuda()

    ## initialize classification layer by pca
    from pca_init_head import init_head
    base_network.train(False)
    if args.sampler == "subset_sampler":
        indexes = np.random.permutation(len(source_base_dataset_test))[:train_bs * 2000]
        dsets["source_val"] = data_list.SubDataset(source_base_dataset_test, indexes)
        dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=test_bs, shuffle=False,
                                                num_workers=args.worker)
    init_head(base_network,dset_loaders["source_val"],dset_loaders["test"],pretrain_head=False)

    parameter_list = base_network.get_parameters()

    ## set optimizer
    optimizer_config = {"type": torch.optim.SGD, "optim_params":
        {'lr': args.lr, "momentum": 0.9, "weight_decay": 5e-4, "nesterov": True},
                        "lr_type": "inv", "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": 0.75}
                        }
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))

    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    best_acc = 0
    weight_learner = get_weight_net.WeightLearner(input_dim=args.bottleneck_dim)

    #################################################################################################################
    ## building feature bank and score bank
    loader = dset_loaders["test"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, base_network.bottleneck.in_features)
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    base_network.eval()
    with torch.no_grad():
        print("Building feature bank...")
        iter_test = iter(loader)
        for i in tqdm.trange(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            indx = data[-1]
            inputs = inputs.cuda()
            feature, _, output = base_network(inputs.cuda())
            output_norm = torch.nn.functional.normalize(feature)
            outputs = torch.nn.Softmax(-1)(output)
            fea_bank[indx] = output_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  # .cpu()
    #################################################################################################################

    print("Training begins.")
    for i in range(args.max_iterations + 1):
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        ## test
        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            base_network.train(False)
            temp_acc, pseudo_labels = utils.image_classification_test(dset_loaders, base_network,tencrop=False,
                                                                      per_class=True if args.dset=="visda-2017" else False,
                                                                      log_file=args.out_file if args.dset=="visda-2017" else None)
            if best_acc < temp_acc:
                best_acc = temp_acc
                best_model = base_network.state_dict()
            log_str = "\n {} iter: {:05d}, precision: {:.5f}, best_acc: {:.5f} \n".format(args.name,i, temp_acc, best_acc)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)

        ## update weight, loader
        if args.sampler == "weighted_sampler":
            if i % args.weight_update_interval == 0 and i>0:
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, _ = get_features(dset_loaders["test"], base_network)
                if len(all_source_features) >=20000:
                    weights = weight_learner.get_weight_large(all_source_features, all_target_features, args.rho)
                else:
                    weights = weight_learner.get_weight(all_source_features, all_target_features, args.rho)
                weights = torch.Tensor(weights[:])
                dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                                    sampler=WeightedRandomSampler(weights, num_samples=len(weights),
                                                                                  replacement=True),
                                                    num_workers=args.worker, drop_last=True)
        if args.sampler == "subset_sampler":
            if i % args.weight_update_interval == 0 and i > 0:
                indexes = np.random.permutation(len(source_base_dataset_test))[:train_bs * 2000]
                dsets["source"] = data_list.SubDataset(source_base_dataset_train, indexes)
                dsets["source_val"] = data_list.SubDataset(source_base_dataset_test, indexes)
                dset_loaders["source_val"] = DataLoader(dsets["source_val"], batch_size=test_bs, shuffle=False,
                                                        num_workers=args.worker)
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, _ = get_features(dset_loaders["test"], base_network)
                weights = weight_learner.get_weight(all_source_features, all_target_features,args.rho)
                weights = torch.Tensor(weights[:])
                dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                                    sampler=WeightedRandomSampler(weights, num_samples=len(weights),
                                                                                  replacement=True),
                                                    num_workers=args.worker, drop_last=True)
        if args.sampler == "uniform_sampler":
            if i == 0:
                weights = torch.ones(len(dsets["source_val"]))
            elif i % args.weight_update_interval == 0:
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, _ = get_features(dset_loaders["test"], base_network)
                weights = weight_learner.get_weight(all_source_features, all_target_features, args.rho)
                weights = torch.Tensor(weights[:])

        if i % len(dset_loaders["source"]) == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len(dset_loaders["target"]) == 0:
            iter_target = iter(dset_loaders["target"])

        ## forward
        inputs_source, labels_source,ids_source = iter_source.__next__()
        inputs_target, _,ids_target = iter_target.__next__()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()

        _,_, outputs_source = base_network(inputs_source)
        features_target_low,features_target, _ = base_network(inputs_target)

        ##source (smoothed) cross entropy loss
        if args.sampler == "weighted_sampler" or args.sampler == "subset_sampler":
            src_loss = loss.weighted_smooth_cross_entropy(outputs_source, labels_source)
        else:
            weight = weights[ids_source].cuda()
            src_loss = loss.weighted_smooth_cross_entropy(outputs_source, labels_source, weight)

        ##target loss
        fc = copy.deepcopy(base_network.fc)
        for param in fc.parameters():
            param.requires_grad = False
        softmax_tar_out = torch.nn.Softmax(dim=1)(fc(features_target))
        tar_entropy_loss = torch.mean(loss.entropy(softmax_tar_out))
        tar_alpha_power_loss = torch.mean(loss.lp_loss(softmax_tar_out,p=args.p))
        total_loss = src_loss

        ################################################################################################################
        ## local consistency loss
        with torch.no_grad():
            args.K = 4
            args.KK = 3
            if args.dset == "visda-2017":
                args.K = args.KK = 5
            features_test = features_target_low
            softmax_out = softmax_tar_out
            output_f_norm = torch.nn.functional.normalize(features_test)
            output_f_ = output_f_norm.cpu().detach().clone()

            fea_bank[ids_target] = output_f_.detach().clone().cpu()
            score_bank[ids_target] = softmax_out.detach().clone()

            distance = output_f_ @ fea_bank.T
            _, idx_near = torch.topk(distance,
                                        dim=-1,
                                        largest=True,
                                        k=args.K + 1)
            idx_near = idx_near[:, 1:]  # batch x K

            fea_near = fea_bank[idx_near]  # batch x K x num_dim
            fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
            distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
            _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                            k=args.KK + 1)  # M near neighbors for each of above K ones
            idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
            tar_idx_ = ids_source.unsqueeze(-1).unsqueeze(-1)
            match = (
                    idx_near_near == tar_idx_).sum(-1).float()  # batch x K
            weight = torch.where(
                match > 0., match,
                torch.ones_like(match).fill_(0.1))  # batch x K

            weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                    args.KK)  # batch x K x M
            weight_kk = weight_kk.fill_(0.1)

            score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
            weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                    -1)  # batch x KM

            score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                            args.class_num)  # batch x KM x C

        # nn of nn
        output_re = softmax_out.unsqueeze(1).expand(-1, args.K * args.KK,
                                                    -1)  # batch x C x 1
        const = torch.mean(
            (torch.nn.functional.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                weight_kk.cuda()).sum(
                1))  # kl_div here equals to dot product since we do not use log for score_near_kk
        con_loss = torch.mean(const)
        ################################################################################################################

        if i>args.start_adapt:
            total_loss = total_loss - args.lp_weight*tar_alpha_power_loss
            total_loss += args.con_weight * con_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        print("step:{:d} \t src_loss:{:.4f} \t tar_loss:{:.4f}"
              "".format(i,src_loss.item(),tar_entropy_loss.item()))

    torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))

    log_str = 'Acc: ' + str(np.round(best_acc * 100, 2)) + '\n'
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Adversarial Reweighting for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='3', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50"])
    parser.add_argument('--dset', type=str, default='visda-2017',
                        choices=["office", "office_home", "imagenet_caltech", "domainnet","visda-2017"])
    parser.add_argument('--root', type=str, default='/data/guxiang/dataset',help="root to data")
    parser.add_argument('--p', type=float, default=6)
    parser.add_argument('--lp_weight',type=float, default=0.3)
    parser.add_argument('--rho', type=float, default=5)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.start_adapt = 0
    args.normalize_classifier = True
    args.gamma = 0.001
    args.lr = 1e-3
    args.worker = 4
    args.con_weight = 1.0
    args.output = f"run_{args.seed}"

    if args.dset == 'domainnet':
        names = ['clipart', 'painting', 'real', 'sketch']
        k = 40
        args.class_num = 126
        args.max_iterations = 8000
        args.test_interval = 1000
        args.weight_update_interval = 1000
        args.start_adapt = args.weight_update_interval
        args.sampler = "weighted_sampler"

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 25
        args.class_num = 65
        args.test_interval = 500
        args.max_iterations = 3000
        args.weight_update_interval = 500
        args.sampler = "uniform_sampler"
        args.start_adapt = args.weight_update_interval

    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        k = 10
        args.class_num = 31
        args.max_iterations = 4000
        args.test_interval = 200
        args.weight_update_interval = 500
        args.lr = 5e-4
        args.lp_weight = 1.0
        args.start_adapt = 1500
        args.sampler = "uniform_sampler"

    if args.dset == 'imagenet_caltech':
        names = ['imagenet', 'caltech']
        k = 84
        if args.s == 1:
            args.class_num = 256
            args.max_iterations = 10000
            args.weight_update_interval = 1000
            args.sampler = "weighted_sampler"
        else:
            args.class_num = 1000
            args.max_iterations = 20000
            args.weight_update_interval = 2000
            args.sampler = "subset_sampler"
            args.gamma = 0.0004
        args.test_interval = 1000
        args.start_adapt = args.weight_update_interval

    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        k = 6
        args.class_num = 12
        args.max_iterations = 6000
        args.test_interval = 1000
        args.weight_update_interval = 1000
        if args.s == 0:
            args.lr = 1e-4
            args.sampler = "weighted_sampler"
            args.normalize_classifier = False
        else:
            raise NotImplementedError
        args.lp_weight = 1.0

    args.radius = utils.recommended_radius(args.class_num)
    args.bottleneck_dim = utils.recommended_bottleneck_dim(args.class_num)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    data_folder = './data/'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + '.txt'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '_' + str(k) + '.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join('ckp/', args.dset, args.name, args.output)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.out_file = open(os.path.join(args.output_dir, "log.txt"), "w")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    args.out_file.write(str(args) + '\n')
    args.out_file.flush()

    train(args)

