import argparse
import copy
import os
import random
import numpy as np
import torch
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
        if args.t == 0 or args.t == 1:
            args.radius *= 1.
        params = {"resnet_name": args.net, "bottleneck_dim": args.bottleneck_dim,
                  'class_num': args.class_num,"radius":args.radius,"normalize_classifier":args.normalize_classifier}
        base_network = network.ResNetFc(**params)

    base_network = base_network.cuda()

    base_network.train(False)
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
    count_patience = 0
    print("Training begins.")
    for i in range(args.max_iterations + 1):
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        ## test
        if (i % args.test_interval == 0 and i > 0) or (i == args.max_iterations):
            base_network.train(False)
            temp_acc = utils.image_classification(dset_loaders, base_network, args.out_file, 0.65)
            if best_acc < temp_acc:
                best_acc = temp_acc
                best_model = base_network.state_dict()
                torch.save(best_model, os.path.join(args.output_dir, "best_model.pt"))
                count_patience = 0
            else:
                count_patience += 1

            log_str = "\n {} iter: {:05d}, precision: {:.5f}, best_acc: {:.5f} \n".format(args.name,i, temp_acc, best_acc)
            args.out_file.write(log_str + "\n")
            args.out_file.flush()
            print(log_str)

        if count_patience >= args.patient:
            break

        if args.sampler == "uniform_sampler":
            if i % args.weight_update_interval == 0:
                base_network.train(False)
                all_source_features, _, _ = get_features(dset_loaders["source_val"], base_network)
                all_target_features, _, all_target_outputs = get_features(dset_loaders["test"], base_network)
                weights = weight_learner.get_weight(all_target_features, all_source_features,  args.rho)
                weights = torch.Tensor(weights[:])

                weights_unk = torch.ones(len(dsets["test"]))
                weights[torch.argsort(weights)[:int(0.75 * len(weights))]] = 0.0
                weights_unk[torch.argsort(weights)[int(0.25 * len(weights)):]] = 0.0

                loader_knw = DataLoader(dsets["target"], batch_size=train_bs,
                                        sampler=WeightedRandomSampler(weights, num_samples=len(weights),
                                                                      replacement=True),
                                        num_workers=args.worker, drop_last=True)
                loader_unk = DataLoader(dsets["target"], batch_size=train_bs,
                                        sampler=WeightedRandomSampler(weights_unk, num_samples=len(weights_unk),
                                                                      replacement=True),
                                        num_workers=args.worker, drop_last=True)

        try:
            inputs_source, labels_source, ids_source = iter_source.__next__()
        except:
            iter_source = iter(dset_loaders["source"])
            inputs_source, labels_source, ids_source = iter_source.__next__()

        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()
        _,_, outputs_source = base_network(inputs_source)
        src_loss = loss.weighted_smooth_cross_entropy(outputs_source, labels_source)
        total_loss = 1.0*src_loss

        if i>=args.start_osda:
            fc = copy.deepcopy(base_network.fc)
            for param in fc.parameters():
                param.requires_grad = False
            if (i - args.start_osda) % len(dset_loaders["target"]) == 0:
                iter_target_knw = iter(loader_knw)
                iter_target_unk = iter(loader_unk)
            inputs_target_knw, _, ids_target = iter_target_knw.__next__()
            inputs_target_unk, _, _ = iter_target_unk.__next__()
            inputs_target_knw, inputs_target_unk = inputs_target_knw.cuda(), inputs_target_unk.cuda()
            features_low, outputs_target, _ = base_network(torch.cat((inputs_target_knw,inputs_target_unk),0))
            outputs_target_knw = outputs_target[:len(inputs_target_knw)]
            outputs_target_unk = outputs_target[len(inputs_target_knw):]

            prob_knw = torch.nn.Softmax(dim=1)(fc(outputs_target_knw))
            prob_unk = torch.nn.Softmax(dim=1)(fc(outputs_target_unk))
            loss_knw = torch.mean(torch.sum(prob_knw ** 6, dim=1))
            loss_unk = torch.mean(torch.sum(prob_unk ** 6, dim=1))

            total_loss = total_loss - args.lp_weight * loss_knw + args.lp_weight * loss_unk
            print("step:{:d} \t src_loss:{:.4f} \t loss_knw:{:.4f} \t loss_unk:{:.4f}"
                  "".format(i, src_loss.item(), loss_knw.item(), loss_unk.item()))
        else:
            try:
                inputs_target, _, ids_target = iter_target.__next__()
            except:
                loader = DataLoader(dsets["target"], batch_size=64, shuffle=True, num_workers=args.worker,
                                    drop_last=True)
                iter_target = iter(loader)
                inputs_target, _, ids_target = iter_target.__next__()
            inputs_target = inputs_target.cuda()
            _, feat_tar, outputs_target = base_network(inputs_target)
            fc = copy.deepcopy(base_network.fc)
            for param in fc.parameters():
                param.requires_grad = False
            prob_tar = torch.nn.Softmax(dim=1)(fc(feat_tar))
            loss_tar = -torch.mean(torch.sum(prob_tar ** 6, dim=1))
            total_loss += 0.1*loss_tar
            print("step:{:d} \t src_loss:{:.4f} \t tar_loss:{:.4f} "
                  "".format(i, src_loss.item(),loss_tar.item()))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    log_str = 'Acc: ' + str(np.round(best_acc, 2)) + '\n'
    args.out_file.write(log_str)
    args.out_file.flush()
    print(log_str)

    return best_acc

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Adversarial Reweighting for Partial Domain Adaptation')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='2', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=3, help="target")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument('--batch_size', type=int, default=24, help="batch_size")
    parser.add_argument('--net', type=str, default='ResNet50', choices=["ResNet50"])
    parser.add_argument('--dset', type=str, default='office_home',
                        choices=["office_home"])
    parser.add_argument('--root', type=str, default='/data/guxiang/dataset',help="root to data")
    parser.add_argument('--p', type=float, default=6)
    parser.add_argument('--lp_weight',type=float, default=0.05)
    parser.add_argument('--rho', type=float, default=5)

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.start_osda = 0
    args.normalize_classifier = True
    args.gamma = 0.001
    args.lr = 1e-3
    args.worker = 4
    args.output = f"run_{args.seed}"
    args.patient = 5

    if args.dset == 'office_home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        k = 65
        args.class_num = 25
        args.test_interval = 500
        args.max_iterations = 10000
        args.weight_update_interval = 500
        args.sampler = "uniform_sampler"
        if args.t ==0 or args.t == 1 :
            args.start_osda = 1500
        else:
            args.start_osda = 0

    args.radius = utils.recommended_radius(k)
    args.bottleneck_dim = utils.recommended_bottleneck_dim(k)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    data_folder = './data/'
    args.t_dset_path = data_folder + args.dset + '/' + names[args.t] + '.txt'
    args.s_dset_path = data_folder + args.dset + '/' + names[args.s] + f'_{args.class_num}.txt'

    args.name = names[args.s][0].upper() + names[args.t][0].upper()
    args.output_dir = os.path.join('ckp1/', args.dset, args.name, args.output)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    args.out_file = open(os.path.join(args.output_dir, f"log_{args.batch_size}.txt"), "w")

    args.out_file.write(str(args) + '\n')
    args.out_file.flush()

    train(args)

