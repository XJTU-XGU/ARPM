def recommended_bottleneck_dim(num_class):
    j = 8
    while True:
        if 3*num_class <= 256:
            dim = 256
            break
        elif 3*num_class > 2**j and 3*num_class <= 2**(j+1):
            dim = 2**(j+1)
            break
        j += 1
    return dim

import tqdm
import torch

def image_classification_test(loader, model,tencrop=False,per_class = False,log_file=None):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        if not tencrop:
            for i in tqdm.trange(len(loader['test'])):
                data = iter_test.__next__()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _,_, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
        else:
            iter_test = iter(loader["test_ten"])
            for i in tqdm.trange(len(loader['test_ten'])):
                data = iter_test.__next__()
                inputs = data[0]
                bs, nc, c, h, w = inputs.size()
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels
                _, _,outputs = model(inputs.view(-1,c,h,w))
                outputs = outputs.view(bs, nc, -1).mean(1)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    if per_class:
        class_num = 6
        subclasses_correct = np.zeros(class_num)
        subclasses_tick = np.zeros(class_num)
        correct = 0
        for i in range(predict.size()[0]):
            subclasses_tick[int(all_label[i])] += 1
            if predict[i].float() == all_label[i]:
                correct += 1
                subclasses_correct[predict[i]] += 1
        subclasses_result = np.divide(subclasses_correct, subclasses_tick)
        print("========accuracy per class==========")
        print(subclasses_result, subclasses_result.mean())
        log_file.write("\n\n========accuracy per class==========\n")
        log_file.write(f"perclass_acc:{subclasses_result}, \n"
                       f"perclass_mean:{subclasses_result.mean()}\n")
        accuracy = subclasses_result.mean()

    return accuracy,predict


def get_features(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm.trange(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, feats, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_feature = feats.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_feature = torch.cat((all_feature,feats.float().cpu()),0)
                all_label = torch.cat((all_label, labels.float()), 0)
    return all_feature, all_label, all_output


import numpy as np

def lower_bound_of_radius(num_class,Pw=0.999):
    K = num_class
    bound = (K-1)/K*np.log((K-1)*Pw/(1-Pw))
    return bound

def recommended_radius(num_class):
    if num_class < 25:
        Pw = 0.999
    elif num_class >= 25 and num_class < 75:
        Pw = 0.9999
    elif num_class >= 75 and num_class < 150:
        Pw = 0.99999
    elif num_class >= 150:
        Pw = 0.999999
    return lower_bound_of_radius(num_class,Pw)

if __name__ == "__main__":
    print(recommended_radius(65))



