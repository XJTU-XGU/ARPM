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

def image_classification(loader, model,log_file,thr=0.9,n_share = 25,n_class=25):
    correct = 0
    correct_close = 0
    size = 0
    class_list = [i for i in range(n_share)]
    open_class = n_class
    class_list.append(open_class)
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    per_class_correct_cls = np.zeros((n_share + 1)).astype(np.float32)
    all_pred = []
    all_gt = []
    for batch_idx, data in tqdm.tqdm(enumerate(loader["test"])):
        with torch.no_grad():
            img_t, label_t= data[0], data[1]
            img_t, label_t = img_t.cuda(), label_t.cuda()
            _,_,out_t = model(img_t)
            out_t = torch.nn.Softmax(dim=1)(out_t)
            prob = (out_t.data.max(1)[0]).data.cpu().numpy()
            pred = out_t.data.max(1)[1]
            k = label_t.data.size()[0]
            pred_cls = pred.cpu().numpy()
            pred = pred.cpu().numpy()

            pred_unk = np.where(prob<thr)
            label_t[label_t >= n_share] = n_share
            pred[pred_unk[0]] = open_class
            all_gt += list(label_t.data.cpu().numpy())
            all_pred += list(pred)
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                correct_ind_close = np.where(pred_cls[t_ind[0]] == i)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_correct_cls[i] += float(len(correct_ind_close[0]))
                per_class_num[i] += float(len(t_ind[0]))
                correct += float(len(correct_ind[0]))
                correct_close += float(len(correct_ind_close[0]))
            size += k
    per_class_acc = per_class_correct / per_class_num*100
    os = np.mean(per_class_acc[:-1])
    unk = per_class_acc[-1]
    H = 2*os*unk/(os+unk)
    print("thr:{:.2f}\t os*:{:.2f}\t unk:{:.2f}\t os:{:.2f}\t H:{:.2f}\n".format(thr,os, unk,np.mean(per_class_acc), H))
    log_file.write("thr:{:.2f}\t os*:{:.2f}\t unk:{:.2f}\t os:{:.2f}\t H:{:.2f}\n".format(thr,os, unk,np.mean(per_class_acc),
                                                                                                         H))
    log_file.flush()
    torch.save(prob,"prob.pt")
    return H

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



