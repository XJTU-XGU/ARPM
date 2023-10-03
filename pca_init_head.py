import torch
import numpy as np
import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def get_features(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm.trange(len(loader)):
            data = iter_test.__next__()
            inputs = data[0]
            inputs = inputs.cuda()
            _,feats,_ = model(inputs)
            if start_test:
                all_feature = feats.float().cpu()
                all_label = data[1]
                start_test = False
            else:
                all_feature = torch.cat((all_feature,feats.float().cpu()),0)
                all_label = torch.cat((all_label, data[1]), 0)
    return all_feature,all_label

def init_head(model,loader_source,loader_target,pretrain_head=False):
    features,labels = get_features(loader_source,model)
    features_target, _= get_features(loader_target, model)
    features = torch.cat((features,features_target),dim=0)
    feat = features.numpy()
    pca = PCA(n_components=model.fc.out_features)
    pca.fit(feat[len(labels):])
    scores = pca.transform(feat)[:len(labels)]
    pred = np.argmax(scores,axis=1)
    M = confusion_matrix(labels.numpy(),pred)

    weight = pca.components_
    M = M/(np.sum(M,axis=1,keepdims=True)+1e-5)
    weight = M@weight
    model.fc.weight.data = torch.Tensor(weight).cuda()

    if pretrain_head:
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
        for i in range(500):
            loss = torch.nn.functional.cross_entropy(model.fc(features[:len(labels)].cuda()), labels.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.fc.weight.data = torch.nn.functional.normalize(model.fc.weight.data, dim=1)
            print(f"{i} {loss.item()}")
