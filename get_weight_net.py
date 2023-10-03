import torch
import numpy as np
import cvxpy as cvx
from network import WassersteinDiscriminatorSN
from torch.utils.data import DataLoader,TensorDataset
import tqdm
from torch_ema import ExponentialMovingAverage

class WeightLearner(object):
    def __init__(self,input_dim=256,lr=0.001):
        self.adnet = WassersteinDiscriminatorSN(input_dim, 1024).cuda()
        self.optimizer = torch.optim.Adam(self.adnet.parameters(), lr=lr)
        self.ema = ExponentialMovingAverage(self.adnet.parameters(),decay=0.9)
        self.weight = None

    def update_weight(self,weight,decay=0.):
        if self.weight is None:
            self.weight = 1.0*weight
        else:
            self.weight = decay*self.weight + (1-decay)*weight

    def get_weight(self,feature_source,feature_target,rho=5.0):
        loader_s = DataLoader(TensorDataset(feature_source), batch_size=36, shuffle=True, drop_last=True)
        loader_t = DataLoader(TensorDataset(feature_target), batch_size=36, shuffle=True, drop_last=True)

        num_steps = 15000
        for i in tqdm.trange(num_steps):
            if i % len(loader_s) == 0:
                iter_s = iter(loader_s)
            if i % len(loader_t) == 0:
                iter_t = iter(loader_t)
            feat_s = iter_s.__next__()[0].cuda()
            feat_t = iter_t.__next__()[0].cuda()
            out_s = self.adnet(feat_s)
            out_t = self.adnet(feat_t)
            wdist = out_s.mean() - out_t.mean()
            loss = -wdist
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()

        outs_s = self.adnet(feature_source.cuda()).cpu()
        outs_t = self.adnet(feature_target.cuda()).cpu()

        ds = np.reshape(outs_s.data.numpy(), (-1,))
        dt = np.reshape(outs_t.data.numpy(), (-1,))

        max_value = max(np.max(np.abs(ds)), np.max(np.abs(dt)))
        ds_ = ds/max_value
        dt_ = dt/max_value

        n = len(ds)
        w = cvx.Variable(n)
        ones = np.ones(n)
        obj = cvx.Minimize(w @ ds)

        con = [w >= 0,
               cvx.sum_squares(w - ones) <= rho * n,
               cvx.sum(w) == n,
               ]
        prob = cvx.Problem(obj, con)
        prob.solve(cvx.ECOS, max_iters=500)
        op_wdist = w.value @ ds_ / n - np.mean(dt_)
        print("status:", prob.status)
        print("original dist:", np.mean(ds_) - np.mean(dt_))
        print("optimal dist:", op_wdist)

        weight = w.value
        self.update_weight(weight)
        return self.weight

    def get_weight_large(self,feature_source,feature_target,rho=5.0):
        loader_s = DataLoader(TensorDataset(feature_source), batch_size=36, shuffle=True, drop_last=True)
        loader_t = DataLoader(TensorDataset(feature_target), batch_size=36, shuffle=True, drop_last=True)

        num_steps = 15000
        for i in tqdm.trange(num_steps):
            if i % len(loader_s) == 0:
                iter_s = iter(loader_s)
            if i % len(loader_t) == 0:
                iter_t = iter(loader_t)
            feat_s = iter_s.__next__()[0].cuda()
            feat_t = iter_t.__next__()[0].cuda()
            out_s = self.adnet(feat_s)
            out_t = self.adnet(feat_t)
            wdist = out_s.mean() - out_t.mean()
            loss = -wdist
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.ema.update()

        outs_s = self.adnet(feature_source.cuda()).cpu()
        outs_t = self.adnet(feature_target.cuda()).cpu()

        ds = np.reshape(outs_s.data.numpy(), (-1,))
        dt = np.reshape(outs_t.data.numpy(), (-1,))

        max_value = max(np.max(np.abs(ds)), np.max(np.abs(dt)))
        ds_ = ds / max_value
        dt_ = dt / max_value

        ####################################################
        # split source datasets
        num_splits = len(ds)//20000
        Splits_index = []
        for i in range(num_splits):
            Splits_index.append([num_splits*j + i for j in range(len(ds)//num_splits)])
        global_weights = np.zeros_like(ds)
        for i in range(num_splits):
            print (f"the {i}/{num_splits}-th split:")
            ds_i = ds[Splits_index[i]]
            ds_i_ = ds_[Splits_index[i]]
            n = len(ds_i)
            w = cvx.Variable(n)
            ones = np.ones(n)
            obj = cvx.Minimize(w @ ds_i)

            con = [w >= 0,
                   cvx.sum_squares(w - ones) <= rho * n,
                   cvx.sum(w) == n,
                   ]
            prob = cvx.Problem(obj, con)
            prob.solve(cvx.ECOS, max_iters=500)
            op_wdist = w.value @ ds_i_ / n - np.mean(dt_)
            print("status:", prob.status)
            print("original dist:", np.mean(ds_i_) - np.mean(dt_))
            print("optimal dist:", op_wdist)

            weight_i = w.value
            global_weights[Splits_index[i]] = weight_i
        self.update_weight(global_weights)
        return self.weight