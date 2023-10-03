from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import Parameter

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        nn.init.zeros_(m.bias)

resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, bottleneck_dim=256, class_num=1000, radius=8.5, normalize_classifier=True):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](ResNet50_Weights.IMAGENET1K_V1)

    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
    if normalize_classifier:
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.fc = SLR_layer(bottleneck_dim, class_num, bias=True)
        self.__in_features = bottleneck_dim
    else:
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        self.bottleneck.apply(init_weights)
        self.fc.apply(init_weights)
        self.__in_features = bottleneck_dim
    self.radius = radius

  def forward(self, x):
    x = self.feature_layers(x)
    x_ = x.view(x.size(0), -1)
    x = self.bottleneck(x_)
    x = self.radius*F.normalize(x,dim=1)
    y = self.fc(x)
    return x_, x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                    {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                    {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]

    return parameter_list

class SLR_layer(nn.Module):
    def __init__(self, in_features, out_features,bias=True):
        super(SLR_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias=torch.nn.Parameter(torch.zeros(out_features))
        self.bias_bool = bias
        nn.init.kaiming_normal_(self.weight)

    def forward(self, input):
        r=input.norm(dim=1).detach()[0]
        if self.bias_bool:
            cosine = F.linear(input, F.normalize(self.weight),r*torch.tanh(self.bias))
        else:
            cosine = F.linear(input, F.normalize(self.weight))
        output=cosine
        return output

class WassersteinDiscriminator(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(WassersteinDiscriminator, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.LeakyReLU()
    self.relu2 = nn.LeakyReLU()
    self.apply(self.init_weight_)

  def init_weight_(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
            nn.init.kaiming_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.zeros_(m.bias)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    y = self.ad_layer3(x)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]

class WassersteinDiscriminatorSN(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(WassersteinDiscriminatorSN, self).__init__()
    self.ad_layer1 = SpectralNorm(nn.Linear(in_feature, hidden_size))
    self.ad_layer2 = SpectralNorm(nn.Linear(hidden_size, hidden_size))
    self.ad_layer3 = SpectralNorm(nn.Linear(hidden_size, 1))
    self.relu1 = nn.LeakyReLU()
    self.relu2 = nn.LeakyReLU()
    self.apply(self.init_weight_)

  def init_weight_(self, m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('SpectralNorm') != -1:
        nn.init.kaiming_uniform_(m.module.weight_bar)
        nn.init.zeros_(m.module.bias)

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    y = self.ad_layer3(x)
    return y

  def output_num(self):
    return 1

  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":1, 'decay_mult':2}]


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False


    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

