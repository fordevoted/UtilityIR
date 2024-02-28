import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi
import numpy as np
import math
import random
import torchvision.transforms as transforms
import cv2
from PIL import Image


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_zero(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.zeros_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs + 1 - self.decay_start_epoch)


#################################
#           Model
#################################
class UtilityIR(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, deg_dim=16):
        super(UtilityIR, self).__init__()
        self.die = DegradationInformationEncoder(in_channels=in_channels, dim=dim, n_downsample=n_downsample, deg_dim=deg_dim)
        self.R = RestoreNet(in_channels=in_channels, dim=dim, n_residual=n_residual, n_downsample=n_downsample, deg_dim=deg_dim)
        self.recon = ReconNet(dim=dim, n_upsample=n_downsample, n_residual=n_residual, deg_dim=deg_dim)
    def forward(self, x, deg_type=None, deg_severity=None, training=True):
        if deg_type is None or deg_severity is None:
            deg_type, qa, deg_severity = self.die(x, psnr_weight=50)
        feat = self.R(x, deg_type, deg_severity)
        x = self.recon(feat) + x
        if training:
            return deg_type, qa, deg_severity, x
        else:
            return x

#################################
#           Encoder
#################################

class RestoreNet(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_residual=3, n_downsample=2, deg_dim=16):
        super(RestoreNet, self).__init__()
        # Generator
        

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(2),
            nn.Conv2d(in_channels, dim, 5),
            AdaIN(deg_dim, dim),
            nn.GELU(),
            nn.ReflectionPad2d(2),
            nn.Conv2d(dim, dim, 5),
            AdaIN(deg_dim, dim),
            nn.GELU(),
        ]
        self.feat_extarct = nn.Sequential(*layers)
        # Downsampling
        self.down_0 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
            AdaIN(deg_dim, dim * 2),
            nn.GELU(),
        )
        dim *= 2
        self.down_1 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
            AdaIN(deg_dim, dim * 2),
            nn.GELU(),
        )
        dim *= 2

        layers = []
        # Residual blocks
        for _ in range(n_residual + 2):
            layers += [
                ResidualBlock(dim, norm="WI-LGAdaIN", deg_dim=deg_dim)]

        self.model = nn.Sequential(*layers)
        self.att = DGCA(embed_dim=dim, num_heads=2, deg_dim=deg_dim)
        # self.att = ContentSelfAttention(embed_dim=dim, num_heads=2)
        # Initiate mlp (predicts AdaIN parameters)
        self.deg_dim = deg_dim

    def assign_affine_params(self, deg_severity, deg_type):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "GlobalAffine" or m.__class__.__name__ == "AdaIN":
                # Extract mean and std predictions
                weight = m.weight_enc(deg_severity)
                bias = m.bias_enc(deg_severity)
                # Update bias and weight
                m.bias = bias.contiguous()
                m.weight = weight.contiguous()
            if m.__class__.__name__ == "LocalAffine":
                # Extract mean and std predictions
                weight = m.weight_enc(deg_type)
                bias = m.bias_enc(deg_type)
                m.weight = weight.contiguous()
                m.bias = bias.contiguous()


    def forward(self, x, deg_type, deg_severity):
        self.assign_affine_params(deg_severity, deg_type)
        # self.assign_local_affine_params(sty_cls)
        feat = self.feat_extarct(x)
        down_0 = self.down_0(feat)
        down_1 = self.down_1(down_0)
        feat = self.model(down_1)
        att = self.att(feat, deg_type=deg_type)

        return att


class DegradationInformationEncoder(nn.Module):
    def __init__(self, in_channels=3, dim=64, n_downsample=2, n_residual=3, deg_dim=8, num_classes=3):
        super(DegradationInformationEncoder, self).__init__()

        # Initial conv block
        layers = [nn.ReflectionPad2d(2),
                  nn.Conv2d(in_channels, dim, 5),
                  nn.GELU(),
                  nn.ReflectionPad2d(2),
                  nn.Conv2d(dim, dim, 5),
                  nn.GELU(), ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.GELU()]
            dim *= 2

        # Downsampling with constant depth
        # for _ in range(n_downsample -2):
        #     layers += {nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.GELU()}
        self.feat = nn.Sequential(*layers)
        self.feat0 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1), nn.GELU())
        self.feat1 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1), nn.GELU())
        self.feat2 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1), nn.GELU())
        self.feat3 = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1), nn.GELU())
        self.feat_enc = nn.Sequential(nn.Conv2d(dim * 4, deg_dim, 1, stride=1, padding=0), nn.GELU())
        # Average pool and output layer
        # layers += [nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, deg_dim, 1, 1, 0)]

        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, deg_dim, 1, 1, 0), nn.Flatten(start_dim=1))
        # self.classifier = nn.Sequential(nn.Conv2d(dim, deg_dim, kernel_size=3, padding=1), nn.PReLU(), nn.Conv2d(deg_dim, num_classes, 3, padding=1), nn.AdaptiveAvgPool2d(1))
        self.IQA_regressor = nn.Sequential(nn.Linear(deg_dim, deg_dim // 2), nn.GELU(), nn.Linear(deg_dim // 2, 1))

    def forward(self, x, psnr_weight=10):
        feat = self.feat(x)
        feat0 = self.feat0(feat)
        feat1 = self.feat1(feat0)
        feat2 = self.feat2(feat1)
        feat3 = self.feat3(feat2)
        deg_type = self.feat_enc(torch.cat([feat0, feat1, feat2, feat3], dim=1))
        deg_severity = self.avgpool(feat3)
        iqa = psnr_weight * self.IQA_regressor(deg_severity)
        # cls = self.classifier(feat).squeeze()
        # return deg_type, cls
        return deg_type, iqa, deg_severity


class ReconNet(nn.Module):
    def __init__(self, out_channels=3, dim=64, n_residual=3, n_upsample=2, deg_dim=8):
        super(ReconNet, self).__init__()

        layers = []
        dim = dim * 2 ** n_upsample
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="layer")]
        self.res = nn.Sequential(*layers)

        # Upsampling

        self.up_0 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=5, padding=2, stride=2, output_padding=1),
            # LayerNorm(dim // 2),
            nn.GELU(),
            )
        dim = dim // 2
        self.up_1 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=5, padding=2, stride=2, output_padding=1),
            # LayerNorm(dim // 2),
            nn.GELU(),
            )
        dim = dim // 2
        layers = []
        # Output layer
        layers += [nn.ReflectionPad2d(2), nn.Conv2d(dim, dim // 2, 5), nn.ReflectionPad2d(2),
                   nn.Conv2d(dim // 2, out_channels, 5), nn.Tanh()]

        self.model = nn.Sequential(*layers)
    def forward(self, x):
        res = self.res(x)  # + down_1
        up_0 = self.up_0(res)  # + down_0
        up_1 = self.up_1(up_0)  # + feat
        out = self.model(up_1)
        return out
        

######################################
#   MLP (predicts AdaIn parameters)
######################################

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), nn.GELU()]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.GELU()]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##############################
#       Custom Blocks
##############################


class ResidualBlock(nn.Module):
    def __init__(self, features, norm="WI-LGAdaIN", deg_dim=128):
        super(ResidualBlock, self).__init__()

        norm_layer = WI_LGAdaIN if norm == "WI-LGAdaIN" else LayerNorm

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features=features, deg_dim=deg_dim) if norm == 'WI-LGAdaIN' else norm_layer(features),
            nn.GELU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features=features, deg_dim=deg_dim) if norm == 'WI-LGAdaIN' else norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


##############################
#        Custom Layers
##############################
class WI_LGAdaIN(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, features, deg_dim=128, num_features=64, eps=1e-5, momentum=0.1):
        super(WI_LGAdaIN, self).__init__()
        self.IN = nn.InstanceNorm2d(features)
        self.local_affine = LocalAffine(deg_dim)
        self.global_affine = GlobalAffine(deg_dim, features)

    def forward(self, x):
        x = self.IN(x)
        x = self.local_affine(x)
        x = self.global_affine(x)
        return x


class AdaIN(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, deg_dim=128, num_features=64, eps=1e-5, momentum=0.1):
        super(AdaIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.weight_enc = MLP(input_dim=deg_dim, output_dim=num_features, dim=32)
        self.bias_enc = MLP(input_dim=deg_dim, output_dim=num_features, dim=32)

    def forward(self, x):
        assert (
                self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class GlobalAffine(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, deg_dim=128, num_features=64, eps=1e-5, momentum=0.1):
        super(GlobalAffine, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.weight_enc = MLP(input_dim=deg_dim, output_dim=num_features, dim=32)
        self.bias_enc = MLP(input_dim=deg_dim, output_dim=num_features, dim=32)

    def forward(self, x):
        assert (
                self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        self.weight = self.weight[:, :, None, None]
        self.bias = self.bias[:, :, None, None]
        out = self.weight * x + self.bias

        return out

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LocalAffine(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(LocalAffine, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        self.weight_enc = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
                                        nn.GELU(),
                                        nn.Conv2d(num_features, 1, kernel_size=3, padding=1))
        self.bias_enc = nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                      nn.GELU(),
                                      nn.Conv2d(num_features, 1, kernel_size=3, stride=1, padding=1))
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
                self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling LocalAffine!"
        b, c, h, w = x.size()

        out = self.weight * x + self.bias

        return out

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class DGCA(nn.Module):
    def __init__(self, embed_dim, num_heads, deg_dim, dropout=0.0, bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None,
                 vdim=None, batch_first=False, device=None, dtype=None):
        super(DGCA, self).__init__()
        self.q_enc = nn.Sequential(nn.Conv2d(deg_dim, embed_dim, kernel_size=3, padding=1),
                                   nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1))
        self.k_enc = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                                   nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1))
        self.v_enc = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                                   nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, padding=1))

        # self.positional_encoding = PositionalEncodingPermute2D(embed_dim)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, kdim=embed_dim, vdim=embed_dim, num_heads=num_heads,
                                         dropout=dropout, bias=bias, add_bias_kv=add_bias_kv,
                                         add_zero_attn=add_zero_attn)
        self.embed_dim = embed_dim

    def forward(self, feat, deg_type):
        # pe =  self.positional_encoding(cont)
        # x = cont+ pe

        q = self.q_enc(deg_type)
        k = self.k_enc(feat)
        v = self.v_enc(feat)

        q = torch.flatten(q.permute(2, 3, 0, 1), start_dim=0, end_dim=1)  # shape: (L, N, E) = (H*W, N, E)
        k = torch.flatten(k.permute(2, 3, 0, 1), start_dim=0, end_dim=1)  # shape: (L, N, E) = (H*W, N, E)
        v = torch.flatten(v.permute(2, 3, 0, 1), start_dim=0, end_dim=1)  # shape: (L, N, E) = (H*W, N, E)
        try:
            out, weight = self.att(q, k, v, need_weights=False)  # shape: (L, N, E); (N, L, S)
        except:
            out, weight = self.att.cpu()(q.cpu(), k.cpu(), v.cpu(), need_weights=False)
            out = out.cuda()
            self.att.cuda()
        out = out.permute(1, 2, 0).view(feat.size(0), feat.size(1), feat.size(2), feat.size(3))
        # q = q.view(q.size(0), q.size(1)*q.size(2), q.size(3))

        return out + feat
