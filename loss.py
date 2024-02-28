from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import kornia


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average,
                    val_range=1)


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, weighting=1, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize
        self.weighting = weighting

    def forward(self, input, target, feature_layers=[0, 1, 2], style_layers=[2, 3]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return self.weighting * loss


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class Contrastive(nn.Module):
    def __init__(self, temperture=0.4, epsilon=0.25, vgg=True):
        super(Contrastive, self).__init__()
        self.temperture = temperture
        self.epsilon = epsilon
        if vgg:
            blocks = []
            blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
            blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
            for bl in blocks:
                for p in bl:
                    p.requires_grad = False
            self.blocks = torch.nn.ModuleList(blocks)
            self.transform = torch.nn.functional.interpolate
            self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.weights = [4.0 / 47, 8.0 / 47, 16.0 / 47, 32.0 / 47, 32 * 4.0 / 47]

    def forward(self, x, x_pos, x_neg, type='img', feature_layers=[0, 1, 2], style_layers=[3]):
        # img
        if type == 'img':
            if x.shape[1] != 3:
                x = x.repeat(1, 3, 1, 1)
                x_pos = x_pos.repeat(1, 3, 1, 1)
                x_neg = x_neg.repeat(1, 3, 1, 1)
            x = (x - self.mean) / self.std
            x_pos = (x_pos - self.mean) / self.std
            x_neg = (x_neg - self.mean) / self.std
            loss = 0.0

            for i, block in enumerate(self.blocks):
                x = block(x)
                x_pos = block(x_pos)
                x_neg = block(x_neg)
                if i in feature_layers:
                    loss += self.weights[i] * (
                            torch.nn.functional.l1_loss(x, x_pos) / (torch.nn.functional.l1_loss(x, x_neg)))
                if i == len(self.blocks) - 1:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_x_pos = x_pos.reshape(x_pos.shape[0], x_pos.shape[1], -1)
                    act_x_neg = x_neg.reshape(x_neg.shape[0], x_neg.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_x_pos = act_x_pos @ act_x_pos.permute(0, 2, 1)
                    gram_x_neg = act_x_neg @ act_x_neg.permute(0, 2, 1)
                    pos = torch.nn.functional.l1_loss(gram_x, gram_x_pos)
                    neg = torch.nn.functional.l1_loss(gram_x, gram_x_neg)
                    loss += self.weights[i] * pos / (neg)
            return loss

        if type == 'img_per':
            if x.shape[1] != 3:
                x = x.repeat(1, 3, 1, 1)
                x_pos = x_pos.repeat(1, 3, 1, 1)
                x_neg = x_neg.repeat(1, 3, 1, 1)
            x = (x - self.mean) / self.std
            x_pos = (x_pos - self.mean) / self.std
            x_neg = (x_neg - self.mean) / self.std
            loss = 0.0

            for i, block in enumerate(self.blocks):
                x = block(x)
                x_pos = block(x_pos)
                if i in feature_layers:
                    loss += self.weights[i] * (
                        torch.nn.functional.l1_loss(x, x_pos))
                if i == len(self.blocks) - 1:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_x_pos = x_pos.reshape(x_pos.shape[0], x_pos.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_x_pos = act_x_pos @ act_x_pos.permute(0, 2, 1)
                    pos = torch.nn.functional.l1_loss(gram_x, gram_x_pos)
                    loss += pos
                    return loss
        # sty
        if type == 'deg':
            # a = torch.mean(x * x_pos)/(torch.norm(x)*torch.norm(x_pos))
            # b = torch.mean(x * x_neg) / (torch.norm(x)*torch.norm(x_neg))
            a = torch.nn.functional.l1_loss(x, x_pos)
            b = torch.nn.functional.l1_loss(x, x_neg)
            # numer = torch.exp(b / self.temperture)
            # denom = torch.exp(a / self.temperture)
            # return -torch.log(numer/denom)
            return a / b
        if type == 'deg_r':
            b = 0
            x = x.view(x.size(0), -1)
            x_pos = x_pos.view(x_pos.size(0), -1)
            for neg in x_neg:
                neg = neg.view(neg.size(0), -1)
                b += torch.exp(torch.nn.functional.cosine_similarity(x, neg) / self.temperture)
            a = torch.exp(torch.nn.functional.cosine_similarity(x, x_pos) / self.temperture)
            return (-torch.log(a / (a + b))).mean()
        if type == 'deg_m_r':
            b = 0
            x = x.view(x.size(0), -1)
            x_pos = x_pos.view(x_pos.size(0), -1)
            for neg in x_neg:
                neg = neg.view(neg.size(0), -1)
                cos = torch.nn.functional.cosine_similarity(x, neg)
                cos = torch.where(cos < (-1 + self.epsilon), torch.clamp(cos - self.epsilon, -1, 1), cos)
                b += torch.exp(cos / self.temperture)

            cos = torch.nn.functional.cosine_similarity(x, x_pos)
            cos = torch.where(cos > (1 - self.epsilon), torch.clamp(cos + self.epsilon, -1, 1), cos)
            a = torch.exp(cos / self.temperture)
            return (-torch.log(a / (a + b))).mean()


class LaplacianPyramidLoss(torch.nn.Module):
    def __init__(self, max_levels=3, channels=3, kernel_size=5, sigma=1, device=torch.device('cpu'), dtype=torch.float):
        super(LaplacianPyramidLoss, self).__init__()
        self.max_levels = max_levels
        self.kernel = self.gaussian_kernel(size=kernel_size, channels=channels, sigma=sigma, dtype=dtype)

    def gaussian_kernel(self, size=5, device=torch.device('cuda'), channels=3, sigma=1, dtype=torch.float):
        # Create Gaussian Kernel. In Numpy
        interval = (2 * sigma + 1) / (size)
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel /= np.sum(kernel)
        # Change kernel to PyTorch. reshapes to (channels, 1, size, size)
        kernel_tensor = torch.as_tensor(kernel, dtype=dtype)
        kernel_tensor = kernel_tensor.repeat(channels, 1, 1, 1)
        kernel_tensor.to(device)
        return kernel_tensor

    def gaussian_conv2d(self, x, g_kernel, dtype=torch.float):
        # Assumes input of x is of shape: (minibatch, depth, height, width)
        # Infer depth automatically based on the shape
        channels = g_kernel.shape[0]
        padding = g_kernel.shape[-1] // 2  # Kernel size needs to be odd number
        if len(x.shape) != 4:
            raise IndexError(
                'Expected input tensor to be of shape: (batch, depth, height, width) but got: ' + str(x.shape))
        y = F.conv2d(x, weight=g_kernel, stride=1, padding=padding, groups=channels)
        return y

    def downsample(self, x):
        # Downsamples along  image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
        return x[:, :, ::2, ::2]

    def create_laplacian_pyramid(self, x, kernel, levels):
        upsample = torch.nn.Upsample(scale_factor=2)  # Default mode is nearest: [[1 2],[3 4]] -> [[1 1 2 2],[3 3 4 4]]
        pyramids = []
        current_x = x
        for level in range(0, levels):
            gauss_filtered_x = self.gaussian_conv2d(current_x, kernel)
            down = self.downsample(gauss_filtered_x)
            # Original Algorithm does indeed: L_i  = G_i  - expand(G_i+1), with L_i as current laplacian layer, and G_i as current gaussian filtered image, and G_i+1 the next.
            # Some implementations skip expand(G_i+1) and use gaussian_conv(G_i). We decided to use expand, as is the original algorithm
            laplacian = current_x - upsample(down)
            pyramids.append(laplacian)
            current_x = down
        pyramids.append(current_x)
        return pyramids

    def forward(self, x, target):
        self.kernel = self.kernel.to(x.device)
        input_pyramid = self.create_laplacian_pyramid(x, self.kernel, self.max_levels)
        target_pyramid = self.create_laplacian_pyramid(target, self.kernel, self.max_levels)
        return sum(torch.nn.functional.l1_loss(x, y) for x, y in zip(input_pyramid, target_pyramid))


class TruncatedSmoothL1(nn.Module):
    def __init__(self, beta, truncated):
        super(TruncatedSmoothL1, self).__init__()
        self.beta = beta
        self.truncated = truncated

    def forward(self, x, y):
        diff = torch.abs(x - y)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)
        loss = torch.where(loss >= self.truncated, self.truncated, loss.double())

        return 0.5 * torch.mean(loss)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        dif = torch.pow(x - y, 2)
        return torch.mean(torch.sqrt(dif + self.eps))


class QualityAssessment(nn.Module):
    def __init__(self, metric='psnr', max=50.0, scale=5.0):
        super(QualityAssessment, self).__init__()
        self.metric = metric
        self.max = max
        self.scale = scale

    def forward(self, x, gt):
        if self.metric == 'psnr':
            mse = torch.mean(torch.pow(x - gt, 2), dim=[1, 2, 3]).unsqueeze(-1)
            psnr = 10 * torch.log10(1 / mse)
            psnr = torch.where(psnr < self.max, psnr, torch.tensor(self.max).cuda())
            return self.scale * (1 - psnr / self.max)
        if self.metric == 'psnr2':
            mse = torch.mean(torch.pow(x - gt, 2), dim=[1, 2, 3]).unsqueeze(-1)
            psnr = 10 * torch.log10(1 / mse)
            psnr = torch.clamp(psnr, 0, self.max)
            return psnr / self.max


class MarginalQualityRankingLoss(nn.Module):
    def __init__(self, eps=0.3):
        super(MarginalQualityRankingLoss, self).__init__()
        self.epsilon = eps  # == psnr=3

    def forward(self, en_a, en_b, gt_qa_a, gt_qa_b):
        diff_en = en_a - en_b
        diff_gt = gt_qa_a - gt_qa_b
        diff = torch.abs(diff_gt - diff_en)
        # out = torch.where(diff > torch.abs(diff_gt), diff * 2, torch.max(torch.tensor(0.0).cuda(), diff - self.epsilon))
        out = torch.where(diff >= (torch.abs(diff_gt) + torch.abs(diff_en)), diff,
                          torch.max(torch.tensor(0.0).cuda(), diff - self.epsilon))
        return out.mean()


class FocalPixelLoss(nn.Module):
    def __init__(self):
        super(FocalPixelLoss, self).__init__()

    def forward(self, x, gt):
        weight = torch.abs(x - gt) ** 1
        weight = torch.clamp(weight, 0, 1)
        loss = F.l1_loss(x, gt, reduction='none')
        return torch.mean(weight * loss)
