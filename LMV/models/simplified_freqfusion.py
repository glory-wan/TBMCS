import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from typing import Tuple
import importlib


def load_ext(name, funcs):
    ext = importlib.import_module('mmcv.' + name)
    for fun in funcs:
        assert hasattr(ext, fun), f'{fun} miss in module {name}'
    return ext


ext_module = load_ext('_ext', [
    'carafe_naive_forward', 'carafe_naive_backward', 'carafe_forward',
    'carafe_backward'
])


class CARAFEFunction(Function):
    """
        ICCV 2019: CARAFE：Content-Aware ReAssembly of FEatures
        conde is from https: //github.com/open-mmlab/mmdetection
    """
    @staticmethod
    def symbolic(g, features: Tensor, masks: Tensor, kernel_size: int,
                 group_size: int, scale_factor: int) -> Tensor:
        return g.op(
            'mmcv::MMCVCARAFE',
            features,
            masks,
            kernel_size_i=kernel_size,
            group_size_i=group_size,
            scale_factor_f=scale_factor)

    @staticmethod
    def forward(ctx, features: Tensor, masks: Tensor, kernel_size: int,
                group_size: int, scale_factor: int) -> Tensor:
        assert scale_factor >= 1
        assert masks.size(1) == kernel_size * kernel_size * group_size
        assert masks.size(-1) == features.size(-1) * scale_factor
        assert masks.size(-2) == features.size(-2) * scale_factor
        assert features.size(1) % group_size == 0
        assert (kernel_size - 1) % 2 == 0 and kernel_size >= 1
        ctx.kernel_size = kernel_size
        ctx.group_size = group_size
        ctx.scale_factor = scale_factor
        ctx.feature_size = features.size()
        ctx.mask_size = masks.size()

        n, c, h, w = features.size()
        output = features.new_zeros((n, c, h * scale_factor, w * scale_factor))
        routput = features.new_zeros(output.size(), requires_grad=False)
        rfeatures = features.new_zeros(features.size(), requires_grad=False)
        rmasks = masks.new_zeros(masks.size(), requires_grad=False)
        ext_module.carafe_forward(
            features,
            masks,
            rfeatures,
            routput,
            rmasks,
            output,
            kernel_size=kernel_size,
            group_size=group_size,
            scale_factor=scale_factor)

        if features.requires_grad or masks.requires_grad or \
                torch.__version__ == 'parrots':
            ctx.save_for_backward(features, masks, rfeatures)
        return output

    @staticmethod
    def backward(
            ctx,
            grad_output: Tensor) -> Tuple[Tensor, Tensor, None, None, None]:
        features, masks, rfeatures = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        group_size = ctx.group_size
        scale_factor = ctx.scale_factor

        rgrad_output = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input_hs = torch.zeros_like(grad_output, requires_grad=False)
        rgrad_input = torch.zeros_like(features, requires_grad=False)
        rgrad_masks = torch.zeros_like(masks, requires_grad=False)
        grad_input = torch.zeros_like(features, requires_grad=False)
        grad_masks = torch.zeros_like(masks, requires_grad=False)
        ext_module.carafe_backward(
            grad_output.contiguous(),
            rfeatures,
            masks,
            rgrad_output,
            rgrad_input_hs,
            rgrad_input,
            rgrad_masks,
            grad_input,
            grad_masks,
            kernel_size=kernel_size,
            group_size=group_size,
            scale_factor=scale_factor)
        return grad_input, grad_masks, None, None, None


carafe = CARAFEFunction.apply


class Simplify_FreqFusion(nn.Module):
    """
        TPAMI 2024：Frequency-aware Feature Fusion for Dense Image Prediction
        This is a simplified version of FreqFusion, source address:
        https://github.com/Linwei-Chen/FreqFusion
    """
    def __init__(self,
                 hr_channels,
                 lr_channels,
                 compressed_channels=64,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 scale_factor=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 ):
        super().__init__()
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.scale_factor = scale_factor
        self.lowpass_kernel = lowpass_kernel
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.hr_channel_compressor = nn.Conv2d(hr_channels, self.compressed_channels, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, self.compressed_channels, 1)

        self.content_encoder = nn.Conv2d(  # ALPF generator
            self.compressed_channels,
            self.lowpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)

        self.content_encoder2 = nn.Conv2d(  # AHPF generator
            self.compressed_channels,
            self.highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
            self.encoder_kernel,
            padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
            dilation=self.encoder_dilation,
            groups=1)


    @staticmethod
    def kernel_normalizer(mask, kernel, hamming=1):
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)

        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, hr_feat, lr_feat):
        compressed_hr_feat = self.hr_channel_compressor(hr_feat)
        compressed_lr_feat = self.lr_channel_compressor(lr_feat)

        mask_hr_hr_feat = self.content_encoder2(compressed_hr_feat)
        mask_hr_init = self.kernel_normalizer(mask_hr_hr_feat, self.highpass_kernel)
        # print(compressed_hr_feat.shape)
        # print(mask_hr_init.shape)
        compressed_hr_feat = compressed_hr_feat + compressed_hr_feat - carafe(compressed_hr_feat,
                                                                              mask_hr_init,
                                                                              self.highpass_kernel,
                                                                              self.up_group, 1)

        mask_lr_hr_feat = self.content_encoder(compressed_hr_feat)
        mask_lr_init = self.kernel_normalizer(mask_lr_hr_feat, self.lowpass_kernel)

        mask_lr_lr_feat_lr = self.content_encoder(compressed_lr_feat)
        mask_lr_lr_feat = F.interpolate(
            carafe(mask_lr_lr_feat_lr, mask_lr_init, self.lowpass_kernel, self.up_group, 2),
            size=compressed_hr_feat.shape[-2:], mode='nearest')
        mask_lr = mask_lr_hr_feat + mask_lr_lr_feat

        mask_lr_init = self.kernel_normalizer(mask_lr, self.lowpass_kernel )
        mask_hr_lr_feat = F.interpolate(
            carafe(self.content_encoder2(compressed_lr_feat), mask_lr_init, self.lowpass_kernel,
                   self.up_group, 2), size=compressed_hr_feat.shape[-2:], mode='nearest')
        mask_hr = mask_hr_hr_feat + mask_hr_lr_feat

        mask_lr = self.kernel_normalizer(mask_lr, self.lowpass_kernel )

        lr_feat = carafe(lr_feat, mask_lr, self.lowpass_kernel, self.up_group, 2)

        mask_hr = self.kernel_normalizer(mask_hr, self.highpass_kernel )
        hr_feat_hf = hr_feat - carafe(hr_feat, mask_hr, self.highpass_kernel, self.up_group, 1)
        hr_feat = hr_feat_hf + hr_feat

        return mask_lr, hr_feat, lr_feat


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    """ usage of Simplify_FreqFusion """
    # hr_channels = 64
    # lr_channels = 9
    # ff = Simplify_FreqFusion(hr_channels=hr_channels, lr_channels=lr_channels).to(device)
    # hr_feat = torch.rand(1, hr_channels, 32, 32).to(device)
    # lr_feat = torch.rand(1, lr_channels, 16, 16).to(device)
    # mask_lr, hr, lr = ff(hr_feat=hr_feat, lr_feat=lr_feat)  # lr_feat [1, 64, 32, 32]
    #
    # for i in [mask_lr, hr, lr]:
    #     print(i.shape)

    """ usage of carafe """
    # kernel_size = 3
    # group_size = 1
    # scale_factor = 2
    #
    # mask_channel = kernel_size * kernel_size * group_size
    #
    # feature = torch.rand(1, 16, 32, 32).to(device)
    # mask = torch.rand(1, mask_channel, 64, 64).to(device)
    # mask = carafe(feature, mask, kernel_size, group_size, scale_factor).to(device)
    #
    # print(mask.shape)  # [1, 16, 64, 64]


