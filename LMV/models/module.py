import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from LMV.models.conv import autopad, Conv, C3k2, DWConv


class Backbone(nn.Module):
    def __init__(self, c1=3, c2=16, c3=32, c4=64, c5=128):
        super().__init__()
        # input [bs, 3, 2760, 3632]
        # input [bs, 3, 640, 640]
        self.Conv1 = Conv(c1=c1, c2=c2, s=2)  # [bs, 16, 320, 320]
        self.Conv2 = Conv(c1=c2, c2=c3, s=2)  # [bs, 32, 160, 160]
        self.Conv3 = C3k2(c1=c3, c2=c3)
        self.Conv4 = Conv(c1=c3, c2=c4, s=2)
        self.Conv5 = C3k2(c1=c4, c2=c4, c3k=True)
        self.Conv6 = Conv(c1=c4, c2=c5, s=2)
        self.Conv7 = C3k2(c1=c5, c2=c5, c3k=True)

        # upsample
        self.Conv8 = Conv(c1=c5, c2=c5)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Conv9 = Conv(c1=c5, c2=c4)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.Conv10 = Conv(c1=c4, c2=c3)

    def forward(self, x):
        c1 = self.Conv1(x)
        c2 = self.Conv2(c1)
        ck1 = self.Conv3(c2)
        x = self.Conv4(ck1)
        ck2 = self.Conv5(x)
        x = self.Conv6(ck2)
        x = self.Conv7(x)  # ck3

        x1 = self.Conv8(x)
        x2 = self.Conv9(x1)
        x2 = self.upsample1(x2)
        x3 = self.Conv10(x2)
        x3 = self.upsample2(x3)

        deep_feature = torch.cat((x, x1), dim=1)
        medium_feature = torch.cat((ck2, x2), dim=1)
        deep_shallow = torch.cat((ck1, x3), dim=1)

        return deep_feature, medium_feature, deep_shallow, c1, c2, ck1


class Backbone2(nn.Module):
    """
        use DeConv to place UnSample
    """

    def __init__(self, c1=3, c2=16, c3=32, c4=64, c5=128):
        super().__init__()
        # input [bs, 3, 2760, 3632]
        # input [bs, 3, 640, 640]
        self.Conv1 = Conv(c1=c1, c2=c2, s=2)  # [bs, 16, 320, 320]
        self.Conv2 = Conv(c1=c2, c2=c3, s=2)  # [bs, 32, 160, 160]
        self.Conv3 = C3k2(c1=c3, c2=c3)
        self.Conv4 = Conv(c1=c3, c2=c4, s=2)
        self.Conv5 = C3k2(c1=c4, c2=c4)
        self.Conv6 = Conv(c1=c4, c2=c5, s=2)
        self.Conv7 = C3k2(c1=c5, c2=c5)

        # DeConv
        self.Conv8 = Conv(c1=c5, c2=c5)
        self.Deconv1 = nn.ConvTranspose2d(in_channels=c5, out_channels=c4, kernel_size=2, stride=2)
        self.Deconv2 = nn.ConvTranspose2d(in_channels=c4, out_channels=c3, kernel_size=2, stride=2)

    def forward(self, x):
        c1 = self.Conv1(x)
        c2 = self.Conv2(c1)
        ck1 = self.Conv3(c2)
        x = self.Conv4(ck1)
        ck2 = self.Conv5(x)
        x = self.Conv6(ck2)
        x = self.Conv7(x)  # ck3

        x1 = self.Conv8(x)
        x2 = self.Deconv1(x1)
        x3 = self.Deconv2(x2)

        deep_feature = torch.cat((x, x1), dim=1)
        medium_feature = torch.cat((ck2, x2), dim=1)
        deep_shallow = torch.cat((ck1, x3), dim=1)

        return deep_feature, medium_feature, deep_shallow, c1, c2, ck1


class Classify(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super().__init__()
        c_ = 1280
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return x if self.training else x.softmax(1)


class Segment(nn.Module):
    def __init__(self, nc=1, in_c_deep=256, in_c_medium=128, in_c_x_ds=64):
        super().__init__()
        out_c = 16 if nc == 1 else 64

        self.upsample1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Deep feature branch
        self.deep_conv1 = Conv(c1=in_c_deep, c2=64, k=1)
        self.deep_conv2 = DWConv(c1=64, c2=32)
        self.deep_conv3 = Conv(c1=64, c2=out_c, k=3)

        # Medium feature branch
        self.medium_conv1 = Conv(c1=in_c_medium, c2=32, k=1)
        self.medium_conv2 = Conv(c1=48, c2=out_c, k=3)

        # Deep & Shallow branch
        self.deep_shallow_conv1 = Conv(c1=in_c_x_ds, c2=16, k=1)
        self.deep_shallow_conv2 = DWConv(c1=19, c2=19, k=3)
        self.deep_shallow_conv3 = Conv(c1=19, c2=out_c, k=1)

        # Final convolution to get the output
        self.final_conv = nn.Conv2d(out_c * 3, nc, kernel_size=1)

    def forward(self, deep_feature, medium_feature, deep_shallow, c1, c2, ck1, x):
        # Process deep feature
        x_deep = self.deep_conv1(deep_feature)
        x_deep = self.upsample1(x_deep)
        x_deep = self.deep_conv2(x_deep)
        x_deep = x_deep + ck1
        x_deep = torch.cat((x_deep, c2), dim=1)
        x_deep = self.deep_conv3(x_deep)
        x_deep = self.upsample1(x_deep)  # [4, 16/64, 640, 640]

        # Process medium feature
        x_medium = self.medium_conv1(medium_feature)
        x_medium = self.upsample1(x_medium)
        x_medium = torch.cat((x_medium, c1), dim=1)
        x_medium = self.medium_conv2(x_medium)
        x_medium = self.upsample2(x_medium)

        # Process deep and shallow combined feature
        x_ds = self.deep_shallow_conv1(deep_shallow)
        x_ds = self.upsample1(x_ds)
        x_ds = torch.cat((x_ds, x), dim=1)
        x_ds = self.deep_shallow_conv2(x_ds)
        x_ds = self.deep_shallow_conv3(x_ds)

        # Concatenate all processed features
        x = torch.cat([x_deep, x_medium, x_ds], dim=1)
        x = self.final_conv(x)

        return x


class LMV_cls(nn.Module):
    def __init__(self, cls_nc=4, seg_nc=1):
        super().__init__()
        self.backbone = Backbone()
        self.classify = Classify(c1=128, c2=cls_nc)

    def forward(self, x):
        deep_feature, medium_feature, deep_shallow, c1, c2, ck1 = self.backbone(x)
        classified_x = self.classify(medium_feature)

        return classified_x


class LMV_seg(nn.Module):
    def __init__(self, seg_nc=1):
        super().__init__()
        self.backbone = Backbone()
        self.segment = Segment(nc=seg_nc)

    def forward(self, x):
        deep_feature, medium_feature, deep_shallow, c1, c2, ck1 = self.backbone(x)
        segment_x = self.segment(deep_feature, medium_feature, deep_shallow, c1, c2, ck1, x)

        return segment_x


# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Running on device: {device}")
#     input_tensor = torch.randn(4, 3, 640, 640).to(device)

    # backbone = Backbone().to(device)
    # deep_feature, medium_feature, deep_shallow, c1, c2, ck1 = backbone(input_tensor)
    #
    # print("Output 1 shape:", deep_feature.shape)  # torch.Size([4, 256, 40, 40])
    # print("Output 2 shape:", medium_feature.shape)  # torch.Size([4, 128, 80, 80])
    # print("Output 3 shape:", deep_shallow.shape)  # torch.Size([4, 64, 160, 160])

    # model = LMV(nc=4).to(device)
    #
    # model.train()
    # output = model(input_tensor)
    # print("Output shape:", output.shape)  # 应该是 (batch_size, c2)，例如 (16, 10)
    #
    # model.eval()
    # output = model(input_tensor)
    # print("Output after softmax (if eval mode):", output)

    # seg = Segment(nc=1).to(device)
    # mask = seg(deep_feature, medium_feature, deep_shallow, c1, c2, ck1, input_tensor)

    # L_seg = LMV_seg(seg_nc=1).to(device)
    # mask = L_seg(input_tensor)
    #
    # print("seg mask shape:", mask.shape)
