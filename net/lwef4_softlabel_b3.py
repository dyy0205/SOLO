import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models import gen_efficientnet
except ModuleNotFoundError:
    from net.models import gen_efficientnet


class Encoder(nn.Module):
    def __init__(self, arch, pretrained=False):
        super(Encoder, self).__init__()
        self.backbone = gen_efficientnet.__getattribute__(arch)(2, pretrained=pretrained)
        del self.backbone.conv_head
        del self.backbone.global_pool
        del self.backbone.bn2
        del self.backbone.classifier
        if self.backbone.bn1 is not None:
            self.layer0 = [
                self.backbone.conv_stem,
                self.backbone.bn1,
                self.backbone.act_fn,
            ]
        else:
            self.layer0 = [
                self.backbone.conv_stem,
                self.backbone.act_fn,
            ]
        self.layers = list(self.backbone.blocks)

    def forward(self, x):
        lxx = []
        for l in self.layer0:
            x = l(x)
        lxx.append(x)
        for l in self.layers:
            x = l(x)
            lxx.append(x)
        return lxx


class RefineBlock(nn.Module):
    def __init__(self, dummy_from_x, dummy_to_x):
        super(RefineBlock, self).__init__()
        channel_from = dummy_from_x.size(1)
        channel_to = dummy_to_x.size(1)
        self.adapt_from = nn.Sequential(
            nn.Conv2d(channel_from, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
            # nn.ReLU(inplace=False),
            # nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            # nn.BatchNorm2d(channel_to),
        ) if channel_to!=channel_from else lambda x: x
        self.adapt_to = nn.Sequential(
            nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
        )
        self.refine_res = nn.Sequential(
            nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_to, channel_to, kernel_size=3, padding=1, bias=False, groups=channel_to),
            nn.BatchNorm2d(channel_to),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
        )
        self.refine_refine = nn.Sequential(
            nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_to, channel_to, kernel_size=3, padding=1, bias=False, groups=channel_to),
            nn.BatchNorm2d(channel_to),
            nn.ReLU(inplace=False),
            nn.Conv2d(channel_to, channel_to, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel_to),
        )

        if dummy_to_x.size(2) % dummy_from_x.size(2) == 0 and dummy_to_x.size(3) % dummy_from_x.size(3) == 0:
            scale_factor = dummy_to_x.size(2) // dummy_from_x.size(2)
            self.upsample = lambda x: F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
        else:
            self.upsample = lambda x: x

    def forward(self, from_x, to_x):
        from_x = self.adapt_from(from_x)
        from_x = self.upsample(from_x)
        to_x = self.adapt_to(to_x)
        to_x = from_x + self.refine_res(from_x + to_x)
        to_x = to_x + self.refine_refine(to_x)
        return to_x

dummy_width = 512
dummy_x = torch.rand(1, 3, dummy_width, dummy_width)  # , device=encoder.device)

class Decoder(nn.Module):
    def __init__(self, encoder, num_classes):
        super(Decoder, self).__init__()
        self.last_stage = 0
        with torch.no_grad():
            dummy_features = encoder(dummy_x)

        # All features
        # print([x.size() for x in dummy_features])
        # Select useful features
        self.feature_stage = [1, 2, 3, 5, 7]
        dummy_features = [dummy_features[i] for i in self.feature_stage]
        # Refine every stage
        self.refine_stages = []
        self.output_stages = []

        self.output_stages.append(
            nn.Conv2d(dummy_features[-1].size(1), num_classes, kernel_size=3, padding=1, bias=False),
        )
        for i in range(len(self.feature_stage) - 1, 0, -1):
            # print(dummy_features[i - 1].shape, dummy_features[i].shape)
            self.refine_stages.append(RefineBlock(dummy_features[i], dummy_features[i - 1]))
            dummy_features[i - 1] = self.refine_stages[len(self.feature_stage) - 1 - i](dummy_features[i], dummy_features[i - 1])
            # print(dummy_features[i - 1].shape)
            self.output_stages.append(
                nn.Conv2d(dummy_features[i - 1].size(1), num_classes, kernel_size=3, padding=1, bias=False),
            )
        self.output_stages_seq = nn.Sequential(*self.output_stages)
        self.refine_stages_seq = nn.Sequential(*self.refine_stages)

    def forward(self, features):
        features = [features[i] for i in self.feature_stage]

        outs = []
        outs.append(self.output_stages[0](features[-1]))
        for i in range(len(self.feature_stage) - 1, 0, -1):
            features[i - 1] = self.refine_stages[len(self.feature_stage) - 1 - i](features[i], features[i - 1])
            outs.append(self.output_stages[len(self.feature_stage) - i](features[i - 1]))
        return outs


def _initialize_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.01)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


class LWef(nn.Module):
    def __init__(self, num_classes, arch='tflite_mnasnet_100', pretrained=False):
        super(LWef, self).__init__()
        self.encoder = Encoder(arch, pretrained)
        self.decoder = Decoder(self.encoder, num_classes)
        # self.fix_encoder = False
        # self.input_mean = -0.449
        # self.input_std = 1. / 0.226
        # if pretrained:
        #     _initialize_weights(self.decoder.modules())
        # else:
        #     _initialize_weights(self.modules())
        # print('success')

    def forward(self, x):
        # x = (x + self.input_mean) * self.input_std
        features = self.encoder(x)
        outs = self.decoder(features)
        out_segm = F.interpolate(outs[-1], scale_factor=2, mode='bilinear', align_corners=False)

        out_segm = F.softmax(out_segm, 1).split(1, 1)[1]
        post_avgpool_kernel = 2
        out_segm = F.max_pool2d(-out_segm, post_avgpool_kernel, 1, 0)
        out_segm = out_segm * -255
        out_segm = torch.clamp(out_segm, 0, 255)
        return out_segm


if __name__ == '__main__':
    human_ckpt_path = '/home/dingyangyang/human_append/ckpt/ckpt_519_b3_512_bili2.ckpt'
    human_checkpoint = torch.load(human_ckpt_path)
    human_net = LWef(2, arch='tf_efficientnet_b3', pretrained=False)
    print('loading human checkpoint %s' % (human_ckpt_path))
    human_net.load_state_dict(human_checkpoint['state_dict'], strict=False)
    human_net.cuda().eval()
