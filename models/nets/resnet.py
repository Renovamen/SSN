from torch import nn
from torch.nn import functional as F

from ..param_bank import *
from .utils import Conv2d, Linear


def conv3x3(in_planes, out_planes, stride=1, bank=None):
    return Conv2d(
        bank = bank,
        in_channels = in_planes,
        out_channels = out_planes,
        kernel_size = 3,
        stride = stride,
        padding = 1,
        bias = False
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, bank=None):
        super().__init__()

        self.conv1 = conv3x3(in_planes, planes, stride=stride, bank=bank)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, bank=bank)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(
                    bank = bank,
                    in_channels = in_planes,
                    out_channels = self.expansion * planes,
                    kernel_size = 1,
                    stride = stride,
                    bias = False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        nf,
        input_size,
        num_classes,
        groups,
        share_type: str,
        upsample_type,
        upsample_window,
        max_params,
        num_templates
    ):
        super().__init__()

        self.in_planes = nf
        self.input_size = input_size
        self.num_classes = num_classes

        self.bank = None
        if share_type != 'none':
            self.bank = ParameterGroups(groups, share_type, upsample_type, upsample_window, max_params, num_templates)

        self.conv1 = conv3x3(input_size[0], nf * 1, bank=self.bank)
        self.bn1 = nn.BatchNorm2d(nf * 1)

        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # hardcoded for now
        self.last_hid_size = nf * 8 * block.expansion if input_size[1] in [8, 16, 21, 32, 42] else 640
        self.linear = Linear(
            bank = self.bank,
            in_features = self.last_hid_size,
            out_features = num_classes
        )

        if self.bank:
            self.bank.setup_bank()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bank=self.bank))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x):
        bsz = x.size(0)
        #pre_bn = self.conv1(x.view(bsz, 3, 32, 32))
        #post_bn = self.bn1(pre_bn, 1 if is_real else 0)
        #out = F.relu(post_bn)
        out = F.relu(self.bn1(self.conv1(x.view(bsz, *self.input_size))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x):
        out = self.return_hidden(x)
        out = self.linear(out)
        return out


def resnet18(
    share_type,
    upsample_type,
    upsample_window,
    max_params,
    num_templates,
    nf = 20,
    input_size = (3, 32, 32),
    num_classes = 10,
    groups = None
):
    return ResNet(
        block = BasicBlock,
        num_blocks = [2, 2, 2, 2],
        nf = nf,
        input_size = input_size,
        num_classes = num_classes,
        groups = groups,
        share_type = share_type,
        upsample_type = upsample_type,
        upsample_window = upsample_window,
        max_params = max_params,
        num_templates = num_templates
    )
