import math

from os.path import join as pjoin
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_plugin_layer

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1, plugins=None):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3', 'after_res']
            assert all(p['position'] in allowed_position for p in plugins)
        self.with_plugins = plugins is not None
        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]
            self.after_res_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_res'
            ]
        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                cmid, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                cmid, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                cout, self.after_conv3_plugins)
            self.after_res_plugin_names = self.make_block_plugins(
                cout, self.after_res_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """make plugins for block.

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.
        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        """Forward function for plugins."""
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        if self.with_plugins:
            y = self.forward_plugin(y, self.after_conv1_plugin_names)
        y = self.relu(self.gn2(self.conv2(y)))
        if self.with_plugins:
            y = self.forward_plugin(y, self.after_conv2_plugin_names)
        y = self.gn3(self.conv3(y))
        if self.with_plugins:
            y = self.forward_plugin(y, self.after_conv3_plugin_names)

        y = self.relu(residual + y)
        if self.with_plugins:
            y = self.forward_plugin(y, self.after_res_plugin_names)

        return y

class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor, in_channels=3, output_tuple=False, plugins=None):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.output_tuple = output_tuple
        self.num_stages = 3
        self.plugins = plugins
        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(in_channels, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        stage_plugins = []
        for i in range(self.num_stages):
            stage_plugins.append(
                self.make_stage_plugins(plugins, i) if plugins is not None else None)

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width,
                    cout=width*4,
                    cmid=width,
                    plugins=stage_plugins[0]))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*4,
                    cout=width*4,
                    cmid=width,
                    plugins=stage_plugins[0]))
                 for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width*4,
                    cout=width*8,
                    cmid=width*2,
                    stride=2,
                    plugins=stage_plugins[1]))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*8,
                    cout=width*8,
                    cmid=width*2,
                    plugins=stage_plugins[1]))
                 for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(
                    cin=width*8,
                    cout=width*16,
                    cmid=width*4,
                    stride=2,
                    plugins=stage_plugins[2]))] +
                [(f'unit{i:d}', PreActBottleneck(
                    cin=width*16,
                    cout=width*16,
                    cmid=width*4,
                    plugins=stage_plugins[2]))
                 for i in range(2, block_units[2] + 1)],
                ))),
        ]))
        from mmengine.logging import MMLogger, print_log
        logger: MMLogger = MMLogger.get_current_instance()
        print_log('Print the model:', logger)
        print_log('\n' + str(self), logger=logger)

    def make_stage_plugins(self, plugins, stage_idx):
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        if self.output_tuple:
            features.append(x)
            return tuple(features)
        else:
            return x, features[::-1]