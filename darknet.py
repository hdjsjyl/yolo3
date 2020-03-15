
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

def parse_cfg(cfg_file):
    """
    Takes a configuration file and interpret it

    return a list of blocks each block desccribes a block in the neural network
    to be built. Block is represented as a dictionary in the list
    :param cfg_file:
    :return:
    """

    file = open(cfg_file, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) != 0]   # remove empty line
    lines = [x for x in lines if x[0] != '#']   # remove comments line
    lines = [x.strip() for x in lines]          # remove fringe whitespaces

    blocks = []
    block = {}
    for line in lines:
        if line[0] == '[':
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block['type'] = line[1:-1].strip()
        else:
            key, value = line.split('=')
            block[key.strip()] = value.strip()

    if len(block) != 0:
        blocks.append(block)

    return blocks


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def create_modules(blocks):
    net_info = blocks[0]            # it includes the pre-processing and input information
    module_list = nn.ModuleList()
    prev_filters = 3                # it is the channels of input image as previous filter size
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        # check the type of block, create a new module for the block, append it to module_list

        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['bacth_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['size'])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # add convolutional layer
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module('conv_{index}'.format(index=index), conv)

            # add batch norm layer
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{index}'.format(index=index), bn)

            # check the activation; it is either linear or a leaky relu for yolo
            if activation == 'leaky':
                activ = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{index}'.format(index=index), activ)

        # if it is an upsampling layer, we use bilinear2dupsampling
        elif x['type'] == 'upsampling':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=stride, mode='bilinear')
            module.add_module('upsample_{index}'.format(index=index), upsample)

        # if it is a route layer, concatenation of feature maps
        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            # start of a route
            start = int(x['layers'][0].strip())
            # end, if exists
            try:
                end = int(x['layers'][1].strip())
            except:
                end = 0

            if start > 0:
                start -= index
            if end > 0:
                end -= index
                route = EmptyLayer()
                module.add_module('route_{index}'.format(index=index), route)
            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]
            
        # if it is a shortcut layer, element-wise-sum operation of feature maps
        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{index}'.format(index=index), shortcut)
            
        
        # yolo is the detection layer
        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            anchors = x['anchors'].split(',')
            anchors = [int(x) for x in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('Detection_{index}'.format(index=index), detection)


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list




blocks = parse_cfg('./yolov3.cfg')
blocks_list = create_modules(blocks)
print(len(blocks_list))
print(blocks_list)