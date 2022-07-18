import torch
import torch.nn as nn
from collections import OrderedDict
import copy

from custom.quantized_convolution_layer import QuantConv2D
from custom.quantized_fully_connected_layer import QuantLinear

def create_model_copy(net, layer_type, model_type='appx'):
    """
    Create a copy of the original model, replacing the specified layers with custom versions.
    :param net: The original network to copy.
    :param layer_type: List of layer types to replace. ['Conv2d', 'Linear', 'BatchNorm2d', 'ReLU']
    :param model_type: Convert the model to 'appx' or 'custom'.
    :return: Copied model.
    """
    if model_type == 'appx':
        new_conv_layer = 'MyConv2d'
        new_linear_layer = 'MyLinear'
        # new_bn_layer = 'MyBatchNorm'
        new_relu_layer = 'Custom_ReLU'
    elif model_type == 'custom':
        new_conv_layer = 'QuantConv2D'
        new_linear_layer = 'QuantLinear'
        # new_bn_layer = 'QuantBatchNorm'
        new_relu_layer = 'Custom_ReLU'
    else:
        print('ERROR: Can only create an appx or custom copy of the model')
        return

    model = copy.deepcopy(net)

    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and ("Conv2D" in layer_type):
            dilation = model._modules[name].dilation
            kernel_size = model._modules[name].kernel_size
            padding = model._modules[name].padding
            stride = model._modules[name].stride
            out_channels = model._modules[name].out_channels
            in_channels = model._modules[name].in_channels
            weight = model._modules[name].weight
            bias = model._modules[name].bias

            model._modules[name] = globals()[new_conv_layer](in_channels,
                                                             out_channels,
                                                             kernel_size,
                                                             stride=stride,
                                                             padding=padding,
                                                             dilation=dilation)

            model._modules[name].weight = weight
            model._modules[name].bias = bias

        elif isinstance(layer, nn.Linear) and ("Linear" in layer_type):
            in_features = model._modules[name].in_features
            out_features = model._modules[name].out_features
            weight = model._modules[name].weight
            bias = model._modules[name].bias
            model._modules[name] = globals()[new_linear_layer](in_features, out_features)

            model._modules[name].weight = weight
            model._modules[name].bias = bias

        '''
        elif isinstance(layer, nn.BatchNorm2d) and ("BatchNorm2d" in layer_type):
            num_features = model._modules[name].num_features
            eps = model._modules[name].eps
            momentum = model._modules[name].momentum
            affine = model._modules[name].affine
            track_running_stats = model._modules[name].track_running_stats
            running_mean = model._modules[name].running_mean
            running_var = model._modules[name].running_var
            weight = model._modules[name].weight
            bias = model._modules[name].bias

            model._modules[name] = globals()[new_bn_layer](num_features,
                                                           eps=eps,
                                                           momentum=momentum,
                                                           affine=affine,
                                                           track_running_stats=track_running_stats)
            model._modules[name].weight = weight
            model._modules[name].bias = bias
            model._modules[name].running_mean = running_mean
            model._modules[name].running_var = running_var
            '''
    return model


def copy_state_dict(source_model):
    # Get the list of layers first. Need this to know how to propagate scale and zero_point
    # to the next layer along.

    layer_list = []
    for name, layer in source_model.named_modules():
        # Ignore relu layers and packed params for FC layers
        if "relu" not in name and "_packed_params" not in name and name != '':
            layer_list.append(name)

    source_state_dict = source_model.state_dict()
    new_state_dict = OrderedDict()

    for key in source_state_dict:
        layer_name, layer_type = key.split('.', 1)
        # Since custom is always the first layer, it's values get set as the
        # in values of the first layer in the list.
        next_layer = layer_list[(layer_list.index(layer_name) + 1)]
        # print(f'Current Layer:{layer_name}, Next Layer:{next_layer}')
        if layer_name == 'custom':
            new_state_dict[next_layer + '.in_' + layer_type] = source_state_dict[key]
        if 'conv' in layer_name:
            if layer_type in ['scale', 'zero_point']:
                # values in the layer are the output values of the layer.
                # So they should be pushed as the in values of the next layer in the list.
                new_state_dict[next_layer + '.in_' + layer_type] = source_state_dict[key]
            if layer_type == 'weight':
                # Scale in the weight is for the weights themselves.
                weight = source_state_dict[key]
                new_state_dict[layer_name + '.wt_scale'] = torch.tensor(weight.q_scale())
                new_state_dict[layer_name + '.wt_zero_point'] = torch.tensor(weight.q_zero_point())
                new_state_dict[layer_name + '.weight'] = torch.dequantize(weight)
            if layer_type == 'bias':
                new_state_dict[layer_name + '.' + layer_type] = source_state_dict[key]
        elif 'fc' in layer_name:
            if layer_type in ['scale', 'zero_point']:
                # For now, doesn't seem to be a point setting params for dequant as the
                # output from the last layer should already be dequantized anyway.
                if next_layer != 'dequant':
                    new_state_dict[next_layer + '.in_' + layer_type] = source_state_dict[key]
            if layer_type == '_packed_params._packed_params':
                weight, new_state_dict[layer_name + '.bias'] = source_state_dict[key]
                new_state_dict[layer_name + '.wt_scale'] = torch.tensor(weight.q_scale())
                new_state_dict[layer_name + '.wt_zero_point'] = torch.tensor(weight.q_zero_point())
                new_state_dict[layer_name + '.weight'] = torch.dequantize(weight)
        # else:
        #     print("Unrecognized parameter in source state_dict().")

    return new_state_dict
