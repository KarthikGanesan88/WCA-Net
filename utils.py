import os

import torch

# eps_names_mnist = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5']
# eps_values_mnist = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# eps_names_cifar = ['  0/255', '  1/255', '  2/255', '  4/255', '  8/255', ' 16/255', ' 32/255', ' 64/255', '128/255']
# eps_values_cifar = [0. / 255, 1. / 255, 2. / 255, 4. / 255, 8. / 255, 16. / 255, 32. / 255, 64. / 255, 128. / 255]
eps_names_mnist = ['0.3']
eps_values_mnist = [0.3]
eps_names_cifar = ['  8/255']
eps_values_cifar = [8. / 255]

dataset_to_attack_strength = {
    'mnist': {'eps_names': eps_names_mnist, 'eps_values': eps_values_mnist},
    'fmnist': {'eps_names': eps_names_mnist, 'eps_values': eps_values_mnist},
    'cifar10': {'eps_names': eps_names_cifar, 'eps_values': eps_values_cifar},
    'cifar100': {'eps_names': eps_names_cifar, 'eps_values': eps_values_cifar},
    'svhn': {'eps_names': eps_names_cifar, 'eps_values': eps_values_cifar},
}

attack_to_dataset_config = {
    'FGSM': dataset_to_attack_strength,
    'PGD': dataset_to_attack_strength,
    'BIM': dataset_to_attack_strength,
    'C&W': {'cifar10': {'eps_names': ['None'], 'eps_values': [None]}},
    'Few-Pixel': {'cifar10': {'eps_names': ['1p', '2p', '3p'], 'eps_values': [1, 2, 3]}},
}


mean_cifar10 = (0.4914, 0.4822, 0.4465)
std_cifar10 = (0.2023, 0.1994, 0.2010)

mean_cifar100 = (0.5071, 0.4867, 0.4408)
std_cifar100 = (0.2675, 0.2565, 0.2761)

mean_generic = (0.5, 0.5, 0.5)
std_generic = (0.5, 0.5, 0.5)

def print_log(logfile, print_string, log_only=False):
    if not log_only:
        print("{}".format(print_string))

    log = open(logfile, 'a')
    log.write('{}\n'.format(print_string))
    log.flush()
    log.close()

def normalize_cifar10(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar10[0]) / std_cifar10[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar10[1]) / std_cifar10[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar10[2]) / std_cifar10[2]
    return t


def normalize_cifar100(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_cifar100[0]) / std_cifar100[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_cifar100[1]) / std_cifar100[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_cifar100[2]) / std_cifar100[2]
    return t


def normalize_generic(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean_generic[0]) / std_generic[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean_generic[1]) / std_generic[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean_generic[2]) / std_generic[2]
    return t


########################################################
#   Modify network layers to use custom versions.
#    Supports converting the model to use approximate versions
#    as well as PNI layer versions.
#########################################################

import torch.nn as nn
from custom.custom_conv import CustomConv
from custom.custom_fc import CustomLinear

def modify_layers(model):
    # Dict of custom layers, selected based on layer type and model type.
    new_layers = {'conv': CustomConv, 'fc': CustomLinear}

    for name, layer in model.named_modules():
        # print('layer_name:', name)
        layer_type = None
        if isinstance(layer, nn.Conv2d):
            layer_type = 'conv'
            layer_params = ['dilation', 'kernel_size', 'padding', 'stride',
                            'out_channels', 'in_channels']

        elif isinstance(layer, nn.Linear):
            layer_type = 'fc'
            layer_params = ['in_features', 'out_features']

        if layer_type is not None:
            # For sequential and custom blocks, need to get the pointer to the
            # actual layer to get its parameters and set parameters for the new layer.
            name_parts = name.split('.')

            # Get the module that this layer belongs to. Could be the model itself,
            # a sequential layer or custom block.
            base_module = model
            for part in name_parts[:-1]:
                base_module = getattr(base_module, part)

            saved_params = {}

            layer_module = getattr(base_module, name_parts[-1])

            for param in layer_params:
                saved_params[param] = getattr(layer_module, param)

            # Saving weight and bias separately since they are not used
            # when initializing the layer.
            saved_weight = getattr(layer_module, 'weight')
            saved_bias = getattr(layer_module, 'bias')

            # Create the new layer with these parameters.
            # new_layer = AppxConv2d(saved_params['in_channels'],
            #                        saved_params['out_channels'],
            #                        saved_params['kernel_size'],
            #                        stride=saved_params['stride'],
            #                        padding=saved_params['padding'],
            #                        dilation=saved_params['dilation'])
            new_layer = new_layers[layer_type](**saved_params)

            new_layer.weight = saved_weight
            new_layer.bias = saved_bias

            # Replace the old layer with the new one in the model.
            setattr(base_module, name_parts[-1], new_layer)
