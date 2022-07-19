import json
import os
import sys
from datetime import datetime

import torch
from tqdm.auto import tqdm

from attacks.one_pixel import one_pixel_attack
from data_loaders import get_data_loader
from models.model_list import model_factory
from test import test_attack
from train import train_vanilla, train_stochastic, train_stochastic_adversarial, get_norm_func
from utils import attack_to_dataset_config, print_log, modify_layers
import metrics
from metrics import accuracy as accuracy
# from resnet_folded import VanillaResNet18_folded

from custom.custom_conv import CustomConv
from custom.custom_fc import CustomLinear

def parse_args():
    mode = sys.argv[1]
    if mode not in ('train', 'test', 'train+test', 'quantize', 'test_multiple'):
        raise ValueError()
    config_file = sys.argv[2]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())
    return mode, args


"""
# Code to try to quantize using pytorch directly. But doesn't work very well.
class quant_model(torch.nn.Module):
    def __init__(self, model_base):
        super().__init__()
        self.custom = torch.quantization.QuantStub()
        self.model = model_base
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.custom(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

def quantize(args, device):
    print(args)
    print('MODE: Quantize')
    model = model_factory(
        args['dataset'], args['training_type'], args['var_type'], args['feature_dim'], args['num_classes'])
    model.to(device)
    # model.load(os.path.join(args['output_path']['models'], 'ckpt_best'))
    model.load(os.path.join(args['output_path']['models'], 'ckpt_last'))

    model_fp32 = quant_model(model_base=model)
    model_fp32.eval()

    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)
    data_norm = get_norm_func(args)
    test_acc = accuracy(model_fp32, test_loader, device=device, norm=data_norm)
    print(f'FP32 test accuracy: {100. * test_acc:.3f}%')

    # for key in model_fp32.state_dict().keys():
    #     print(key)
    # breakpoint()

    model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
                                                       [['model.gen.rn.conv1', 'model.gen.rn.bn1'],
                                                        ['model.gen.rn.layer1.0.conv1', 'model.gen.rn.layer1.0.bn1'],
                                                        ['model.gen.rn.layer1.0.conv2', 'model.gen.rn.layer1.0.bn2'],
                                                        ['model.gen.rn.layer1.1.conv1', 'model.gen.rn.layer1.1.bn1'],
                                                        ['model.gen.rn.layer1.1.conv2', 'model.gen.rn.layer1.1.bn2'],
                                                        ['model.gen.rn.layer2.0.conv1', 'model.gen.rn.layer2.0.bn1'],
                                                        ['model.gen.rn.layer2.0.conv2', 'model.gen.rn.layer2.0.bn2'],
                                                        ['model.gen.rn.layer2.1.conv1', 'model.gen.rn.layer2.1.bn1'],
                                                        ['model.gen.rn.layer2.1.conv2', 'model.gen.rn.layer2.1.bn2'],
                                                        ['model.gen.rn.layer3.0.conv1', 'model.gen.rn.layer3.0.bn1'],
                                                        ['model.gen.rn.layer3.0.conv2', 'model.gen.rn.layer3.0.bn2'],
                                                        ['model.gen.rn.layer3.1.conv1', 'model.gen.rn.layer3.1.bn1'],
                                                        ['model.gen.rn.layer3.1.conv2', 'model.gen.rn.layer3.1.bn2'],
                                                        ['model.gen.rn.layer4.0.conv1', 'model.gen.rn.layer4.0.bn1'],
                                                        ['model.gen.rn.layer4.0.conv2', 'model.gen.rn.layer4.0.bn2'],
                                                        ['model.gen.rn.layer4.1.conv1', 'model.gen.rn.layer4.1.bn1'],
                                                        ['model.gen.rn.layer4.1.conv2', 'model.gen.rn.layer4.1.bn2'],
                                                        ])
    model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

    for data, target in tqdm(test_loader):
        data = data.to(device)
        if data_norm is not None:
            data = data_norm(data)
        logits = model_fp32_prepared(data)

    model_fp32_prepared = model_fp32_prepared.cpu()
    model_int8 = torch.quantization.convert(model_fp32_prepared)
    model_int8 = model_int8.cpu()

    for key, value in model_int8.state_dict().items():
        if 'scale' in key.split('.') or 'zero_point' in key.split('.'):
            print(key, value)
    breakpoint()
"""


def quantize(args, device):
    print(args)
    print('MODE: Quantize')
    model = model_factory(args['model'],
                          args['dataset'], args['training_type'], args['var_type'], args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    model_path = os.path.join(f"./output/models/wcanet_cifar10_m0", 'ckpt_last')
    model.load(model_path)

    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)
    data_norm = get_norm_func(args)
    # test_acc = accuracy(model, test_loader, device=device, norm=data_norm)
    # print(f'FP32 test accuracy: {100. * test_acc:.3f}%')

    modify_layers(model)

    for name, layer in model.named_modules():
        if isinstance(layer, CustomConv) or isinstance(layer, CustomLinear):
            layer.appx_mode = torch.Tensor([1]).cuda()
            # print(f'{name}: Appx mode:{layer.appx_mode}')

    test_acc = accuracy(model, test_loader, device=device, norm=data_norm)
    print(f'Custom test accuracy: {100. * test_acc:.3f}%')

    breakpoint()

    model_fused = VanillaResNet18_folded(args['feature_dim'], args['num_classes'])
    new_state_dict = model_fused.state_dict()

    # First fold Conv2D and BatchNorm2D layers.
    layers_to_fuse = [
        ['gen.rn.conv1', 'gen.rn.bn1'],
        ['gen.rn.layer1.0.conv1', 'gen.rn.layer1.0.bn1'],
        ['gen.rn.layer1.0.conv2', 'gen.rn.layer1.0.bn2'],
        ['gen.rn.layer1.1.conv1', 'gen.rn.layer1.1.bn1'],
        ['gen.rn.layer1.1.conv2', 'gen.rn.layer1.1.bn2'],
        ['gen.rn.layer2.0.conv1', 'gen.rn.layer2.0.bn1'],
        ['gen.rn.layer2.0.conv2', 'gen.rn.layer2.0.bn2'],
        ['gen.rn.layer2.0.shortcut.0', 'gen.rn.layer2.0.shortcut.1'],
        ['gen.rn.layer2.1.conv1', 'gen.rn.layer2.1.bn1'],
        ['gen.rn.layer2.1.conv2', 'gen.rn.layer2.1.bn2'],
        ['gen.rn.layer3.0.conv1', 'gen.rn.layer3.0.bn1'],
        ['gen.rn.layer3.0.conv2', 'gen.rn.layer3.0.bn2'],
        ['gen.rn.layer3.0.shortcut.0', 'gen.rn.layer3.0.shortcut.1'],
        ['gen.rn.layer3.1.conv1', 'gen.rn.layer3.1.bn1'],
        ['gen.rn.layer3.1.conv2', 'gen.rn.layer3.1.bn2'],
        ['gen.rn.layer4.0.conv1', 'gen.rn.layer4.0.bn1'],
        ['gen.rn.layer4.0.conv2', 'gen.rn.layer4.0.bn2'],
        ['gen.rn.layer4.1.conv1', 'gen.rn.layer4.1.bn1'],
        ['gen.rn.layer4.1.conv2', 'gen.rn.layer4.1.bn2']]

    for key, value in model.state_dict().items():
        for layer in layers_to_fuse:
            if key == layer[0] + '.weight':
                # conv_weight = (model.state_dict()[layer[0] + '.weight'], model.state_dict()[layer[0] + '.bias'])
                conv_weight = model.state_dict()[layer[0] + '.weight']
                conv_bias = torch.zeros(conv_weight.shape[0]).cuda()
                bn_weight = model.state_dict()[layer[1] + '.weight']
                bn_bias = model.state_dict()[layer[1] + '.bias']
                bn_mean = model.state_dict()[layer[1] + '.running_mean']
                bn_var = model.state_dict()[layer[1] + '.running_var']
                # print(conv_weight[0].shape, conv_weight[1].shape)
                # print(bn_weight[0].shape, bn_weight[1].shape, bn_weight[2].shape, bn_weight[3].shape)

                new_weight, new_bias = fold_conv_and_bn(conv_weight, conv_bias, bn_weight, bn_bias,
                                                        bn_mean, bn_var)
                # Set the new weight and new bias for the corresponding layer in the new model.
                new_state_dict[layer[0] + '.weight'] = new_weight
                new_state_dict[layer[0] + '.bias'] = new_bias

    remaining_layers = ['fc1.weight', 'fc1.bias', 'proto.weight', 'proto.bias']
    for key, value in model.state_dict().items():
        if key in remaining_layers:
            new_state_dict[key] = value

    model_fused.load_state_dict(new_state_dict)
    model_fused.cuda()
    test_acc = accuracy(model_fused, test_loader, device=device, norm=data_norm)
    print(f'Quant test accuracy: {100. * test_acc:.3f}%')

    # Replace the layers in this model with the custom versions.
    modify_layers(model_fused)

    # for name, layer in model_fused.named_modules():
    #     print(name, type(layer))

    test_acc = accuracy(model_fused, test_loader, device=device, norm=data_norm)
    print(f'Custom test accuracy: {100. * test_acc:.3f}%')

    breakpoint()


# Run the PyTorch custom code with fusing to get the right
# Weights and bias or conv + bn layers and check if the below is right.

def fold_conv_and_bn(conv_weight, conv_bias, bn_weight, bn_bias, bn_mean, bn_var, epsilon=1e-5):
    norm_factor = bn_weight / torch.sqrt(bn_var) + epsilon
    new_weight = conv_weight * torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(norm_factor, 1), 1), 1)

    new_bias = bn_bias + norm_factor * (conv_bias - bn_mean)

    '''
    # prepare filters
    w_conv = conv_weight.clone().view(conv_weight.shape[0], -1)
    w_bn = torch.diag(bn_weight.div(torch.sqrt(bn_var + epsilon)))

    new_weight = (torch.mm(w_bn, w_conv).view(conv_weight.size()))

    # prepare spatial bias
    if conv_bias is not None:
        b_conv = conv_bias
    else:
        b_conv = torch.zeros(conv_weight.size(0))

    b_bn = bn_bias - bn_weight.mul(bn_mean).div(torch.sqrt(bn_var + epsilon))
    new_bias = torch.matmul(w_bn, b_conv) + b_bn
    '''

    return new_weight, new_bias
    '''
    gamma = bn_weights[0].reshape((bn_weights[0].shape[0], 1, 1, 1))
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3].reshape((bn_weights[3].shape[0], 1, 1, 1))

    print('gamma', gamma.shape, 'beta', beta.shape, 'mean', mean.shape, 'variance', variance.shape)

    a = variance + epsilon
    b = torch.sqrt_(a)
    c = conv_weights[0] * gamma
    d = c / b

    new_weights = (conv_weights[0] * gamma / torch.sqrt_(variance + epsilon))
    new_bias = beta + (conv_weights[1] - mean) * gamma / torch.sqrt_(variance + epsilon)

    print('new weights', new_weights.shape, 'new bias', new_bias.shape)

    breakpoint()
    return new_weights, new_bias
    '''


def train(args, device):
    print(args)

    train_type = args['var_type'] if args['training_type'] == "stochastic" else args['training_type']
    model_path = os.path.join(
        f"./output/{args['model']}_{args['dataset']}_{train_type}_{args['feature_dim']}")

    os.makedirs(model_path, exist_ok=True)
    logfile = os.path.join(model_path, f'{datetime.now().strftime("%Y_%M_%d_%H_%M_%S")}.log')
    log = open(logfile, 'a')
    for key, value in args.items():
        log.write(f'{key}: {value}, ')
    log.write('\n')
    log.flush()
    log.close()

    train_loader = get_data_loader(args['dataset'], args['batch_size'], train=True, shuffle=True, drop_last=True)
    test_loader = get_data_loader(args['dataset'], args['batch_size'], train=False, shuffle=False, drop_last=False)
    model = model_factory(args['model'],
                          args['dataset'], args['training_type'], args['var_type'], args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    if args['pretrained'] is not None:
        if args['pretrained'] not in ('ckpt_best', 'ckpt_last', 'ckpt_robust'):
            raise ValueError('Pre-trained model name must be: [ckpt_best|ckpt_last|ckpt_robust]')
        model.load(os.path.join(model_path, args['pretrained']))
    if args['training_type'] == 'vanilla':
        print_log(logfile, 'Vanilla training.')
        train_vanilla(model, train_loader, test_loader, args, model_path, logfile, device=device)
    elif args['training_type'] == 'stochastic':
        print_log(logfile, 'Stochastic training.')
        train_stochastic(model, train_loader, test_loader, args, model_path, logfile, device=device)
    elif args['training_type'] == 'stochastic+adversarial':
        print_log(logfile, 'Adversarial stochastic training.')
        train_stochastic_adversarial(model, train_loader, test_loader, args, model_path, logfile, device=device)
    else:
        raise NotImplementedError(
            'Training "{}" not implemented. Supported: [vanilla|stochastic|stochastic+adversarial].'.format(
                args['training_type']))
    print_log(logfile, 'Finished training.')

def test_multiple(args, device):
    print(args)
    model = model_factory(args['model'],
                          args['dataset'], args['training_type'], args['var_type'], args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    train_type = args['var_type'] if args['training_type'] == "stochastic" else args['training_type']
    model_path = os.path.join(
        f"./output/{args['model']}_{args['dataset']}_{train_type}_{args['feature_dim']}")

    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)

    # To test multiple models in one run.
    for model_num in range(50, args['num_epochs'], 50):
        print(f'Running model saved at epoch: {model_num}')
        model.load(os.path.join(model_path, f'ckpt_{model_num}'))
        model.eval()

        test_acc = metrics.accuracy(model, test_loader, device=device, norm=get_norm_func(args))
        print(f'Accuracy: {100. * test_acc:.3f}%\n\n')

        attack_names = ['FGSM']  # 'BIM', 'C&W', 'Few-Pixel'
        print('Adversarial testing.')
        for idx, attack in enumerate(attack_names):
            print('Attack: {}'.format(attack))
            if attack == 'Few-Pixel':
                if args['dataset'] == 'cifar10':
                    preproc = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
                else:
                    raise NotImplementedError('Only CIFAR-10 supported for the one-pixel attack.')
                one_pixel_attack(
                    model, test_loader, preproc, device, pixels=1, targeted=False, maxiter=1000, popsize=400,
                    verbose=False)
            else:
                eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
                eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
                robust_accuracy = test_attack(model, test_loader, attack, eps_values, args, device)
                for eps_name, eps_value, acc in zip(eps_names, eps_values, robust_accuracy):
                    print('Attack Strength: {}, Accuracy: {:.3f}%'.format(eps_name, 100. * acc.item()))
        print('Finished testing.')

def test(args, device):
    print(args)
    model = model_factory(args['model'],
                          args['dataset'], args['training_type'], args['var_type'], args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    train_type = args['var_type'] if args['training_type'] == "stochastic" else args['training_type']
    model_path = os.path.join(
        f"./output/{args['model']}_{args['dataset']}_{train_type}_{args['feature_dim']}")

    # model.load(os.path.join(model_path, 'ckpt_best'))
    model.load(os.path.join(model_path, 'ckpt_last'))
    model.eval()
    test_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)

    test_acc = metrics.accuracy(model, test_loader, device=device, norm=get_norm_func(args))
    print(f'Accuracy: {100. * test_acc:.3f}%')

    attack_names = ['FGSM', 'PGD']  # 'BIM', 'C&W', 'Few-Pixel'
    print('Adversarial testing.')
    for idx, attack in enumerate(attack_names):
        print('Attack: {}'.format(attack))
        if attack == 'Few-Pixel':
            if args['dataset'] == 'cifar10':
                preproc = {'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]}
            else:
                raise NotImplementedError('Only CIFAR-10 supported for the one-pixel attack.')
            one_pixel_attack(
                model, test_loader, preproc, device, pixels=1, targeted=False, maxiter=1000, popsize=400, verbose=False)
        else:
            eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
            eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
            robust_accuracy = test_attack(model, test_loader, attack, eps_values, args, device)
            for eps_name, eps_value, acc in zip(eps_names, eps_values, robust_accuracy):
                print('Attack Strength: {}, Accuracy: {:.3f}%'.format(eps_name, 100. * acc.item()))
    print('Finished testing.')


def main(mode, args):
    if args['device'] is not None:
        device = args['device']
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if mode == 'train':
        train(args, device)
    elif mode == 'test':
        test(args, device)
    elif mode == 'quantize':
        quantize(args, device)
    elif mode == 'test_multiple':
        test_multiple(args, device)
    else:
        train(args, device)
        test(args, device)


if __name__ == '__main__':
    try:
        mode, args = parse_args()
    except ValueError:
        print('Invalid mode. Usage: python run.py <mode[train|test|train+test]> <config. file>')
    except IndexError:
        print('Path to configuration file missing. Usage: python run.py <mode[train|test|train+test]> <config. file>')
        sys.exit()
    except FileNotFoundError:
        print('Incorrect path to configuration file. File not found.')
        sys.exit()
    except json.JSONDecodeError:
        print('Configuration file is an invalid JSON.')
        sys.exit()
    main(mode, args)
