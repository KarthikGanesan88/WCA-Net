import json
import os
import sys
from datetime import datetime
from torchinfo import summary

import torch
from tqdm.auto import tqdm

from attacks.one_pixel import one_pixel_attack
from data_loaders import get_data_loader
from models.model_list import model_factory
from test import foolbox_attack
from train import train_vanilla, train_stochastic, train_stochastic_adversarial, train_adversarial, get_norm_func
from utils import attack_to_dataset_config, print_log, modify_layers
import metrics
from torchattacks import *

import warnings

try:
    from custom.custom_conv import CustomConv
    from custom.custom_fc import CustomLinear
except ImportError:
    warnings.warn('Custom layers not found.', ImportWarning)


def parse_args():
    mode = sys.argv[1]
    if mode not in ('train', 'test', 'train+test', 'test_DA', 'test_multiple'):
        raise ValueError()
    config_file = sys.argv[2]
    with open(config_file, 'r') as fp:
        args = json.loads(fp.read().strip())

    args['run_id'] = int(sys.argv[3]) if len(sys.argv) > 2 else -1
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
"""


def train(args, device):
    print(args)

    if args['training_type'] == "stochastic":
        train_type = args['var_type']
    elif args['training_type'] == "adversarial":
        train_type = "adversarial"
    else:
        train_type = args['training_type']

    # Add support for training lots of models at once.

    if args['run_id'] != -1:
        model_path = os.path.join(
            f"./output/{args['model']}_{args['dataset']}_{train_type}_runid{args['run_id']}")
    else:
        model_path = os.path.join(
            f"./output/{args['model']}_{args['dataset']}_{train_type}")

    print(f'Training type: {train_type}')

    os.makedirs(model_path, exist_ok=True)
    logfile = os.path.join(model_path, f'{datetime.now().strftime("%Y_%M_%d_%H_%M_%S")}.log')
    log = open(logfile, 'a')
    for key, value in args.items():
        log.write(f'{key}: {value}, ')
    log.write('\n')
    log.flush()
    log.close()

    train_loader, _ = get_data_loader(args['dataset'], args['batch_size'], train=True, shuffle=True, drop_last=True)
    test_loader, subset_loader = get_data_loader(args['dataset'], args['batch_size'], train=False, shuffle=False,
                                                 drop_last=False)
    model = model_factory(args['model'],
                          args['dataset'],
                          train_type,
                          args['var_type'],
                          args['feature_dim'],
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
    elif args['training_type'] == 'adversarial':
        print_log(logfile, 'Adversarial training.')
        train_adversarial(model, train_loader, test_loader, args, model_path, logfile, device=device)
    else:
        raise NotImplementedError(
            'Training "{}" not implemented. Supported: [vanilla|stochastic|stochastic+adversarial].'.format(
                args['training_type']))
    print_log(logfile, 'Finished training.')


def print_log(log, print_string, enable_logging=False):
    if enable_logging:
        log.write('{}\n'.format(print_string))
        log.flush()


def test_multiple(args, device):
    print(args)

    if args['training_type'] == "stochastic":
        train_type = args['var_type']
    elif args['training_type'] == "stochastic+adversarial":
        train_type = "adversarial"
    else:
        train_type = args['training_type']

    # For test multiple, only do FGSM as its quicker.
    attack_names = ['FGSM']

    # if this is to test a bunch of runs, accept the runid here.
    if args['run_id'] != -1:
        model_path = os.path.join(
            f"./output/{args['model']}_{args['dataset']}_{train_type}_runid{args['run_id']}")

        enable_logging = True

        logfile = f"./output/logs/{args['model']}_{args['dataset']}.log"
        log = open(logfile, 'a')
        # log.write(f'Run ID, best epoch, best robust accuracy, classification accuracy at best epoch\n')
        log.flush()

    else:
        model_path = os.path.join(
            f"./output/{args['model']}_{args['dataset']}_{train_type}")
        enable_logging = False
        log = None

    test_loader, _ = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False, drop_last=False)

    model = model_factory(args['model'],
                          args['dataset'],
                          # args['training_type'],
                          train_type,
                          args['var_type'],
                          args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    accuracy_dict = {}
    robust_dict = {}
    # robust_dict = dict.fromkeys(attack_names, {})  # This only supports a single epsilon per attack for now.

    # To test multiple models in one run.
    for epoch in range(25, args['num_epochs'], 25):
        if os.path.exists(os.path.join(model_path, f'ckpt_{epoch}.pt')):
            model.load(os.path.join(model_path, f'ckpt_{epoch}'))
        else:
            # Checkpoints past this epoch number do not exist.
            break

        print(f'\nRunning model saved at epoch: {epoch}')
        model.eval()

        test_acc = metrics.accuracy(model, test_loader, device=device, norm=get_norm_func(args))
        acc = 100. * test_acc
        print(f'Accuracy: {acc:.3f}%')
        accuracy_dict[epoch] = acc
        # print_log(log, f"{args['run_id']}, {model_num}, {acc}", enable_logging=enable_logging)

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
                robust_accuracy = foolbox_attack(model, test_loader, attack, eps_values, args, device)
                for eps_name, eps_value, acc in zip(eps_names, eps_values, robust_accuracy):
                    print('Attack Strength: {}, Accuracy: {:.3f}%'.format(eps_name, 100. * acc.item()))
                    robust_dict[epoch] = acc
        print('Finished testing.')

    # Need to do this outside the main for loop.
    best_epoch = -1
    best_acc = 0
    for epoch, acc in robust_dict.items():
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch

    # log.write(f'Run ID, best epoch, best robust accuracy, classification accuracy at best epoch')
    print_log(log, f"{args['run_id']}, {best_epoch}, {best_acc}, {accuracy_dict[best_epoch]}",
              enable_logging=enable_logging)

    if args['run_id'] != -1:
        log.close()


def test_DA(args, device):
    print(args)

    if args['training_type'] == "stochastic":
        train_type = args['var_type']
    elif args['training_type'] == "stochastic+adversarial":
        train_type = "adversarial"
    else:
        train_type = args['training_type']

    model_path = os.path.join(
        f"./output/{args['model']}_{args['dataset']}_{train_type}")

    model = model_factory(args['model'],
                          args['dataset'],
                          # args['training_type'],
                          train_type,
                          args['var_type'],
                          args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    model.load(os.path.join(model_path, 'ckpt_best'))

    modify_layers(model)

    for name, layer in model.named_modules():
        if isinstance(layer, CustomConv) or isinstance(layer, CustomLinear):
            layer.appx_mode = torch.Tensor([1]).cuda()
            # print(f'{name}: Appx mode:{layer.appx_mode}')

    model.eval()
    test_loader, subset_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False,
                                                 drop_last=False, subset_size=100)

    test_acc = metrics.accuracy(model, subset_loader, device=device, norm=get_norm_func(args))
    print(f'DA accuracy: {100. * test_acc:.3f}%')


def test(args, device):
    print(args)

    if args['training_type'] == "stochastic":
        train_type = args['var_type']
    elif args['training_type'] == "stochastic+adversarial":
        train_type = "adversarial"
    else:
        train_type = args['training_type']

    model_path = os.path.join(
        f"./output/{args['model']}_{args['dataset']}_{train_type}")

    model = model_factory(args['model'],
                          args['dataset'],
                          # args['training_type'],
                          train_type,
                          args['var_type'],
                          args['feature_dim'],
                          args['num_classes'])
    model.to(device)

    model.load(os.path.join(model_path, 'ckpt_best'))
    # model.load(os.path.join(model_path, 'ckpt_last'))
    model.eval()

    # summary(model,
    #         input_size=(args['batch_size'], 3, 32, 32),
    #         # col_names=['input_size', 'output_size', 'kernel_size'],
    #         depth=5
    #         )

    # lc = 0
    # print('Layer name, stride, padding')
    # for name, layer in model.named_modules():
    #     if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.Linear):
    #         # pad = 'TRUE' if layer.padding[0] == 1 else 'FALSE'
    #         # print(name, layer.stride[0], pad)
    #         lc += 1
    # print('Number of layers:', lc)
    # breakpoint()

    test_loader, subset_loader = get_data_loader(args['dataset'], args['batch_size'], False, shuffle=False,
                                                 drop_last=False, subset_size=128)

    test_norm = get_norm_func(args)
    test_acc = metrics.accuracy(model, test_loader, device=device, norm=test_norm)
    print(f'Accuracy: {100. * test_acc:.3f}%')

    # attack_names = ['FGSM', 'PGD', 'C&W', 'Square', 'Pixle']
    attack_names = ['C&W']
    eps_names, eps_values = ['8/255'], [8. / 255]

    bb_attacks = {
        'FGSM': FGSM(model, eps=8./255),
        # 'Square': Square(model, eps=8. / 255, n_queries=2000),
        # 'Pixle': Pixle(model),
        # 'OnePixel': OnePixel(model, pixels=1, popsize=400, steps=75),
        # 'CW': CW(model, kappa=0.1, lr=0.0005, steps=1000)
        # 'C&W': L2CarliniWagnerAttack(steps=1000, stepsize=5e-4,
        # confidence=0., initial_const=1e-3, binary_search_steps=9)
    }

    # print('Adversarial testing.')
    for idx, attack in enumerate(attack_names):
        # print('Attack: {}'.format(attack))
        if attack in bb_attacks.keys():
            correct, total = 0, 0
            inp_min, inp_max = 10.0, -10.0
            for i, (image, target) in enumerate(tqdm(subset_loader, leave=False)):
                image, target = image.cuda(), target.cuda()
                if test_norm is not None:
                    image = test_norm(image)
                inp_min = min(inp_min, torch.min(image).item())
                inp_max = max(inp_max, torch.max(image).item())
                total += len(target)
                adv_images = bb_attacks[attack](image, target)
                output = model(adv_images)
                _, preds = output.max(1)
                correct += preds.eq(target).sum().item()
            print(f'Image min: {inp_min}, Image max: {inp_max}')
            print(f'{attack} Accuracy:{100 * (correct / total)}%')
        else:
            # eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
            # eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
            robust_accuracy = foolbox_attack(model, subset_loader, attack, eps_values, args, device)
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
    elif mode == 'test_DA':
        test_DA(args, device)
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
