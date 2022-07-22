import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm.auto import tqdm

from attacks.fgsm import fgsm
from attacks.pgd import pgd
from metrics import accuracy
from utils import normalize_cifar10, normalize_cifar100, normalize_generic, attack_to_dataset_config, print_log
from test import test_attack


def get_stochastic_model_optimizer(model, args):
    if args['var_type'] == 'isotropic':
        trainable_noise_params = {'params': model.base.sigma, 'lr': args['lr'], 'weight_decay': args['wd']}
    elif args['var_type'] == 'anisotropic':
        trainable_noise_params = {'params': model.base.L, 'lr': args['lr'], 'weight_decay': args['wd']}
    optimizer = Adam([
        {'params': model.base.gen.parameters(), 'lr': args['lr']},
        {'params': model.base.fc1.parameters(), 'lr': args['lr']},
        trainable_noise_params,
        {'params': model.proto.parameters(), 'lr': args['lr'], 'weight_decay': args['wd']}
    ])
    return optimizer


def get_norm_func(args):
    if args['dataset'] == 'cifar10':
        norm_func = normalize_cifar10
    elif args['dataset'] == 'cifar100':
        norm_func = normalize_cifar100
    elif args['dataset'] == 'svhn':
        norm_func = normalize_generic
    elif args['dataset'] in ('mnist', 'fmnist'):
        norm_func = None
    return norm_func


def train_vanilla(model, train_loader, test_loader, args, model_path, logfile, device='cpu'):
    optimizer = Adam(model.parameters(), lr=args['lr'], weight_decay=args['wd'])
    loss_func = nn.CrossEntropyLoss()
    norm_func = get_norm_func(args)
    best_test_acc = -1.

    for epoch in range(args['num_epochs']):
        for data, target in tqdm(train_loader, leave=False):
            data = data.to(device)
            target = target.to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
            logits = model(data)
            optimizer.zero_grad()
            loss = loss_func(logits, target)
            loss.backward()
            optimizer.step()
        train_acc = accuracy(model, train_loader, device=device, norm=norm_func)
        test_acc = accuracy(model, test_loader, device=device, norm=norm_func)
        print_log(logfile, 'Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(model_path, 'ckpt_best.pt'))
            print_log(logfile, 'Best test accuracy achieved on epoch {}.'.format(epoch + 1))
        model.save(os.path.join(model_path, 'ckpt_last'))
        if epoch % 25 == 0 and epoch > 10:
            # Every 50 epochs, run the attacks to see how this changes.
            # attack_names = ['FGSM']  # 'BIM', 'C&W', 'Few-Pixel'
            # print_log(logfile, 'Adversarial testing.')
            # for idx, attack in enumerate(attack_names):
            #     eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
            #     eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
            #     robust_accuracy = test_attack(model, test_loader, attack, eps_values, args, device='cpu')
            #     for eps_name, eps_value, acc in zip(eps_names, eps_values, robust_accuracy):
            #         print_log(logfile, 'Attack Strength: {}, Accuracy: {:.3f}%'.format(eps_name, 100.*acc.item()))
            # Also save the model separately each 50 epochs in case it performs better earlier.
            torch.save(model.state_dict(), os.path.join(model_path, f'ckpt_{epoch}.pt'))


def train_stochastic(model, train_loader, test_loader, args, model_path, logfile, device='cpu'):
    optimizer = get_stochastic_model_optimizer(model, args)
    scheduler = StepLR(optimizer, int(args['num_epochs'] / 3), 0.1)
    # Uncomment for the "train model and noise separately" ablation. But first train a model with disable_noise=True.
    # model.freeze_model_params()
    loss_func = nn.CrossEntropyLoss()
    norm_func = get_norm_func(args)
    best_test_acc = -1.

    for epoch in range(args['num_epochs']):
        for data, target in tqdm(train_loader, leave=False):
            data = data.to(device)
            target = target.to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
            logits = model(data)
            optimizer.zero_grad()
            if args['var_type'] == 'isotropic':
                wca = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
            elif args['var_type'] == 'anisotropic':
                wca = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
            loss = loss_func(logits, target) - torch.log(wca)
            loss.backward()
            optimizer.step()
            if args['var_type'] == 'anisotropic':
                with torch.no_grad():
                    model.base.L.data = model.base.L.data.tril()
        scheduler.step()
        train_acc = accuracy(model, train_loader, device=device, norm=norm_func)
        test_acc = accuracy(model, test_loader, device=device, norm=norm_func)
        print_log(logfile, 'Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(os.path.join(model_path, 'ckpt_best'))
            print_log(logfile, 'Best test accuracy achieved on epoch {}.'.format(epoch + 1))
            if epoch >= 50:
                if best_test_acc < 0.5:
                    print_log(logfile, 'After 50 epochs, model has failed to reach 50% test accuracy. Aborting '
                                       'training.')

        model.save(os.path.join(model_path, 'ckpt_last'))

        if epoch % 25 == 0 and epoch > 10:
            # Every 50 epochs, run the attacks to see how this changes.
            # attack_names = ['FGSM']  # 'BIM', 'C&W', 'Few-Pixel'
            # print_log(logfile, 'Adversarial testing.')
            # for idx, attack in enumerate(attack_names):
            #     eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
            #     eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
            #     robust_accuracy = test_attack(model, test_loader, attack, eps_values, args, device='cpu')
            #     for eps_name, eps_value, acc in zip(eps_names, eps_values, robust_accuracy):
            #         print_log(logfile, 'Attack Strength: {}, Accuracy: {:.3f}%'.format(eps_name, 100.*acc.item()))
            # Also save the model separately each 50 epochs in case it performs better earlier.
            torch.save(model.state_dict(), os.path.join(model_path, f'ckpt_{epoch}.pt'))


def train_stochastic_adversarial(model, train_loader, test_loader, args, model_path, logfile, device='cpu'):
    optimizer = get_stochastic_model_optimizer(model, args)
    scheduler = StepLR(optimizer, int(args['num_epochs'] / 3), 0.1)
    # Uncomment for the "train model and noise separately" ablation. But first train a model with disable_noise=True.
    # model.freeze_model_params()
    loss_func = nn.CrossEntropyLoss()
    norm_func = get_norm_func(args)
    if args['attack'] == 'fgsm':
        attack_func = fgsm
    elif args['attack'] == 'pgd':
        attack_func = pgd
    best_test_acc = -1.

    for epoch in range(args['num_epochs']):
        for data, target in tqdm(train_loader, leave=False):
            data = data.to(device)
            target = target.to(device)
            perturbed_data = attack_func(model, data, target, epsilon=args['epsilon']).to(device)
            model.train()
            if norm_func is not None:
                data = norm_func(data)
                perturbed_data = norm_func(perturbed_data)
            logits = model(data)
            adv_logits = model(perturbed_data)
            optimizer.zero_grad()
            clean_loss = loss_func(logits, target)
            adv_loss = loss_func(adv_logits, target)
            if args['var_type'] == 'isotropic':
                wca = (model.proto.weight @ model.sigma.diag() @ model.proto.weight.T).diagonal().sum()
            elif args['var_type'] == 'anisotropic':
                wca = (model.proto.weight @ model.sigma @ model.proto.weight.T).diagonal().sum()
            loss = args['w_ct'] * clean_loss + args['w_at'] * adv_loss - torch.log(wca)
            loss.backward()
            optimizer.step()
            if args['var_type'] == 'anisotropic':
                with torch.no_grad():
                    model.base.L.data = model.base.L.data.tril()
        scheduler.step()
        train_acc = accuracy(model, train_loader, device=device, norm=norm_func)
        test_acc = accuracy(model, test_loader, device=device, norm=norm_func)
        print_log(logfile, 'Epoch {:03}, Train acc: {:.3f}, Test acc: {:.3f}'.format(epoch + 1, train_acc, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            model.save(os.path.join(model_path, 'ckpt_best'))
            print_log(logfile, 'Best test accuracy achieved on epoch {}.'.format(epoch + 1))
        model.save(os.path.join(model_path, 'ckpt_last'))
        if epoch % 25 == 0 and epoch > 10:
            # Every 50 epochs, run the attacks to see how this changes.
            # attack_names = ['FGSM']  # 'BIM', 'C&W', 'Few-Pixel'
            # print_log(logfile, 'Adversarial testing.')
            # for idx, attack in enumerate(attack_names):
            #     eps_names = attack_to_dataset_config[attack][args['dataset']]['eps_names']
            #     eps_values = attack_to_dataset_config[attack][args['dataset']]['eps_values']
            #     robust_accuracy = test_attack(model, test_loader, attack, eps_values, args, device='cpu')
            #     for eps_name, eps_value, acc in zip(eps_names, eps_values, robust_accuracy):
            #         print_log(logfile, 'Attack Strength: {}, Accuracy: {:.3f}%'.format(eps_name, 100.*acc.item()))
            # Also save the model separately each 50 epochs in case it performs better earlier.
            torch.save(model.state_dict(), os.path.join(model_path, f'ckpt_{epoch}.pt'))
