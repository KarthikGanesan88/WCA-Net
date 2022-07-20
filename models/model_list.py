from .preactresnet import VanillaPreActResNet18, WCANet_PreActResNet18
from .resnet import VanillaResNet18, WCANet_ResNet18
from .lenet import VanillaLeNet, WCANet_LeNet
from .lenetplus import VanillaLeNetPlus, WCANet_LeNetPlus
from .wideresnet import VanillaWideResNet32, WCANet_WideResNet32
from .alexnet import VanillaAlexNet, WCANet_AlexNet

def model_factory(model, dataset, training_type, variance_type, feature_dim, num_classes):
    model_list = {
        'lenet': {
            'vanilla': VanillaLeNet,
            'stochastic': WCANet_LeNet
        },
        'lenetplus': {
            'vanilla': VanillaLeNetPlus,
            'stochastic': WCANet_LeNetPlus
        },
        'alexnet': {
            'vanilla': VanillaAlexNet,
            'stochastic': WCANet_AlexNet
        },
        'resnet18': {
            'vanilla': VanillaResNet18,
            'stochastic': WCANet_ResNet18
        },
        'preactresnet18': {
            'vanilla': VanillaPreActResNet18,
            'stochastic': WCANet_PreActResNet18
        },
        'wideresnet32': {
            'vanilla': VanillaWideResNet32,
            'stochastic': WCANet_WideResNet32
        },
    }
    if training_type == 'vanilla':
        return model_list[model][training_type](feature_dim, num_classes)
    elif training_type == 'stochastic':
        return model_list[model][training_type](feature_dim, num_classes, variance_type)

"""
def model_factory_old(model, dataset, training_type, variance_type, feature_dim, num_classes):
    if variance_type is not None and variance_type not in ('isotropic', 'anisotropic'):
        raise NotImplementedError('Only "isotropic" and "anisotropic" variance types supported.')
    if dataset == 'mnist':
        if training_type == 'vanilla':
            model = VanillaCNN(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_CNN(feature_dim, num_classes, variance_type)
    elif dataset == 'fmnist':
        if training_type == 'vanilla':
            model = VanillaCNN(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_CNN(feature_dim, num_classes, variance_type)
    elif dataset == 'cifar10':
        if training_type == 'vanilla':
            if model == 'resnet18':
                model = VanillaResNet18(feature_dim, num_classes)
            elif model == 'preactresnet18':
                model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            if model == 'resnet18':
                model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
            elif model == 'preactresnet18':
                model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
    elif dataset == 'cifar100':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
    elif dataset == 'svhn':
        if training_type == 'vanilla':
            model = VanillaResNet18(feature_dim, num_classes)
        elif training_type in ('stochastic', 'stochastic+adversarial'):
            model = WCANet_ResNet18(feature_dim, num_classes, variance_type)
    else:
        raise NotImplementedError('Model for dataset {} not implemented.'.format(dataset))
    return model
"""
