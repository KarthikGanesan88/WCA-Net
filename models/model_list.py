from .preactresnet import VanillaPreActResNet18, WCANet_PreActResNet18
from .resnet import VanillaResNet18, WCANet_ResNet18
from .lenet import VanillaLeNet, WCANet_LeNet
from .lenetplus import VanillaLeNetPlus, WCANet_LeNetPlus
from .wideresnet import VanillaWideResNet32, WCANet_WideResNet32
from .alexnet import VanillaAlexNet, WCANet_AlexNet
from .vgg16 import VanillaVGG16, WCANet_VGG16
from .mobilenetv2 import VanillaMobileNetV2, WCANet_MobileNetV2

def model_factory(model, dataset, training_type, variance_type, feature_dim, num_classes):
    model_list = {
        'lenet': {
            'vanilla': VanillaLeNet,
            'anisotropic': WCANet_LeNet,
            'adversarial': WCANet_LeNet
        },
        'lenetplus': {
            'vanilla': VanillaLeNetPlus,
            'anisotropic': WCANet_LeNetPlus,
            'adversarial': WCANet_LeNetPlus
        },
        'alexnet': {
            'vanilla': VanillaAlexNet,
            'anisotropic': WCANet_AlexNet,
            'adversarial': WCANet_AlexNet
        },
        'vgg16': {
            'vanilla': VanillaVGG16,
            'anisotropic': WCANet_VGG16,
            'adversarial': WCANet_VGG16
        },
        'mobilenetv2': {
            'vanilla': VanillaMobileNetV2,
            'anisotropic': WCANet_MobileNetV2,
            'adversarial': WCANet_MobileNetV2
        },
        'resnet18': {
            'vanilla': VanillaResNet18,
            'anisotropic': WCANet_ResNet18,
            'adversarial': WCANet_ResNet18
        },
        'preactresnet18': {
            'vanilla': VanillaPreActResNet18,
            'anisotropic': WCANet_PreActResNet18,
            'adversarial': WCANet_PreActResNet18
        },
        'wideresnet32': {
            'vanilla': VanillaWideResNet32,
            'anisotropic': WCANet_WideResNet32,
            'adversarial': WCANet_WideResNet32
        },
    }
    # print('training type (in model_factory)', training_type)
    if training_type == 'vanilla':
        return model_list[model][training_type](feature_dim, num_classes)
    elif training_type in ['anisotropic', 'adversarial']:
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
