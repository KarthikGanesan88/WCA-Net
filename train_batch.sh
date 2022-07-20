#!/bin/bash

python run.py train config/cifar10/preactresnet18/m2.json
python run.py train config/cifar10/wideresnet32/m2.json - done
python run.py train config/cifar10/wideresnet32/m0.json
python run.py train config/cifar10/preactresnet18/m0.json


