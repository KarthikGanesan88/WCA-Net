#!/bin/bash

# cifar10/wideresnet40 cifar100/resnet34 cifar100/wideresnet40

for model in cifar10/resnet18
do
	for config in vanilla noise
	do
		echo $model/$config
		python run.py test config/$model/$config.json
	done
done


