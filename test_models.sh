#!/bin/bash

for dataset in cifar10 cifar100
do
  for model in preactresnet18 wideresnet32
  do
    for num in m0 m2
    do
      printf "Running model: config/%s/%s/%s.json -1\n" ${dataset} ${model} ${num}
      python run.py test config/${dataset}/${model}/${num}.json -1
    done
  done
done

#python run.py test config/cifar10/preactresnet18/m0.json -1
#python run.py test config/cifar100/preactresnet18/m0.json -1
#python run.py test config/cifar10/wideresnet32/m0.json -1
#python run.py test config/cifar100/wideresnet32/m0.json -1

