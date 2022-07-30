#!/bin/bash

#for i in {1..10}
#do
#  python run.py test_multiple config/cifar10/mobilenetv2/m2.json $i
#done

for dataset in cifar10 cifar100
do
  for model in preactresnet18 wideresnet32 vgg16 mobilenetv2
  do
    for config in m0 m2
    do
      echo $dataset/$model/$config
      python run.py test config/$dataset/$model/$config.json -1
    done
  done
done


