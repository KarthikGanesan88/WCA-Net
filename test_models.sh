#!/bin/bash

for model_num in 2 4
do
   printf 'Testing all checkpoints for model '$model_num
   python run.py test_multiple config/cifar10/preactresnet18/m2_32.json $model_num
done

