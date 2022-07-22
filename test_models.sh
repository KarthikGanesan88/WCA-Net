#!/bin/bash

for model_num in {5..19}
do
   printf 'Testing all checkpoints for model '$model_num
   python run.py test_multiple config/cifar10/preactresnet18/m2_32.json $model_num
done

