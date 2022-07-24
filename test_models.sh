#!/bin/bash

for model_num in {1..10}
do
   printf 'Testing all checkpoints for model '$model_num
   python run.py test_multiple config/cifar100/wideresnet32/m2.json $model_num
done

