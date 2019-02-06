#1/bin/bash

for var in "$@"
do
    CUDA_VISIBLE_DEVICES=2 python trainmodel.py --hidden-size 200 --log-interval 10 --decay 0 --epochs 1000 --batch-size 10 --lr "$var"
done

