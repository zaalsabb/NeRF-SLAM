#!/usr/bin/env sh

python3 ./examples/slam_demo.py --dataset_dir=./datasets/Replica/office0 --dataset_name=nerf --buffer=100 --slam --img_stride=2 --fusion='nerf' --multi_gpu --eval
# python3 ./examples/slam_demo.py --dataset_dir=./datasets/nerf-cube-diorama-dataset/room --dataset_name=nerf --buffer=100 --img_stride=1 --fusion='nerf' --gui --eval