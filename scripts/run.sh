#!/usr/bin/env sh

# python3 ./examples/slam_demo.py --dataset_dir=./datasets/Replica/office0 --dataset_name=nerf --buffer=100 --slam --parallel_run --img_stride=2 --fusion='nerf' --multi_gpu --gui
python3 ./examples/slam_demo.py --dataset_dir=./datasets/nerf-cube-diorama-dataset/room --dataset_name=nerf --buffer=100 --img_stride=1 --fusion='nerf' --gui --eval