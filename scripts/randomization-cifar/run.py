#! /usr/bin/env python3.6

import tensorflow as tf
import sys
sys.path.append("../../")
sys.path.append("../../nnet")
sys.path.append("../../scripts")
import os


if __name__ == "__main__":
    # sys.argv[1] proportion-random * 100
    for sparsity in np.arange(5, 10)/10:
        os.system(f"python3 prune_resnet_dns.py train \
            --dataset cifar10_train.tfrecord \
            --train-dir out-prune-{sys.argv[1]} \
            --warm_start out-{sys.argv[1]}-{int(sparsity*10)}/ \
            --target-sparsity {sparsity} \
            --proportion-random {proportion-random/100}")



