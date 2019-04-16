#! /usr/bin/env python3.6

import tensorflow as tf
import sys
sys.path.append("../../")
sys.path.append("../../nnet")
sys.path.append("../../scripts")
import os
import numpy as np


if __name__ == "__main__":
	# sys.argv[1] train or eval: {train, eval}
    # sys.argv[2] proportion-random
    trn_or_ev = sys.argv[1] 
    trn_or_ts = "train" if (trn_or_ev == "train") else "test"
    prop_rand = float(sys.argv[2])

    for sp in np.arange(5, 10)/10:  # sparsity
    	os.system(f"python3 prune_resnet_dns.py {trn_or_ev} \
            --dataset cifar10_{trn_or_ts}.tfrecord \
            --train-dir out-prune-{int(prop_rand*100)}-{int(sp*10)} \
            --warm_start out-{int(prop_rand*100)}/ \
            --target-sparsity {sp} \
            --proportion-random {prop_rand}")



