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
    max_steps = sys.argv[3] 

    for sp in np.arange(5, 10)/10:  # sparsity
    	#os.system
    	print("python3 prune_resnet_dns.py %s \
            --dataset cifar10_%s.tfrecord \
            --train-dir out-prune-%d-%d \
            --warm-start out-int(prop_rand*100) \
            --target-sparsity %s \
            --proportion-random %s \
            --max-steps %s" % (
            	trn_or_ev, trn_or_ts,
            	int(prop_rand*100), int(sp*10),
            	str(sp), str(prop_rand), max_steps))



