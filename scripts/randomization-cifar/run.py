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
    prop_rand = float(sys.argv[2])
    max_steps = sys.argv[3] 

	if trn_or_ev == "train":
		for sp in np.arange(5, 10)/10:  # sparsity
			os.system("python3 prune_resnet_dns.py train \
				--dataset cifar10_train.tfrecord \
				--train-dir out-prune-%d-%d \
				--warm-start out-%d \
				--target-sparsity %s \
				--proportion-random %s \
				--max-steps %s" % (
					int(prop_rand*100), int(sp*10), int(prop_rand*100),
					str(sp), str(prop_rand), max_steps))
	elif trn_or_ev == "eval":
		for sp in np.arange(5, 10)/10:  # sparsity
			os.system("python3 prune_resnet_dns.py eval \
				--dataset cifar10_test.tfrecord \
				--train-dir out-prune-%d-%d \
				--target-sparsity %s \
				--max-steps %s" % (
					int(prop_rand*100), int(sp*10),
					str(sp), str(prop_rand), max_steps))
	else:
		raise Exception("Please specify either train or eval!")



