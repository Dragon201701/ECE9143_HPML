from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class args(object):
	def __init__(self):
		self.parser = argparse.ArgumentParser(description='ECE9143 template')
		self.parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
		self.parser.add_argument('--optim', default='sgd')
		self.parser.add_argument('--num_workers', default=2, type=int, help='number of workers of train and test loader')
		self.parser.add_argument('--data_path', default='./data')
		# self.parser.add_argument('--cpu_only', action='store_true', help='Use GPU by default without this option.')
		self.parser.add_argument('--batch_size', default=32, type=int, help='Batch size.')

	def parse(self):
		return self.parser.parse_args()