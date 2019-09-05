import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
	# Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T

		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################

		T = len(self.betas)
		N = len(features)
		re = np.zeros(N,)
		for i in range(T):
			htx = np.array(self.clfs_picked[i].predict(features))
			temp_re = self.betas[i] * htx
			re += temp_re
		re_f = np.sign(re).astype(int).tolist()

		return re_f

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		#initial
		N = len(features)
		D = np.full((N), 1/N)
		y = np.array(labels)

		for i in range(self.T):

			#step1:
			re_list = []
			for clf in self.clfs:
				pred = np.array(clf.predict(features)) * y
				ix = (pred < 0).astype(int)
				re_list.append(np.sum(D * ix))
			ht_idx = re_list.index(min(re_list))
			ht = list(self.clfs)[ht_idx]
			self.clfs_picked.append(ht)

			#step2:
			et = min(re_list)

			#step3:
			beta_t = 1/2 * (np.log((1-et)/et))
			self.betas.append(beta_t)

			#step4:

			ht_array = ht.predict(features) * y
			for n in range(len(D)):
				if ht_array[n] > 0:
					D[n] = D[n] * np.exp(-beta_t)
				else:
					D[n] = D[n] * np.exp(beta_t)

			#step5:
			D = D/np.sum(D)







	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



