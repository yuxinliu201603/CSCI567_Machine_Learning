import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################

			input = np.array(branches)
			weight = np.sum(input, axis = 0)
			temp = input / weight
			a,b = temp.shape
			for m in range(a):
				for n in range(b):
					if temp[m,n] != 0:
						temp[m,n] = temp[m,n]*(np.log2(temp[m,n]))
					else:
						temp[m, n] = 0
			temp = np.sum(temp , axis = 0)
			re = np.sum(-(temp* weight/np.sum(weight)), axis = -1)
			return re


		min_entropy = float('+inf')
		entro_list = []
		for idx_dim in range(len(self.features[0])):


		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			total_classes = np.array(self.features)[:,idx_dim]
			uniq_classes = np.unique(total_classes)
			branches = []
			for i in uniq_classes:
				idx_i = (total_classes == i)
				labels_i = np.array(self.labels)[idx_i]
				branch_list = []


				for label in np.unique(self.labels):
					count = np.count_nonzero(labels_i == label)
					branch_list.append(count)
				branches.append(branch_list)
				# print(branch_list,'bran_list')

			branches = np.transpose(np.array(branches))
			re = conditional_entropy(branches)
			entro_list.append(re)
			# print(re)
			# if re < min_entropy:
			# 	min_entropy = re
			# 	self.dim_split = idx_dim
			# 	self.feature_uniq_split = list(uniq_classes)
		if len(entro_list) == 0:
			self.dim_split = 0
			return
		min_idx = np.array(entro_list).argmin()
		self.dim_split = min_idx
		# self.feature_uniq_split = list(uniq_classes)





		############################################################
		# TODO: split the node, add child nodes
		############################################################
		self.feature_uniq_split = np.unique(np.array(self.features)).tolist()

		# features = np.array(self.features)[:,self.dim_split]
		# for i, v in enumerate(self.feature_uniq_split):
		# 	idx = np.where(features == v)[0]
		# 	child_f = np.array(self.features)[idx]
		# 	child_l = list(np.array(self.labels)[idx])
		# 	child_f = child_f[:,:self.dim_split] + child_f[:,self.dim_split+1:]
		# 	child = TreeNode(list(child_f),child_l,self.num_cls)
		# 	if len(child_f) == 0:
		# 		child.splittable = False
		# 	self.children.append(child)


		for branch in self.feature_uniq_split:
			child_features = []
			child_labels = []

			for i, j in enumerate(self.features):
				if j[self.dim_split] == branch:
					child_features.append(j)
					child_labels.append(self.labels[i])
			child_features = np.delete(child_features, self.dim_split, -1).tolist()
			child = TreeNode(child_features, child_labels, self.num_cls)
			if len(child_features)== 0 or len(child_features[0])== 0:
				child.splittable = False
			self.children.append(child)

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			# print(feature)
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split+1:]
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



