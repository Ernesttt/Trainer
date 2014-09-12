from __future__ import division
from preprocessing import Preprocessing
from vectorization import VectorQuantization
from pattern.vector import Document as pattern_Document
from pattern.vector import Model as pattern_Model 
from pattern.vector import CLASSIFICATION
from pattern.vector import SVM, LINEAR, POLYNOMIAL, RADIAL
from pattern.vector import NB
import codecs


class Model():
	"""
	Creates Bag of Words Model
	"""
	def create(self, corpus_path, model_type="morphosyntactic"):
		preprocessor = Preprocessing()
		vectorizer = VectorQuantization()
		document_list = []
		with codecs.open(corpus_path, 'r', 'utf-8') as corpus:
			line = corpus.readline()
			while line:
				comment = preprocessor.preprocessing(line.split('/|/')[1])
				category = line.split('/|/')[2].split('\n')[0]
				if model_type == "morphosyntactic":
					comment_vector = vectorizer.morphosyntactic_vector(comment)
				elif model_type == "bigram":
					comment_vector = vectorizer.bigram_vector(comment)
				else:
					print "No model defined using default: morphosyntactic"
					comment_vector = vectorizer.morphosyntactic_vector(comment)
				if comment_vector:
					document_list.append(pattern_Document(comment_vector, 
														  type=category))
				line = corpus.readline()
		model = pattern_Model(documents=document_list, weight=None)
		return model

class Classifier():
	"""
	Creates classifier taking a Model and defining a Classifier type
	"""
	def create_svm(self, model, type="CLASSIFICATION", kernel_type="LINEAR", degree=3,
				   gamma=0.1, coeff0=0, cost=1, epsilon=0.1):
		"""		

		Parameter	Value						Description
		type		CLASSIFICATION, REGRESSION	REGRESSION returns a float value.
		kernel		LINEAR(1), POLYNOMIAL(2), 	Kernel function used for separation.
					RADIAL(3)					
		degree		3							Used in POLYNOMIAL kernel.
		gamma		1 / len(SVM.features)		Used in POLYNOMIAL and RADIAL kernel.
		coeff0		0							Used in POLYNOMIAL kernel.
		cost		1							Soft margin for training errors.
		epsilon		0.1							Tolerance for termination criterion.
		cache		100							Cache memory size in MB.
		probability	False						CLASSIFICATION yields (weight, class) values

		"""
		# KERNEL TYPE
		if kernel_type == "RADIAL":
			kernel = RADIAL
		elif kernel_type == "POLYNOMIAL":
			kernel = POLYNOMIAL
		else:
			kernel = LINEAR
		# CLASSIFICATION OR REGRESSION
		if type == "CLASSIFICATION":
			type = CLASSIFICATION
		elif type == "REGRESSION":
			type = REGRESSION
		# CLASSIFIER
		if model:
			classifier = SVM(type=type, kernel=kernel, train=model, degree=degree, gamma=gamma,
							 coeff0=coeff0, cost=cost, epsilon=epsilon)
			return classifier
		else:
			print "No model defined"

		

	def create_mnb(self, model):
		if model:
			classifier = NB(train=model)
			return classifier
		else:
			print "No model defined"

		
