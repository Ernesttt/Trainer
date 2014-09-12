import metrics_reduced
from preprocessing import Preprocessing
import codecs
from collections import Counter
from constants import *

s = metrics_reduced.SpanishTools()
p = Preprocessing()

with codecs.open('corpus/Version_2_classes/corpus_2_classes.txt', 'r', 'utf-8') as file:
	features = {}
	line = file.readline()
	while line:
		line_bigram_list = s.n_grams(p.preprocessing(line.split('/|/')[1]))
		for e in line_bigram_list:
			if e in features:
				features[e] += 1
			else:
				features[e] = 1
		line = file.readline()


with codecs.open('bigram_features_filtered.txt', 'w', 'utf-8') as file:
	counter = Counter(features)
	common = counter.most_common(10000)
	temp_list=[]
	for k in common:
		for e in k[0].split():
			if e in positive_words or e in negative_words:
				temp_list.append(k[0])

	temp_set = set(temp_list)
	for e in temp_set:
		file.write(e)
		file.write('\n')


