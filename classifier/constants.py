from metrics_reduced import SpanishTools
from pattern.vector import Classifier
import os


this_path = os.getcwd()
print this_path

spanish_tools = SpanishTools()

# Part-of-Speech tags
nouns        = ['NN', 'NNS', 'NNP', 'NNPS']
verbs        = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
adjectives   = ['JJ','JJR','JJS']
determiners  = ['DT']
conjunctions = ['IN', 'CC']
adverbs      = ['RB','RBR', 'RBS']
modals       = ['MD']
utterances   = ['UH']

# List of sentiment words
positive_adjectives = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/positive_adjectives.txt')
negative_adjectives = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/negative_adjectives.txt')
positive_adverbs    = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/positive_adverbs.txt')
negative_adverbs    = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/negative_adverbs.txt')
positive_verbs      = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/positive_verbs.txt')
negative_verbs      = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/negative_verbs.txt')
positive_nouns      = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/positive_nouns.txt')
negative_nouns      = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/negative_nouns.txt')
positive_others     = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/positive_others.txt')
negative_others     = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/negative_others.txt')
adversative_conj    = spanish_tools.read_file_to_list(this_path + '/vocabularies/words/adversative_conjunctions.txt')

# Comparison is made without accents to increase precision
positive_words = positive_adjectives + positive_adverbs + positive_verbs + positive_nouns + positive_others
for i, s in enumerate(positive_words):
    positive_words[i] = spanish_tools.remove_accents(s)
negative_words = negative_adjectives + negative_adverbs + negative_verbs + negative_nouns + negative_others
for i, s in enumerate(negative_words):
    negative_words[i] = spanish_tools.remove_accents(s)


# Feature list vector for Morphosyntactic Model and Classifier
feature_list = spanish_tools.read_file_to_list(this_path + '/vocabularies/features/morphosyntactic_feature_list.txt')

# Feature list vector for Bigram Model and Classifier
bigram_feature_list = spanish_tools.read_file_to_list(this_path + '/vocabularies/features/bigram_feature_list.txt')

# SVM Classifiers
classifier_svm = Classifier.load(this_path + '/objects/classifiers/SVM/Morphosyntactic_Classifier_5_Classes_backup')

# MNB Classifiers
classifier_mnb = Classifier.load(this_path + '/objects/classifiers/MNB/Bigram_2500_Classifier_5_Classes')
