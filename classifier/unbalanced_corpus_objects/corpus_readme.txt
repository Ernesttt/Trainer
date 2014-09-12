CORPUS DISTRIBUTION
===================
{u'muy_positivo': 881, u'positivo': 954, u'muy_negativo': 904, u'neutro': 1335, u'negativo': 1000}


CORPUS FILE STRUCTURE
====================
* txt file
* comment by line (separator '\n')
* line example:
		source/|/comment text.../|/category_tag


PATTERN-CLIPS CONFIGURATION FOR MODEL
=====================================
Morphosyntactic Model:
150 features described in morphosyntactic_feature_list.txt
See Content_Analysis.doc and Appendix for clarification

Bigram Model:
2500 features described in bigram_features.txt


PATTERN-CLIPS CONFIGURATION FOR CLASSIFIER
==========================================
SVM Classifier:
type = CLASSIFICATION, kernel = POLYNOMIAL, train=model, degree=6 

NB Classifier:
train=model