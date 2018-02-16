
import lucene
import sys,os
import math
import pickle
import numpy
from nltk.corpus import stopwords
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn import metrics
from java.nio.file import Paths
from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.util import BytesRef, BytesRefIterator
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.index import \
	IndexWriterConfig, IndexWriter, DirectoryReader, IndexOptions, Term


# Fucntion to enumerate all the labels and terms in the document corpus

def getLabelsandTerms(ireader, numDocs):
	labels = []
	terms_list = []
	for doc in xrange(0,numDocs):
		tv = ireader.getTermVector(doc,"contents")
		document = ireader.document(doc)
		topic = document.getField("topic")
		labels.append(topic.stringValue())
		termsEnum = tv.iterator()
		for term in BytesRefIterator.cast_(termsEnum):
			curr_term = term.utf8ToString()
			terms_list.append(curr_term)
	return labels,terms_list

def getTFIDFMatrix(ireader,num_docs,te):
	feature_mat = np.zeros([num_docs, len(te.classes_)])
	for doc in xrange(0, num_docs):
		print "Running For Document number:" + str(doc)
		tv = ireader.getTermVector(doc, "contents")
		termsEnum = tv.iterator()
		for term in BytesRefIterator.cast_(termsEnum):
			str_term = term.utf8ToString()
			term_inst = Term("contents",str_term)
			dpEnum = termsEnum.postings(None)
			dpEnum.nextDoc()        
			freq = dpEnum.freq()
			df = ireader.docFreq(term_inst)
			idf = math.log(num_docs/df)
			tfidf = freq * idf
			
			term_ind = te.transform([str_term])[0]
			feature_mat[doc][term_ind] = tfidf
	return feature_mat
	
def getTermFrequencyMatrix(ireader,num_docs,te):
	feature_mat = np.zeros([num_docs, len(te.classes_)])
	for doc in xrange(0, num_docs):
		print "Running For Document number:" + str(doc)
		tv = ireader.getTermVector(doc, "contents")
		termsEnum = tv.iterator()
		for term in BytesRefIterator.cast_(termsEnum):
			str_term = term.utf8ToString()
			dpEnum = termsEnum.postings(None)
			dpEnum.nextDoc()        
			freq = dpEnum.freq()
			
			term_ind = te.transform([str_term])[0]
			feature_mat[doc][term_ind] = freq
	return feature_mat

def getTermDocMatrix(ireader,num_docs,te):
	feature_mat = np.zeros([num_docs, len(te.classes_)])
	for doc in xrange(0, num_docs):
		print "Running For Document number:" + str(doc)
		tv = ireader.getTermVector(doc, "contents")
		termsEnum = tv.iterator()
		for term in BytesRefIterator.cast_(termsEnum):
			str_term = term.utf8ToString()
			dpEnum = termsEnum.postings(None)
			dpEnum.nextDoc()        
			freq = dpEnum.freq()
			
			term_ind = te.transform([str_term])[0]
			if freq!=0:
				feature_mat[doc][term_ind] = 1
	return feature_mat
	
if __name__ == '__main__':
	lucene.initVM(vmargs=['-Djava.awt.headless=true'])
	print "Features for clustering"
	print "TF , TFIDF , BOOL"
	opt = raw_input("Enter the feature you want to cluster with: ")
	# opt = 'TF'
	print
	print "**** YOU NEED TO TRAIN ONCE BEFORE TESTING ****"
	print "Modes Available"
	print "test , train "
	mode = raw_input("Enter the mode: ")
	# mode = 'train'
	INDEX_DIR = "IndexFiles.index"
	base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	directory = SimpleFSDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
	ireader = DirectoryReader.open(directory)
	num_docs = ireader.numDocs()
	labels, terms_list = getLabelsandTerms(ireader, num_docs)

	if (mode == 'train'):
		le = preprocessing.LabelEncoder()
		true_labels = le.fit_transform(labels)
		te = preprocessing.LabelEncoder()
		te.fit(terms_list)


		if(opt == 'TFIDF'):
			feature_matrix = getTFIDFMatrix(ireader,num_docs,te)
			numpy.save('tfidf_features.npy', feature_matrix)
			numpy.save('labels_classes_tfidf.npy', le.classes_)
			numpy.save('terms_classes_tfidf.npy', te.classes_)
			filename = 'kclustering_tfidf.pkl'
		elif(opt == 'TF'):
			feature_matrix = getTermFrequencyMatrix(ireader, num_docs, te)
			numpy.save('tf.npy', feature_matrix)
			numpy.save('labels_classes_tf.npy', le.classes_)
			numpy.save('terms_classes_tf.npy', te.classes_)
			filename = 'kclustering_tf.pkl'
		elif(opt == 'BOOL'):
			feature_matrix = getTermDocMatrix(ireader, num_docs, te)
			numpy.save('bool.npy', feature_matrix)
			numpy.save('labels_classes_bool.npy', le.classes_)
			numpy.save('terms_classes_bool.npy', te.classes_)
			filename = 'kclustering_bool.pkl'
		else:
			sys.exit('Invalid option')

		km = KMeans(n_clusters=52, init='k-means++', max_iter=100, n_init=1,verbose=1)
		km.fit(feature_matrix)

		kpkl = open(filename, 'wb')
		pickle.dump(km, kpkl)
		kpkl.close()
	
		print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, km.labels_))
		print("Completeness: %0.3f" % metrics.completeness_score(true_labels, km.labels_))
		print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, km.labels_))
		print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(true_labels, km.labels_))

	elif (mode == 'test'):
		le = preprocessing.LabelEncoder()
		#te = preprocessing.LabelEncoder()

		if(opt == 'TFIDF'):
			feature_matrix = numpy.load('tfidf_features.npy')
			le.classes_ = numpy.load('labels_classes_tfidf.npy')
			#te.classes_ = numpy.load('terms_classes_tfidf.npy')
			true_labels = le.transform(labels)
			filename = 'kclustering_tfidf.pkl'
		elif(opt == 'TF'):
			feature_matrix = numpy.load('tf.npy')
			le.classes_ = numpy.load('labels_classes_tf.npy')
			#te.classes_ = numpy.load('terms_classes_tf.npy')
			true_labels = le.transform(labels)
			filename = 'kclustering_tf.pkl'
		elif(opt == 'BOOL'):
			feature_matrix = numpy.load('bool.npy')
			le.classes_ = numpy.load('labels_classes_bool.npy')
			#te.classes_ = numpy.load('terms_classes_bool.npy')
			true_labels = le.transform(labels)
			filename = 'kclustering_bool.pkl'
		else:
			sys.exit('Invalid option')

		kpkl = open(filename,'rb')
		km = pickle.load(kpkl)
		km.predict(feature_matrix)
		
		print("Homogeneity: %0.3f" % metrics.homogeneity_score(true_labels, km.labels_))
		print("Completeness: %0.3f" % metrics.completeness_score(true_labels, km.labels_))
		print("V-measure: %0.3f" % metrics.v_measure_score(true_labels, km.labels_))
		print("Adjusted Rand-Index: %.3f"% metrics.adjusted_rand_score(true_labels, km.labels_))
	
	else:
		sys.exit('Invalid mode')

	   
