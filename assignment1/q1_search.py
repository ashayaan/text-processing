#!/usr/bin/env python

INDEX_DIR = "IndexFiles.index"

import sys, os, lucene

from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.queryparser.classic import MultiFieldQueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.queryparser.complexPhrase import ComplexPhraseQueryParser
from org.apache.lucene.queryparser.ext import ExtendableQueryParser
from org.apache.lucene.search import BooleanClause 
from org.apache.lucene.queries import CustomScoreQuery
from org.apache.lucene.queries import CustomScoreProvider
#from org.apache.lucene.index import AtomicReader


def CoffeeQuery(CustomScoreQuery):
	def __init__(self, subQuery):
		CustomScoreQuery.__init__(self, subQuery)
		#super(CoffeeQuery, self).__init__(subQuery)

	def getCustomScoreProvider(self, atomic_context):
		return CoffeeScoreProvider(atomic_context)
	
def CoffeeScoreProvider(CustomScoreProvider):
	atomicReader = None
	
	def __init__(self, context):
		CustomScoreProvider.__init__(self, context)
		#super(CoffeeScoreProvider, self).__init__(context)
		CoffeeScoreProvider.atomicReader = context.Reader()

	def customScore(doc, subQueryScore, valSrcScore):
		docAtHand = CoffeeScoreProvider.atomicReader.document(doc)
		topic = docAtHand.getValue('topic')
		cont = docAtHand.getValue('contents')
		print topic
		print cont
		if (topic == 'coffee'):
			return 20.0
		else: 
			return 2.0
	
def run(searcher, analyzer):
	while True:
		print
		print "Hit enter with no input to quit."
		command = raw_input("Query:")
		if command == '':
			return

		print
		print "Searching for:", command
		#query = ComplexPhraseQueryParser("contents", analyzer).parse(command)
		query = QueryParser("contents", analyzer).parse(command)
		# customquery = CoffeeQuery(query)
		# print customquery
		#fields = JArray('string')(['topic','contents'])
		#flags = JArray(BooleanClause.Occur)([BooleanClause.Occur.SHOULD, BooleanClause.Occur.MUST, BooleanClause.MUST_NOT])
		#query = MultiFieldQueryParser("contents", analyzer).parse(command,fields,flags,analyzer)
		scoreDocs = searcher.search(query, 50).scoreDocs
		print "%s total matching documents." % len(scoreDocs)

		for scoreDoc in scoreDocs:
			doc = searcher.doc(scoreDoc.doc)
			print 'ID:', doc.get("id"), 'Topic:', doc.get("topic"), 'Score:', scoreDoc.score


if __name__ == '__main__':
	lucene.initVM(vmargs=['-Djava.awt.headless=true'])
	print 'lucene', lucene.VERSION
	base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
	directory = SimpleFSDirectory(Paths.get(os.path.join(base_dir, INDEX_DIR)))
	searcher = IndexSearcher(DirectoryReader.open(directory))
	analyzer = StandardAnalyzer()
	run(searcher, analyzer)
	del searcher
