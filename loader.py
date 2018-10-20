# loader
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import warnings
import string

class CharEncoder(object):
	"""docstring for CharOnehotEnc"""
	def __init__(self, datapath):
		super(CharEncoder, self).__init__()
		warnings.filterwarnings(action='ignore', category=DeprecationWarning)
		charDict = {}
		for filename in ['train', 'dev', 'test']:
			with open(datapath[filename], 'r', encoding='latin-1') as f:
				line = f.readline()
				while line:
					for c in line:
						charDict.setdefault(c, None)
					line = f.readline()
		charList = list(charDict.keys())
		self.labelEnc = LabelEncoder()
		self.labelEnc.fit(charList)   # integer_encoded = 
		ints = self.labelEnc.transform(charList)
		self.onehotEnc = OneHotEncoder(sparse=False)
		self.onehotEnc.fit(ints.reshape(len(ints), 1))
		self.inplen = self.onehotEnc.n_values_
		
	def trans(self, charList): # a list of chars
		integer_encoded = self.labelEnc.transform(charList)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = self.onehotEnc.transform(integer_encoded)
		return onehot_encoded

	def inverse(self, onehot_encoded):
		return self.labelEnc.inverse_transform([onehot_encoded[i].dot(self.onehotEnc.active_features_).astype(int) for i in range(len(onehot_encoded))])

class _Loader(object):
	"""docstring for Loader"""
	def __init__(self, mode, datapath, endec):
		super(_Loader, self).__init__()
		self.w2n = {'ENGLISH':0, 'ITALIAN':1, 'FRENCH':2}
		self.n2w = ('ENGLISH', 'ITALIAN', 'FRENCH')
		self.mode = mode
		self.datapath = datapath[mode]
		self.endec = endec
		self.data = self._loadAll()  #data[lineIndx][unitIndx]:(English, str]
		self.idx2d = self._get2dIdx()
		
	def _get2dIdx(self):
		idx2d = []
		for i in range(len(self.data)):
			for j in range(len(self.data[i])):
				idx2d.append((i,j))
		return idx2d

	def _breakSentence(self, sentence):
		substrList = []
		wordList = sentence.split(' ')
		for word in wordList:
			if(word in string.punctuation or word=='\n' or word=='--'):
				continue
			if(len(word)<5):
				comp = 5-len(word)
				substrList.append(list(word)+[' ' for i in range(comp)])
			else:
				for i in range(len(word)-5):#list(set([0,len(word)-5])):
					substrList.append(list(word[i:i+5])) # ['s','a','p',...]
		return substrList

	def _loadLines(self):
		lineList = []
		with open(self.datapath, "r", encoding="latin-1") as f:
			line = f.readline()
			while line:
				lineList.append(line)
				line = f.readline()
		return lineList

	def _loadAll(self):
		# load all data
		lineList = self._loadLines()
		# if test, then join in the soln
		if(self.mode=='test'):
			# load soln
			with open(self.datapath+'_solutions', "r", encoding="latin-1") as f:
				cnt = 0
				while True:
					soln = f.readline()
					if(not soln):
						break
					soln = soln.split(' ')[1].strip('\n').upper()
					lineList[cnt] = soln+' '+lineList[cnt]
					cnt += 1

		data = [[] for i in range(len(lineList))]
		for i in range(len(lineList)):
			language, sentence = lineList[i].split(' ',1)
			substrList = self._breakSentence(sentence)
			for j in range(len(substrList)):
				assert(language in self.n2w)
				data[i].append((self.w2n[language], 
					substrList[j]))
		return data

class _TrainDataLoader(_Loader):
	"""docstring for _TrainDataLoader"""
	def __init__(self, mode, datapath, endec, batchSize, isShuffle=True):
		super(_TrainDataLoader, self).__init__(mode, datapath, endec)
		self.batchSize = batchSize
		self.dataSize = len(self.idx2d)
		self.stepNum = int(self.dataSize / self.batchSize)
		self.metaIdx = list(range(self.dataSize))
		self.isShuffle = isShuffle
		if self.isShuffle:
			self.shuffle()
		self.cnt = 0

	def restartCnt(self):
		self.cnt = 0
		return

	def shuffle(self):
		self.metaIdx = np.random.permutation(self.metaIdx)
		return

	def _loadBatch(self):
		cur = self.cnt*self.batchSize
		# batchIdx = self.metaIdx[cur:cur+self.batchSize]
		batchdata = []
		batchlabel = []
		for i in range(self.batchSize):
			(lidx, uidx) = self.idx2d[self.metaIdx[(cur+i)%self.dataSize]]
			label, data = self.data[lidx][uidx]
			data = self.endec.trans(data).flatten()
			label = np.eye(3)[label]
			batchdata.append(data)
			batchlabel.append(label)
		return [np.array(batchlabel), np.array(batchdata)]

	def feed(self):
		if(self.cnt < self.stepNum):
			label, data = self._loadBatch()
			self.cnt += 1 
			return {'label': label,
					'data': data}
		else:
			self.restartCnt()
			if self.isShuffle:
				self.shuffle()
			return self.feed()

class _EvalDataLoader(_Loader):
	"""docstring for _EvalDataLoader"""
	def __init__(self, mode, datapath, endec):
		super(_EvalDataLoader, self).__init__(mode, datapath, endec)
		self.dataSize = len(self.idx2d)
		self.metaIdx = list(range(self.dataSize))
		self.cnt = 0

	def feed(self):
		if(self.cnt < self.dataSize):
			(lidx, uidx) = self.idx2d[self.metaIdx[self.cnt]]
			self.cnt += 1
			label, data = self.data[lidx][uidx]
			data = np.expand_dims(self.endec.trans(data).flatten(),0)
			return {'lineIdx':lidx, 
					'label':label, 
					'data':data} #[lidx, self.data[lidx][uidx]]   # return line index and data from eval loader.
		else:
			self.cnt = 0
			return {}

class LoaderHandler(object):
	"""docstring for DataLoader"""
	"""encryption of _Train and _Eval data loaders."""
	def __init__(self):
		super(LoaderHandler, self).__init__()
		self.modeList = ['train','dev','test']

	def getLoader(self, mode, datapath, endec, batchSize=4, forceEval=False):
		assert(mode in self.modeList)
		if forceEval or mode!='train':
			return _EvalDataLoader(mode, datapath, endec)
		else:
			return _TrainDataLoader(mode, datapath, endec, batchSize)




