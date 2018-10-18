#run

import numpy as np
import sys
import pickle
from trainer import Trainer, Evaluator
from layers import LINet, Criterion
from loader import LoaderHandler, CharEncoder

### data path
dataPath = './languageIdentification.data'
### model params
hiddenSize = 600
### training params
epochNum = 30
batchSize = 128
lr = 5e-3
### 
endec = CharEncoder(dataPath)

def runTrain():
	lhandler = LoaderHandler()
	ldTrain = lhandler.getLoader('train', dataPath, endec, batchSize=batchSize)
	ldDev = lhandler.getLoader('dev', dataPath, endec)
	net = LINet(hiddenSize)
	crit = Criterion()
	trainer = Trainer(net, crit, ldTrain, ldDev, epochNum=epochNum, lr=lr)
	net = trainer.train()
	return net

def runEval(netpath, mode):
	# load net
	net = None
	with open(netpath,'rb') as f:
		net = pickle.load(f)
	# load dataloader & evaluator
	lhandler = LoaderHandler()
	loader = lhandler.getLoader(mode, dataPath, endec)
	evaluator = Evaluator()
	acc = evaluator.eval(net, loader)
	print('Accuracy under '+mode+' mode is:'+str(acc))


def main():
	mode = sys.argv[1]
	netpath = sys.argv[2]
	if(mode=='train'):
		runTrain()
	elif(mode=='dev' or mode=='test'):
		runEval(netpath, mode)
	else:
		print('no such mode. please check your options.')
		exit(0)

if __name__ == "__main__":
	main()


