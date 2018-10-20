#run
import numpy as np
import sys
import pickle
from trainer import Trainer, Evaluator
from layers import LINet, Criterion
from loader import LoaderHandler, CharEncoder

class cfg(object):
	###### CONFIGURATION ######
	### data path
	dataPath = {} #'./languageIdentification.data'
	outDir = './out'
	mode = 'train'
	### if mode=='eval':
	evalset = ''
	netpath = ''
	### model params
	hiddenSize = 100
	### training params
	epochNum = 20
	batchSize = 128
	lr = 1e-1
	### encode
	endec = None

def parseCmd():  # -e train ./net.obj -o
	trainpath, devpath, testpath = sys.argv[1:4]
	cfg.dataPath = {'train':trainpath, 'dev':devpath, 'test':testpath}
	cfg.endec = CharEncoder(cfg.dataPath)
	if('-e' in sys.argv):
		idx = sys.argv.index('-e')
		cfg.mode = 'eval'
		cfg.evalset = sys.argv[idx+1]
		cfg.netpath = sys.argv[idx+2]
	if('-o' in sys.argv):
		idx = sys.argv.index('-o')
		cfg.outDir = sys.argv[idx+1]
	if('-h' in sys.argv):
		idx = sys.argv.index('-h')
		cfg.hiddenSize = int(sys.argv[idx+1])
	if('-l' in sys.argv):
		idx = sys.argv.index('-l')
		cfg.lr = float(sys.argv[idx+1])

def runTrain():
	print('- Start training:')
	lhandler = LoaderHandler()
	net = LINet(cfg.hiddenSize, int(cfg.endec.inplen*5))
	crit = Criterion()
	trainer = Trainer(net, crit, lhandler, cfg, epochNum=cfg.epochNum, lr=cfg.lr, outdir=cfg.outDir)
	net = trainer.train()
	ldTest = lhandler.getLoader('test', cfg.dataPath, cfg.endec)
	evaluator = Evaluator(outdir=cfg.outDir)
	acc = evaluator.eval(net, ldTest)
	print('- Accuracy on test set: '+str(np.round(acc,2)))
	return net

def runEval():
	print('- Start evaluating:')
	net = None
	with open(cfg.netpath,'rb') as f:
		net = pickle.load(f)
	lhandler = LoaderHandler()
	loader = lhandler.getLoader(cfg.evalset, cfg.dataPath, cfg.endec)
	evaluator = Evaluator(outdir=cfg.outDir)
	acc = evaluator.eval(net, loader)
	print('- Accuracy on '+cfg.evalset+' set is:'+str(np.round(acc,2)))


def main():
	parseCmd()
	if(cfg.mode=='train'):
		runTrain()
	else:
		runEval()

if __name__ == "__main__":
	main()
