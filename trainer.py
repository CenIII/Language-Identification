#trainer
import pickle
import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, net, crit, lhandler, cfg, epochNum=3, lr=2e-4, outdir='./out'):
		super(Trainer, self).__init__()
		self.lr = lr 
		self.net = net
		self.crit = crit
		self.loader = lhandler.getLoader('train', cfg.dataPath, cfg.endec, batchSize=cfg.batchSize)
		self.loaderTrain = lhandler.getLoader('train', cfg.dataPath, cfg.endec, forceEval=True)
		self.loaderDev = lhandler.getLoader('dev', cfg.dataPath, cfg.endec)
		self.epochNum = epochNum
		self.evaluator = Evaluator(outdir=outdir)
		self.savedir = outdir
		os.makedirs(self.savedir, exist_ok=True)

	def plotAcc(self, accTrain, accDev):
		xaxis = np.array(list(range(len(accDev))))
		plt.plot(xaxis, accTrain,c='r', label='Train')
		plt.plot(xaxis, accDev,c='b', label='Dev')
		plt.title('Accuracy vs. Epoches')
		plt.xlabel('Epoches')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(os.path.join(self.savedir,'accuracy.png'))
		np.save(os.path.join(self.savedir,'accuracy.npy'),np.array([accTrain, accDev]))

	def train(self):
		maxAcc = 0
		accTrain, accDev = [], []
		for epoch in range(self.epochNum):
			qdar = tqdm.tqdm(range(self.loader.stepNum),
									total= self.loader.stepNum,
									ascii=True)
			for step in qdar:  # this is not a constraint but a alignmentor. loader doesn't care how many steps you run for an epoch. 
				batch = self.loader.feed()
				output = self.net.forward(batch['data'])
				loss = self.crit.forward(output, batch['label'])
				grad_c = self.crit.backward(loss)
				self.net.backward(grad_c)
				self.optimize(self.net)
				qdar.set_postfix(loss=str(np.round(loss*1e4,2)))
			accTrain.append(self.evaluator.eval(self.net, self.loaderTrain))
			accDev.append(self.evaluator.eval(self.net, self.loaderDev))
			print('epoch '+str(epoch)+' Train acc: '+str(np.round(accTrain[-1],2))+' Dev acc: '+str(np.round(accDev[-1],2)))
			if(accDev[-1] > maxAcc):
				maxAcc = accDev[-1]
				with open(os.path.join(self.savedir, 'bestnet.obj'), "wb") as f:
					pickle.dump(self.net, f)
		# plot and save 
		self.plotAcc(accTrain, accDev)
		return self.net

	def optimize(self, net):
		for fc in self.net.fc:
			for key in fc.params:
				fc.params[key] -= self.lr*fc.grad[key]
		return


class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self, outdir='./out'):
		super(Evaluator, self).__init__()
		self.savedir = outdir
		os.makedirs(self.savedir, exist_ok=True)

	def eval(self, net, loader):
		# feed forward
		cur_lidx = 0
		line_preds = []
		predList = []
		gtList = []
		def appendRes(label, line_preds):
			gtList.append(label)
			predList.append(np.argmax(np.bincount(line_preds)))
		while True:
			inp = loader.feed()
			if inp=={}:
				appendRes(label, line_preds)
				break
			lineIdx = inp['lineIdx']
			if lineIdx!=cur_lidx:
				appendRes(label, line_preds)
				cur_lidx = lineIdx
				line_preds = []
			label = inp['label']
			data = inp['data']
			line_preds.append(np.argmax(net.forward(data)))

		# print to file if loader.mode=='test'
		if(loader.mode=='test'):
			#TODO: print to file in self.savedir
			lineList = loader._loadLines()
			with open(os.path.join(self.savedir,'languageIdentificationPart1.output'),'w') as f:
				for i in range(len(lineList)):
					f.write(lineList[i][:-1]+' '+loader.n2w[predList[i]]+'\n')
		# accuracy
		acc = np.equal(predList, gtList)
		acc = acc.sum()/len(acc)
		return acc
		