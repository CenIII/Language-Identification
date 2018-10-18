#trainer
import pickle
import tqdm
import numpy as np

class Trainer(object):
	"""docstring for Trainer"""
	def __init__(self, net, crit, loader, loaderDev, epochNum=3, lr=2e-4):
		super(Trainer, self).__init__()
		self.lr = lr 
		self.net = net
		self.crit = crit
		self.loader = loader
		self.loaderDev = loaderDev
		self.epochNum = epochNum
		self.evaluator = Evaluator()

	def train(self):
		maxAcc = 0
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
			acc = self.evaluator.eval(self.net, self.loaderDev)
			if(acc > maxAcc):
				maxAcc = acc
				with open("net.obj","wb") as f:
					pickle.dump(self.net, f)
			print('epoch '+str(epoch)+' acc: '+str(acc))

	def optimize(self, net):
		for fc in self.net.fc:
			for key in fc.params:
				fc.params[key] -= self.lr*fc.grad[key]
		return


class Evaluator(object):
	"""docstring for Evaluator"""
	def __init__(self):
		super(Evaluator, self).__init__()

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

		# accuracy
		acc = np.equal(predList, gtList)
		acc = acc.sum()/len(acc)
		return acc
		