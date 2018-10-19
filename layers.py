#layers

import numpy as np

class SigmoidLayer(object):
	"""docstring for SigmoidLayer"""
	def __init__(self, hiddenSize):
		super(SigmoidLayer, self).__init__()
		self.params = None
		self.size = hiddenSize
		self.grad = None

	def forward(self, input):
		self.output = 1/(1+np.exp(-input))
		return self.output

	def backward(self, dv_output): 
		dv_input = dv_output*self.output*(1-self.output)
		return dv_input

class LinearLayer(object):
	"""docstring for FC"""
	def __init__(self, inputSize, outputSize):
		super(LinearLayer, self).__init__()
		self.params = {'W': np.random.uniform(-1,1,(outputSize,inputSize)),
					   'b': np.random.uniform(-1,1,outputSize)}
		self.grad = {'W': np.zeros([outputSize, inputSize]),
					 'b': np.zeros(outputSize) }
		self.inputSize = inputSize
		self.outputSize = outputSize

	def forward(self, input): #[batch, inputSize]
		self.input = input
		output = self.input.dot(self.params['W'].transpose())+self.params['b']
		return output

	def backward(self, dv_output): #[batch, outputSize]
		self.grad['W'] = dv_output.transpose().dot(self.input)/len(self.input)
		self.grad['b'] = dv_output.sum(axis=0)/len(self.input)
		dv_input = dv_output.dot(self.params['W'])
		return dv_input #[batch, inputSize]
		
class SoftmaxLayer(object):
	"""docstring for SoftmaxLayer"""
	def __init__(self):
		super(SoftmaxLayer, self).__init__()
		self.params = None
		self.grad = None

	def forward(self, input):
		#self.input = input
		exp_y = np.exp(input)
		self.output = exp_y/exp_y.sum(axis=1)[:,None]
		return self.output

	def backward(self, dv_output):
		#tmpOut = self.forward(self.input) 
		batchSize, dataLen = self.output.shape
		dv_input = np.zeros([batchSize, dataLen])
		for b in range(batchSize):
			y = self.output[b]
			Ly = dv_output[b]
			delta = np.eye(dataLen)
			dv_input[b] = (Ly[:,None]*y[:,None]*(delta - y)).sum(axis=0)
		return dv_input
		
class LINet(object):
	"""docstring for LINet"""
	def __init__(self, hiddenSize, inputSize):
		super(LINet, self).__init__()
		self.fc = []
		self.fc.append(LinearLayer(inputSize, hiddenSize))
		self.sig = SigmoidLayer(hiddenSize)
		self.fc.append(LinearLayer(hiddenSize, 3))
		self.softmax = SoftmaxLayer()

	def forward(self, input):
		fc0_o = self.fc[0].forward(input)
		sig_o = self.sig.forward(fc0_o)
		fc1_o = self.fc[1].forward(sig_o)
		output = self.softmax.forward(fc1_o)
		return output  # [batch, 3]

	def backward(self, dv_output):
		softmax_dvi = self.softmax.backward(dv_output)
		fc1_dvi = self.fc[1].backward(softmax_dvi)
		sig_dvi = self.sig.backward(fc1_dvi)
		fc0_dvi = self.fc[0].backward(sig_dvi)
		return fc0_dvi
		

class Criterion(object):
	"""docstring for Criterion"""
	def __init__(self):
		super(Criterion, self).__init__()

	def forward(self, input, label):
		self.input = input   # [batch, 3]
		self.label = label
		loss = 0.5*((self.input-self.label)**2).sum()/len(self.input)
		return loss

	def backward(self, dv_output):
		dv_input = self.input-self.label
		return dv_input
		