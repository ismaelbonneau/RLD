import numpy as np
from collections import deque
import random

import torch
from torch.nn import SmoothL1Loss

def sample_minibatch(size):
	pass

class DQNMemory(object):
	"""docstring for DQNMemory"""
	def __init__(self, batchsize, max_size=300):
		super(DQNMemory, self).__init__()
		self.max_size = max_size
		self.batchsize = batchsize
		self.memory = deque(maxlen=max_size)

	def append(self, data):
		self.memory.append((obs, chosen_action, reward, new_obs))

	def sample_batch(self, gamma, target_func):
		indices = np.random.choice(list(range(len(self.memory))), self.batchsize, replace=False)

		x_batch = []
		x_actions_batch = []
		y_batch = []

		for i in indices:
			obs, chosen_action, reward, new_obs = self.memory(i)
			x_batch.append(obs)
			x_actions_batch.append(chosen_action)
			y_batch.append(reward + gamma * torch.max(target_func(new_obs))[0].item())

		return torch.Tensor(x_batch), torch.Tensor(y_batch)
		

class NN(torch.nn.Module):
	""" simple Neural Net """
	def __init__(self, inSize, outSize, layers=[]):
		super(NN, self).__init__()
		self.layers = nn.ModuleList([])
		for x in layers:
			self.layers.append(nn.Linear(inSize, x))
			inSize = x
		self.layers.append(nn.Linear(inSize, outSize))

	def forward(self, x):
		x = self.layers[0](x)
		for i in range(1, len(self.layers)):
			x = torch.nn.functional.leaky_relu(x)
			x = self.layers[i](x)
		return x

class QLearner(object):
	""" DQN learner """

	def __init__(self, env, obsSize, batchsize=64, lr=0.001, gamma=0.99, epsilon=0.1):

		self.env = env

		self.batchsize = batchsize
		# replay memory
		self.memory = DQNMemory()
		self.obsSize = obsSize
		self.actionSpaceSize = env.action_space.n
		self.epsilon = epsilon
		self.gamma = gamma

		# fonctions Q apprises
		self.learning_rate = lr
		self.Q = NN(self.obsSize, self.actionSpaceSize, layers=[16])
		self.optim = torch.optim.Adam(self.Q.parameters(), lr=self.learning_rate)
		self.target_func = NN(self.obsSize, self.actionSpaceSize, layers=[16])
		self.target_func.load_state_dict(self.Q.state_dict()) # les deux ont les mêmes poids
		self.loss = SmoothL1Loss()

		self.C = 10 # pourquoi pas?

	def step(self, step_num, obs, envm):

		# epsilon-greedy:
		if random.random() < self.epsilon:
			# sampler une action au hasard
			chosen_action = self.env.action_space.sample()
		else:
			# choisir la meilleure action au sens de Q
			with torch.no_grad():
				actions = self.Q(torch.from_numpy(obs.reshape(1, -1)))
				chosen_action = torch.argmax(actions).item()

		# execute action chosen_action in emulator:
		new_obs, reward, done, _ = envm.step(chosen_action)

		# store transition we just did in memory:
		self.memory.append((obs, chosen_action, reward, new_obs))

		# sample random minibatch of transitions from memory:
		x_batch, y_batch = self.memory.sample_batch()
		

		for i in range(50):
			self.Q.train()
			preds = self.Q(X)
			preds = preds.gather(1, torch.Tensor(X_actions).long().view(-1,1)).reshape(1, -1).squeeze()

			loss = self.loss(torch.Tensor(Y).long(), preds)
			loss.backward()
			self.optim.step()
			self.optim.zero_grad()

		# every C steps, reset target_func = Q:
		if step_num > 0 and (step_num % self.C == 0):
			self.target_func.load_state_dict(self.Q.state_dict())


		return new_obs, reward, done

	def run(self, envm):

		e = 0
		rsum = 0
		while True:

			obs, reward, done = self.step(e, obs, envm)
			e += 1

			rsum += reward

			if done:
				print("done after ", e , " iterations. rsum=", rsum)
				return rsum, e

		



