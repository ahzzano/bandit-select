from random import random, randint
import numpy as np 

class EpsilonGreedy:
	def __init__(self, n_arms: int, epsilon: float):
		self.n_arms = n_arms
		self.epsilon = epsilon
		self.visits = np.zeros(n_arms)
		self.satisfaction = np.zeros(n_arms)
		self.last_output: int = 0

	def tick(self) -> int:
		if random() < self.epsilon:
			self.last_output = randint(0, self.n_arms - 1)	
		else:
			output = np.argmax(self.satisfaction / (self.visits + 1e-5))
			self.last_output = int(output)
		
		return self.last_output

	def update(self, outcome: float):
		if outcome:
			self.satisfaction[self.last_output] += 1

		self.visits[self.last_output] += 1