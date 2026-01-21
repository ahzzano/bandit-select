import numpy as np 

class UpperConfidenceBound:
	def __init__(self, n_arms: int, c: float=1):
		self.c = c
		self.visits = np.zeros(n_arms)
		self.score = np.zeros(n_arms)
		self.ticks = 1
		self.last_output = -1

	def tick(self) -> int:
		ri = self.score / (self.visits + 1e-5)		
		self.ticks += 1
		ucb = ri + self.c * np.sqrt(np.log(self.ticks) / (self.visits + 1e-5))
		self.last_output = int(np.argmax(ucb))

		return self.last_output


	def update(self, outcome: float) -> None:
		self.visits[self.last_output] += 1
		self.score[self.last_output] -= outcome