import numpy as np 

class ThompsonSampling:
	def __init__(self, N):
		self.alpha = np.ones(N)
		self.beta = np.ones(N)
		self.last_result = -1

	def tick(self) -> int:
		result = np.random.beta(self.alpha, self.beta)
		self.last_result = int(np.argmax(result))
		return self.last_result

	def update(self, outcome: float) -> None:
		if outcome:
			self.alpha[self.last_result] += 1
		else:
			self.beta[self.last_result] += 1