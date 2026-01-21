import bandit

class BasicSolver:
	def __init__(self, n_arms: int):
		self.n_arms =  n_arms

	def tick(self) -> int:
		return 0

	def update(self, outcome: float):
		return