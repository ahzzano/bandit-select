class RoundRobin:
	def __init__(self, n_arms: int):
		self.n_arms =  n_arms
		self._current = 0

	def tick(self) -> int:
		return 0

	def update(self, outcome: float):
		self._current = (self._current + 1) % self.n_arms
		return