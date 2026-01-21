class ExploreOnly:
	def __init__(self, N: int):
		self.N = N 
		self.i = N

	def tick(self) -> int:
		self.i = (self.i + 1) % self.N 
		return self.i

	def update(self, outcome: float) -> None:
		return 