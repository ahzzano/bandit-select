from typing import Protocol
from dataclasses import dataclass
from datetime import datetime

class Solver(Protocol):
	def tick(self) -> int:
		...

	def update(self, outcome: float) -> None:
		...

class OVBandit:
	def __init__(self, models, solver: Solver):
		self.models = models
		self.runs = 0
		self.solver = solver
		self.selected_model = 0 # first model in the list is the default one

	def choose(self, model_idx):
		self.selected_model = model_idx

	def _infer(self, model, input):
		s = datetime.now()
		output = self.models[model](input)
		e = datetime.now()

		return (e-s).total_seconds() * 1e6,  output

	def predict(self, input):
		self.runs += 1
		# selected = self.solver.tick()
		inf_time, output = self._infer(self.selected_model, input)

		self.solver.update(inf_time) 

		return output

	def reset(self):
		self.runs = 0