from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def register_field(self, field):
        pass

    @abstractmethod
    def step(self):
        pass
