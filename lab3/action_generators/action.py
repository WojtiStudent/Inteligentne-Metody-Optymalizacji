from dataclasses import dataclass


@dataclass
class Action:
    name: str
    i: int
    j: int
    cycle_index: int
    delta: int
    do: callable
    change_type: str = None
    next_i: int = None
    next_j: int = None


    def __repr__(self):
        return f"{self.name}({self.i}, {self.j}, {self.cycle_index}, {self.delta})"