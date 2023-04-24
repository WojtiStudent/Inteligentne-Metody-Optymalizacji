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
    prev_i: int = None
    prev_j: int = None


    def __repr__(self):
        # return f"{self.name}({self.i}, {self.j}, {self.cycle_index}, {self.delta})"
        return f"""{self.name}(i={self.i}, j={self.j}, cycle={self.cycle_index}, delta={self.delta},
                    next_i={self.next_i}, next_j={self.next_j}, prev_i={self.prev_i}, prev_j={self.prev_j})"""