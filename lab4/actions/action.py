import random
from dataclasses import dataclass


@dataclass
class Action:
    name: str
    i: int
    j: int
    cycle_index: int
    do: callable
    change_type: str = None

    def __init__(self, cycles):
        self.name = random.choice(["swapEdgesInsideCycle", "swapVerticesOutsideCycle"])
        
        if self.name == "swapEdgesInsideCycle":
            self.cycle_index = random.choice([0, 1])
            self.i = random.choice(cycles[self.cycle_index])
            self.j = random.choice(cycles[self.cycle_index])
            while self.i == self.j:
                self.j = random.choice(cycles[self.cycle_index])
            self.change_type = random.choice(["before", "after"])
            self.do = self._swap_edges_inside_cycle
        elif self.name == "swapVerticesOutsideCycle":
            self.cycle_index = -1
            self.i = random.choice(cycles[0])
            self.j = random.choice(cycles[1])
            self.do = self._swap_vertices_outside_cycle


    def __repr__(self):
        return f"{self.name}({self.i}, {self.j}, {self.cycle_index}, {self.delta})"
    
    def _swap_edges_inside_cycle(self, cycle):
        i_index = cycle.index(self.i)
        j_index = cycle.index(self.j)

        if i_index > j_index:
            i_index, j_index = j_index, i_index

        if self.change_type == "after":
            reversed = cycle[i_index + 1 : j_index + 1]
            reversed.reverse()
            cycle[i_index + 1 : j_index + 1] = reversed
        elif self.change_type == "before":
            reversed = cycle[i_index : j_index]
            reversed.reverse()
            cycle[i_index : j_index] = reversed

    def _swap_vertices_outside_cycle(self, cycles):
        i_index = cycles[0].index(self.i)
        j_index = cycles[1].index(self.j)
        cycles[0][i_index], cycles[1][j_index] = cycles[1][j_index], cycles[0][i_index]


    