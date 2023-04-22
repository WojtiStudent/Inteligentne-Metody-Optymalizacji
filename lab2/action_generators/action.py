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
