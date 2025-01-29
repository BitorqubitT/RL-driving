"""
This module provides the SpawnHolder class and Args class.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import csv
import itertools
from dataclasses import dataclass, field
from typing import List

@dataclass
class SpawnHolder:
    """Class for holding and managing spawn data.

    The SpawnHolder class allows loading spawn data from CSV files and retrieving
    the data for specific maps. The data is stored in a dictionary where the key
    is the map name and the value is a list of tuples containing spawn coordinates
    and angle.

    Example usage:
        spawn_holder = SpawnHolder()
        spawn_holder.load_data_from_file('map1')
        data = spawn_holder.get_spawn_data('map1')

    Attributes:
        data: A dictionary where the key is a string representing the map name,
              and the value is a list of tuples. Each tuple contains three elements:
              two integers and a float, representing the spawn coordinates and angle.
    """
    data: Dict[str, List[Tuple[int, int, float]]] = field(default_factory = dict)

    def load_data_from_file(self, map_name: str):
        """Loads spawn data from a CSV file and stores it in the data attribute.

        Args:
            map_name: The name of the map for which to load the spawn data. The method
                      expects a CSV file located in the 'spawn_info' directory.
                      Each row in the CSV file should contain three values: two integers and a float.

        """
        with open("spawn_info/" + map_name + ".csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if map_name in self.data:
                    self.data[map_name].append((int(row[0]), int(row[1]), float(row[2])))
                else:
                    self.data[map_name] = [(int(row[0]), int(row[1]), float(row[2]))]
    
    def get_spawn_data(self, key: str) -> List[Tuple[int, int, float]]:
        """Retrieves the spawn data for a given map.

        Args:
            key: The name of the map for which to retrieve the spawn data.

        Returns:
            A list of tuples, where each tuple contains two integers and a float,
            representing the spawn coordinates and angle.
        """
        return self.data.get(key, [])

@dataclass
class Args:
    batch_size: List[int] = field(default_factory=list)
    gamma: List[float] = field(default_factory=list)
    tau: List[float] = field(default_factory=list)
    lr: List[float] = field(default_factory=list)
    episodes: List[int] = field(default_factory=list)
    eps_start: List[float] = field(default_factory=list)
    eps_end: List[float] = field(default_factory=list)
    eps_decay: List[int] = field(default_factory=list)
    n_actions: List[int] = field(default_factory=list)
    n_observations: List[int] = field(default_factory=list)
    game_map: List[str] = field(default_factory=list)
    architecture: str = ""
    loss_function: str = ""
    start_pos: List[str] = field(default_factory=list)
    minibatch_size: List[int] = field(default_factory=list)
    num_iterations: List[int] = field(default_factory=list)
    seed: int = 0
    num_steps: List[int] = field(default_factory=list)
    num_envs: int = 0
    update_epochs: List[int] = field(default_factory=list)
    clip_coef: List[float] = field(default_factory=list)
    clip_vloss: bool = False
    anneal_lr: bool = False
    gae_lambda: float = 0.0
    norm_adv: bool = False
    ent_coef: float = 0.0
    vf_coef: float = 0.0
    max_grad_norm: float = 0.0
    target_kl: float = 0.0
    
    def check_and_iterate_combinations(self):
        list_attributes = {field_name: value for field_name, value in self.__dict__.items() if isinstance(value, list) and value}
        combinations = list(itertools.product(*list_attributes.values()))
        
        return combinations
