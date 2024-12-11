"""
This module provides the SpawnHolder class for managing spawn data in the game environment.

The SpawnHolder class allows loading spawn data from CSV files and retrieving
the data for specific maps. The data is stored in a dictionary where the key
is the map name and the value is a list of tuples containing spawn coordinates
and angle.

Example usage:
    spawn_holder = SpawnHolder()
    spawn_holder.load_data_from_file('map1')
    data = spawn_holder.get_data('map1')

"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import csv

@dataclass
class SpawnHolder:
    """Class for holding and managing spawn data.

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

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
            ValueError: If the CSV file contains rows that cannot be converted to the expected types.
        """
        with open("spawn_info/" + map_name + ".csv", mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if map_name in self.data:
                    self.data[map_name].append((int(row[0]), int(row[1]), float(row[2])))
                else:
                    self.data[map_name] = [(int(row[0]), int(row[1]), float(row[2]))]
    
    def get_data(self, key: str) -> List[Tuple[int, int, float]]:
        """Retrieves the spawn data for a given map.

        Args:
            key: The name of the map for which to retrieve the spawn data.

        Returns:
            A list of tuples, where each tuple contains two integers and a float,
            representing the spawn coordinates and angle.
        """

        return self.data.get(key, [])