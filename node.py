# Node class that contains the basic data storage for each point

import typing

class Node:

    def __init__(self, location:tuple, pressure: float, temperature: float, density:float, viscosity:float, velocity:tuple):
        self.x_location = location[0]
        self.x_velocity = velocity[0]
        self.y_location = location[1]
        self.y_velocity = velocity[1]
        self.pressure = pressure
        self.temperature = temperature
        self.pressure = pressure
        self.viscosity = viscosity
    
