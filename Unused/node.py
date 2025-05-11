# Node class that contains the basic data storage for each point

import typing

## Define a Node that is the representation of a single point on a grid. It stores all relevant info about
# the fluid at the point given. This includes velocity components, pressure, density, viscosity, location, and mesh id
# Currently this is equip to work on a 2D plane 
class Node:

    def __init__(self, location:tuple, pressure: float, density:float, viscosity:float, velocity:tuple, mesh_id:int):
        self.x_location = location[0]
        self.x_velocity = velocity[0]
        self.y_location = location[1]
        self.y_velocity = velocity[1]
        self.pressure = pressure
        self.pressure = pressure
        self.viscosity = viscosity
        self.mesh_id = mesh_id
        
        
    
