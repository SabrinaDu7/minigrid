import random
import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import PATTERNS, IDX_TO_COLOR


def reject_lava_rooms(env, pos):
    """
    Function to filter out object positions that are in the lava rooms
    """
    x, y = pos
    marked = x <= env.roomsize or y <= env.roomsize or x>= (env.width - env.roomsize - 1) or y >= (env.height - env.roomsize - 1)
    return (not marked)

class Lava_Donut_Env(MiniGridEnv):

    def __init__(
        self,
        size=16,
        Lwidth=10, Lheight=8,
        agent_start_pos=None,
        agent_start_dir=None,
        tri_color='blue',
        plus_color='red',
        x_color='yellow',
        order = 'TPXD',
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir 
        
        self.Lwidth = Lwidth
        self.Lheight = Lheight
        self.tri_color = tri_color
        self.plus_color = plus_color
        self.x_color = x_color
        self.shuffle_indices = [0,1,2]
        self.order = order
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=10*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            **kwargs
        )


    @staticmethod
    def _gen_mission():
        return "avoid the real lava and get to the fake lava square"

    def _gen_grid(self, width, height, regenerate=True):
        if regenerate:
            # Create an empty grid
            self.grid = Grid(width, height)
            
            # Generate the surrounding walls
            #Consider: walls at -1, rather than 0
            self.grid.horz_wall(0,0)
            self.grid.vert_wall(0,0)
            self.grid.horz_wall(0,height-1)
            self.grid.vert_wall(width-1,0)

            offset=6

            x_diam = 8
            y_diam = 4
                
            #Place the shapes
            triloc  =   (width/3-4,height/3-4)
            plusloc =   (2*width/3-2,height/3-4)
            xloc    =   (width/3-3,2*height/3-2)
            dashloc =   (2*width/3-2,2*height/3-2)

            loc = [(width/3-4,height/3-4), (2*width/3-1,height/3-1), (width/3-3,2*height/3-2), (2*width/3-2,2*height/3-2)]

            shapes = {}

            shapes['T'] = {'name': 'triangle', 'color': self.tri_color}
            shapes['P'] = {'name': 'dash', 'color': self.plus_color}
            shapes['X'] = {'name': 'x', 'color': self.x_color}
            shapes['D'] = {'name': 'dash', 'color': self.tri_color}

            # Create dictionary 

            for idx, char in enumerate(self.order):
                self.place_shape(shapes[char]['name'], loc[idx], shapes[char]['color'])

            #Adding shapes on the bottom and top of the map
            self.place_shape('plus', (width/3-1,height/3-5), self.x_color)
            self.place_shape('plus', (width/3,height/3-5), self.x_color)
            self.place_shape('plus', (width/3+1,height/3-5), self.x_color)
            self.place_shape('plus', (width/3+2,height/3-5), self.x_color)
            self.place_shape('plus', (width/3-3,height/3+6), self.plus_color)
            self.place_shape('plus', (width/3-2,height/3+6), self.plus_color)
            self.place_shape('plus', (width/3-1,height/3+6), self.plus_color)
            self.place_shape('plus', (width/3,height/3+6), self.plus_color)

            #Adding the central rooms
            self.grid.horz_wall(int(self.Lwidth/2), int(height/2)-y_diam, length=x_diam)
            self.grid.horz_wall(int(self.Lwidth/2), int(height/2)+y_diam, length=x_diam)
            # self.grid.vert_wall(int(self.Lheight/2)-x_diam, int(width/2), length=y_diam)
            # self.grid.vert_wall(int(self.Lheight/2)+x_diam, int(width/2), length=y_diam)

            
            # Place the agent
            if self.agent_start_pos is not None:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.place_agent()

        else:
            # Place the agent
            if self.agent_start_pos is not None:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.place_agent()
    
    
    def place_shape(self,shape,pos,color):
        """
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid={
            'plus':np.array(
                [[0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,1,1,0,0]]),
            'triangle':np.array(
                [[1,1,1,1,1,0],
                 [1,1,1,1,1,1],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0]]),
            'x':np.array(
                [[0,0,0,0,1,1],
                 [1,1,1,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0]]),
            'dash':np.array(
                [[0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,0,0],
                 [0,0,0,0,1,1],
                 [0,0,0,0,1,1]])
            }
            
        shapecoords = np.transpose(np.nonzero(shapegrid[shape]))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])
        
        


class LavaDonutEnv_16(Lava_Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=16, agent_start_pos=None, **kwargs)

class LavaDonutEnv_18(Lava_Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, **kwargs)

class LavaDonutEnv_20(Lava_Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, **kwargs)