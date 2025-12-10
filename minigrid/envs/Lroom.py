from minigrid.minigrid_env import *
from minigrid.core.world_object import *
from minigrid.core.mission import MissionSpace
import random
import numpy as np


class LEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=16,
        Lwidth=10, 
        Lheight=8,
        agent_start_pos: tuple| None = None,
        agent_start_dir=0,
        new_obj_pos: tuple | None = None,
        plus_color: str = "red",
        **kwargs
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir 
        self.plus_color = plus_color
        self.Lwidth = Lwidth
        self.Lheight = Lheight

        self.new_obj_pos = new_obj_pos
        self.new_obj_color = None if self.new_obj_pos is None else "green"

        mission_space = MissionSpace(mission_func=self._gen_mission)
        max_steps = kwargs.pop("max_steps", 10 * size * size)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
            see_through_walls=True,
            **kwargs
        )
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def _gen_mission():
        return "Reach the goal if there is one. Else wander around!"

    def _gen_grid(self, width, height, regenerate=True):
        # Create an empty grid
        self.grid = Grid(width, height)
        
        # Generate the surrounding walls
        #Consider: walls at -1, rather than 0
        self.grid.horz_wall(0,0)
        self.grid.vert_wall(0,0)
        self.grid.horz_wall(0,height-1)
        self.grid.vert_wall(width-1,0)
        for i in range(self.Lheight+1,height-1):
            self.grid.horz_wall(self.Lwidth+1, i, length=width-self.Lwidth-1)
        
        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            
        #Place the shapes
        triloc  =   (width/3-4,height/3-4)
        plusloc =   (2*width/3-2,height/3-4)
        xloc    =   (width/3-3,2*height/3-2)
        self.place_shape('triangle',triloc,'blue')
        self.place_shape('plus',plusloc, self.plus_color)
        self.place_shape('x',xloc,'yellow')

        # Place the new obj if specified
        self.mission = f"get to the new{self.new_obj_color} square"
        if self.new_obj_pos is not None and self.new_obj_color is not None:
            x, y = self.new_obj_pos
            self.put_obj(FloorBright(self.new_obj_color), x, y)
    
    
    def place_shape(self,shape,pos,color):
        """
        Place a 6x6 shape with lower left corner at (x,y)
        """
        shapegrid={
            'plus':np.array(
                [[0,0,1,1,0,0],
                 [0,0,1,1,0,0],
                 [1,1,1,1,1,1],
                 [1,1,1,1,1,1],
                 [0,0,1,1,0,0],
                 [0,0,1,1,0,0]]),
            'triangle':np.array(
                [[1,0,0,0,0,0],
                 [1,1,0,0,0,0],
                 [1,1,1,0,0,0],
                 [1,1,1,1,0,0],
                 [1,1,1,1,1,0],
                 [1,1,1,1,1,1]]),
            'x':np.array(
                [[1,1,0,0,1,1],
                 [1,1,1,1,1,1],
                 [0,1,1,1,1,0],
                 [0,1,1,1,1,0],
                 [1,1,1,1,1,1],
                 [1,1,0,0,1,1]]),
            }
            
        shapecoords = np.transpose(np.nonzero(shapegrid[shape]))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])
        

# LEnv variants:
# size=20, Lwidth=12, Lheight=10,
# size=18, Lwidth=10, Lheight=8,
# size=16, Lwidth=8, Lheight=6, goal_pos=[7, 2]
# size=18, Lwidth=10, Lheight=8, goal_pos=[9, 2]

class LEnv_16_green_line(LEnv):
      def __init__(self, **kwargs):
          super().__init__(size=16, Lwidth=8, Lheight=6,
                           agent_start_pos=None,
                           **kwargs)

      def _gen_grid(self, width, height, regenerate=True):
          # Let parent build the standard L-room grid
          super()._gen_grid(width, height, regenerate)

          # Add vertical green line between plus and triangle
          line_x = 7  # midpoint between the two shapes
          line_start_y = 2
          for i in range(4):
              self.put_obj(FloorBright("green"), line_x, line_start_y + i)


class LEnv_goal(LEnv):
    def __init__(self, agent_start_pos:tuple, 
                 size: int = 16, 
                 Lwidth: int = 8,
                 Lheight: int = 6,
                 goal_pos = [7, 2], 
                 **kwargs):

        super().__init__(size=size, Lwidth=Lwidth, Lheight=Lheight,
                         agent_start_pos=agent_start_pos,
                         **kwargs)
        
        self.goal_pos = goal_pos
        
    def _gen_grid(self, width, height, regenerate=True):
        super()._gen_grid(width, height, regenerate)
        
        # Place the goal if specified
        self.mission = f"get to the green goal square"
        if self.goal_pos is not None:
            x, y = self.goal_pos
            self.put_obj(Goal(), x, y)
