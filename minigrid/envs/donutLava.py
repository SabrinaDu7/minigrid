import random
import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Floor, Gates, Lava, Fake_Lava
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import PATTERNS, IDX_TO_COLOR


def reject_lava_rooms(env, pos):
    """
    Function to filter out object positions that are in the lava rooms
    """
    x, y = pos
    valid = x <= env.Lwidth/2 or y <= int(env.grid.height/2)-3 or x>= env.Lwidth/2 + 6 or y >= int(env.grid.height/2)+3
    return (not valid)

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
        neg = 0,
        max_steps=200,
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
        self.neg=neg
        
        mission_space = MissionSpace(mission_func=self._gen_mission)
        
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=max_steps,
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

            loc = [(width/3-4,height/3-4), (2*width/3-1,height/3-1), (width/3-3,2*height/3-2), (2*width/3-2,2*height/3-2)]

            shapes = {}

            shapes['T'] = {'name': 'triangle', 'color': self.tri_color}
            shapes['P'] = {'name': 'plus', 'color': self.plus_color}
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
            self.grid.horz_wall(int(self.Lwidth/2), int(height/2)-3, length=7)
            self.grid.horz_wall(int(self.Lwidth/2), int(height/2)+3, length=7)
            self.grid.vert_wall(int(self.Lwidth/2), int(height/2)-3, length=7)
            self.grid.vert_wall(int(self.Lwidth/2)+3, int(height/2)-3, length=7)
            self.grid.vert_wall(int(self.Lwidth/2)+6, int(height/2)-3, length=7)

            self.grid.set(int(self.Lwidth/2)+1, int(height/2)-3, Gates())
            self.grid.set(int(self.Lwidth/2)+2, int(height/2)-3, Gates())
            self.grid.set(int(self.Lwidth/2)+4, int(height/2)-3, Gates())
            self.grid.set(int(self.Lwidth/2)+5, int(height/2)-3, Gates())
            self.grid.set(int(self.Lwidth/2)+1, int(height/2)+3, Gates())
            self.grid.set(int(self.Lwidth/2)+2, int(height/2)+3, Gates())
            self.grid.set(int(self.Lwidth/2)+4, int(height/2)+3, Gates())
            self.grid.set(int(self.Lwidth/2)+5, int(height/2)+3, Gates())
            self.grid.set(int(self.Lwidth/2), int(height/2)-1, Gates())
            self.grid.set(int(self.Lwidth/2), int(height/2), Gates())
            self.grid.set(int(self.Lwidth/2), int(height/2)+1, Gates())
            self.grid.set(int(self.Lwidth/2)+6, int(height/2)-1, Gates())
            self.grid.set(int(self.Lwidth/2)+6, int(height/2), Gates())
            self.grid.set(int(self.Lwidth/2)+6, int(height/2)+1, Gates())


            # Place lava
            self.put_obj(Fake_Lava(), int(self.Lwidth/2)+2, int(height/2))
            self.put_obj(Lava(), int(self.Lwidth/2)+4, int(height/2))
            
            # Place the agent
            if self.agent_start_pos is not None:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.agent_pos = (-1, -1)
                pos = self.place_obj(None, reject_fn=reject_lava_rooms)
                self.agent_pos = pos
                self.agent_dir = self._rand_int(0, 4)

        else:
            # Place the agent
            if self.agent_start_pos is not None:
                self.agent_pos = self.agent_start_pos
                self.agent_dir = self.agent_start_dir
            else:
                self.agent_pos = (-1, -1)
                pos = self.place_obj(None, reject_fn=reject_lava_rooms)
                self.agent_pos = pos
                self.agent_dir = self._rand_int(0, 4)

    
    def step(self, action):
        
        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = tuple(fwd_pos)
            if fwd_cell is not None and (fwd_cell.type == "goal" or fwd_cell.type == "fake_lava"):
                terminated = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == "lava":
                reward = -self.neg #* self._reward()
                terminated = True
            # Move forward again if it's a Gates
            # if fwd_cell is Gates:
            #     fwd_pos = self.front_pos
            #     self.agent_pos = tuple(fwd_pos)


        # Pass
        elif action == self.actions.pickup:
            pass

        else:
            raise ValueError(f"Unknown action: {action}")

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}
    
    
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

class LavaDonutEnv_17(Lava_Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=17, agent_start_pos=None, **kwargs)

class LavaDonutEnv_18(Lava_Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=18, agent_start_pos=None, **kwargs)

class LavaDonutEnv_20(Lava_Donut_Env):
    def __init__(self, **kwargs):
        super().__init__(size=20, agent_start_pos=None, **kwargs)