from __future__ import annotations

import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Gates, Lava, Fake_Lava, Floor
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import PATTERNS, IDX_TO_COLOR

patterns = [
    'lines',
    'cross',
    'checkers',
    'square',
    'triangle'
]

def reject_nonmarked_rooms(env: MiniGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are not in the unique rooms
    """
    x, y = pos
    marked = x <= env.roomsize or y <= env.roomsize or x>= (env.width - env.roomsize - 1) or y >= (env.height - env.roomsize - 1)
    print (pos)
    print (marked)
    return (not marked)


class FakeLavaEnv(MiniGridEnv):

    """
    ## Description

    The agent has to reach the green goal square at the opposite corner of the
    room, and must pass through a narrow gap in a vertical strip of deadly lava.
    Touching the lava terminate the episode with a zero reward. This environment
    is useful for studying safety and safe exploration.

    ## Mission Space

    Depending on the `obstacle_type` parameter:
    - `Lava`: "avoid the lava and get to the green goal square"
    - otherwise: "find the opening and get to the green goal square"

    ## Action Space

    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. The agent falls into lava.
    3. Timeout (see `max_steps`).

    ## Registered Configurations

    S: size of map SxS.

    - `MiniGrid-LavaGapS5-v0`
    - `MiniGrid-LavaGapS6-v0`
    - `MiniGrid-LavaGapS7-v0`

    """

    def __init__(
        self, roomsize=5, roomsv=3, roomsh=4, lava=4, seed=42, max_steps: int | None = None, **kwargs
    ):
        self.roomsize = roomsize
        self.halfsize = int(roomsize/2) + (roomsize%2 > 0)
        self.roomsv = roomsv
        self.roomsh = roomsh
        self.lava = lava

        np.random.seed(seed)
        self.marks = np.arange(25)
        np.random.shuffle(self.marks)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = roomsh * roomsv * roomsize

        super().__init__(
            mission_space=mission_space,
            width=roomsh*(roomsize+1)+1,
            height=roomsv*(roomsize+1)+1,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "avoid the real lava and get to the fake lava square"

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate rooms
        for i in range(self.roomsh):
            for j in range(self.roomsv):
                self.grid.wall_rect(i*(self.roomsize+1), j*(self.roomsize+1), self.roomsize+2, self.roomsize+2)

        # Generate gates
        for i in range(self.roomsh-1):
            self.grid.set((i+1)*(self.roomsize+1), height-self.halfsize-1, Gates())
            for j in range(self.roomsv-1):
                self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)-self.halfsize, Gates())
                self.grid.set((i+1)*(self.roomsize+1)-self.halfsize, (j+1)*(self.roomsize+1), Gates())
        for j in range(self.roomsv-1):
            self.grid.set(width-self.halfsize-1, (j+1)*(self.roomsize+1), Gates())

        # Place agent
        self.agent_pos = (-1, -1)
        pos = self.place_obj(None, reject_fn=reject_nonmarked_rooms)
        self.agent_pos = pos
        self.agent_dir = self._rand_int(0, 4)

        # Generate marks
        n=0
        for i in range(self.roomsh):
            for j in range(self.roomsv):
                if i==0 or i==self.roomsh-1 or j==0 or j==self.roomsv-1:
                    n_shape = self.marks[n]//5
                    n_color = self.marks[n]%5
                    n+=1
                    self.place_shape(
                        PATTERNS[patterns[n_shape]][:self.roomsize,:self.roomsize],
                        (i*(self.roomsize+1)+1, j*(self.roomsize+1)+1),
                        IDX_TO_COLOR[n_color]
                    )

        # Place lava
        if self.roomsv<5:
            goal_put = False
            for i in range(self.roomsh):
                for j in range(self.roomsv):
                    if i!=0 and i!=self.roomsh-1 and j!=0 and j!=self.roomsv-1:
                        if not goal_put:
                            obj = Fake_Lava()
                            goal_put = True
                        else:
                            obj = Lava()
                        self.put_obj(obj, i*(self.roomsize+1)+self.halfsize, j*(self.roomsize+1)+self.halfsize)
                    

        # Place the agent in the top-left corner
        # self.place_agent()

        self.mission = (
            "avoid the real lava and get to the fake lava square"
        )
    def place_shape(self,shape,pos,color):
        """
        Place a shape with upper left corner at (x,y)
        """
            
        shapecoords = np.transpose(np.nonzero(shape))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])