from __future__ import annotations

import numpy as np
from gymnasium import spaces

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Gates, Lava, Fake_Lava, Floor, FloorCustom, WallCustom
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import PATTERNS, IDX_TO_COLOR

patterns = [
    'lines',
    'cross',
    'checkers',
    'square',
    'triangle'
]

gen = np.random.default_rng(seed=42)
wall_colors = gen.choice(100, (500,3))

def reject_nonmarked_rooms(env: MiniGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are not in the unique rooms
    """
    x, y = pos
    marked = x <= env.roomsize or y <= env.roomsize or x>= (env.width - env.roomsize - 1) or y >= (env.height - env.roomsize - 1)
    return (not marked)


def reject_nontarget_rooms(env: MiniGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are not in the unique rooms
    """
    x, y = pos
    xt, yt = env.goalpos
    target = x <= (xt + env.halfsize) and y <= (yt + env.halfsize) and x >= (xt - env.halfsize) and y >= (yt - env.halfsize)
    return (not target)


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
        self, roomsize=5,
        roomsv=3,
        roomsh=4,
        nlava=None,
        target_start=False,
        gates=True,
        neg=0,
        seed=8,
        max_steps: int | None = None,
        **kwargs
    ):
        self.roomsize = roomsize
        self.halfsize = int(roomsize/2) + (roomsize%2 > 0)
        self.roomsv = roomsv
        self.roomsh = roomsh
        self.nlava = nlava
        self.targetstart = target_start
        self.gates = gates
        self.neg = neg


        randgen = np.random.default_rng(seed=seed)
        self.marks = np.arange(25)
        randgen.shuffle(self.marks)
        # np.random.seed(seed)
        # self.marks = np.arange(25)
        # np.random.shuffle(self.marks)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * roomsh * roomsv * roomsize

        super().__init__(
            mission_space=mission_space,
            width=roomsh*(roomsize+1)+1,
            height=roomsv*(roomsize+1)+1,
            # Set this to True for maximum speed
            see_through_walls=False,
            max_steps=max_steps,
            **kwargs,
        )

        self.action_space = spaces.Discrete(4)


    @staticmethod
    def _gen_mission():
        return "avoid the real lava and get to the fake lava square"

    def _gen_grid(self, width, height, regenerate=True):
        if regenerate:
            assert width >= 17 and height >= 13

            # Create an empty grid
            self.grid = Grid(width, height)

            # Generate rooms
            for i in range(self.roomsh-2):
                for j in range(self.roomsv-2):
                    self.grid.wall_rect(
                                        (i+1)*(self.roomsize+1),
                                        (j+1)*(self.roomsize+1),
                                        self.roomsize+2,
                                        self.roomsize+2,
                                        # Gates
                                        )
            # for i in range(self.roomsh):
            #     for j in range(self.roomsv):
            #         self.grid.wall_rect(
            #                             i*(self.roomsize+1),
            #                             j*(self.roomsize+1),
            #                             self.roomsize+2,
            #                             self.roomsize+2,
            #                             # Gates
            #                             )

            # Generate the surrounding walls
            # self.grid.wall_rect(0, 0, width, height, WallCustom, wall_colors)
            self.grid.wall_rect(0, 0, width, height)
            self.grid.set(1,1,WallCustom(add=wall_colors[1]))

            # Generate gates
            if self.gates:
                for i in range(self.roomsh-2):
                    for j in range(self.roomsv-2):
                        self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)+self.halfsize-1, Gates())
                        self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)+self.halfsize, Gates())
                        self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)+self.halfsize+1, Gates())
                        self.grid.set((i+2)*(self.roomsize+1), (j+1)*(self.roomsize+1)+self.halfsize-1, Gates())
                        self.grid.set((i+2)*(self.roomsize+1), (j+1)*(self.roomsize+1)+self.halfsize, Gates())
                        self.grid.set((i+2)*(self.roomsize+1), (j+1)*(self.roomsize+1)+self.halfsize+1, Gates())
                        self.grid.set((i+1)*(self.roomsize+1)+self.halfsize-1, (j+1)*(self.roomsize+1), Gates())
                        self.grid.set((i+1)*(self.roomsize+1)+self.halfsize, (j+1)*(self.roomsize+1), Gates())
                        self.grid.set((i+1)*(self.roomsize+1)+self.halfsize+1, (j+1)*(self.roomsize+1), Gates())
                        self.grid.set((i+1)*(self.roomsize+1)+self.halfsize-1, (j+2)*(self.roomsize+1), Gates())
                        self.grid.set((i+1)*(self.roomsize+1)+self.halfsize, (j+2)*(self.roomsize+1), Gates())
                        self.grid.set((i+1)*(self.roomsize+1)+self.halfsize+1, (j+2)*(self.roomsize+1), Gates())
            # for i in range(self.roomsh-1):
            #     self.grid.set((i+1)*(self.roomsize+1), height-self.halfsize, Gates())
            #     self.grid.set((i+1)*(self.roomsize+1), height-self.halfsize-1, Gates())
            #     self.grid.set((i+1)*(self.roomsize+1), height-self.halfsize-2, Gates())
            #     for j in range(self.roomsv-1):
            #         self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)-self.halfsize-1, Gates())
            #         self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)-self.halfsize, Gates())
            #         self.grid.set((i+1)*(self.roomsize+1), (j+1)*(self.roomsize+1)-self.halfsize+1, Gates())
            #         self.grid.set((i+1)*(self.roomsize+1)-self.halfsize-1, (j+1)*(self.roomsize+1), Gates())
            #         self.grid.set((i+1)*(self.roomsize+1)-self.halfsize, (j+1)*(self.roomsize+1), Gates())
            #         self.grid.set((i+1)*(self.roomsize+1)-self.halfsize+1, (j+1)*(self.roomsize+1), Gates())
            # for j in range(self.roomsv-1):
            #     self.grid.set(width-self.halfsize, (j+1)*(self.roomsize+1), Gates())
            #     self.grid.set(width-self.halfsize-1, (j+1)*(self.roomsize+1), Gates())
            #     self.grid.set(width-self.halfsize-2, (j+1)*(self.roomsize+1), Gates())

            # Place lava
            if self.roomsv<5:
                self.goalpos = None
                for i in range(self.roomsh-2):
                    for j in range(self.roomsv-2):
                        pos = ((i+1)*(self.roomsize+1)+self.halfsize, (j+1)*(self.roomsize+1)+self.halfsize)
                        if not self.goalpos:
                            obj = Fake_Lava()
                            self.goalpos = pos
                        else:
                            obj = Lava()
                        self.put_obj(obj, *pos)

            # Place the agent
            if self.targetstart:
                self.agent_pos = (-1, -1)
                pos = self.place_obj(None, reject_fn=reject_nontarget_rooms)
                self.agent_pos = pos
                self.agent_dir = self._rand_int(0, 4)

            else:
                self.agent_pos = (-1, -1)
                pos = self.place_obj(None, reject_fn=reject_nonmarked_rooms)
                self.agent_pos = pos
                self.agent_dir = self._rand_int(0, 4)

            # Generate marks
            for y in range(1,5):
                self.put_obj(Floor('blue'), 1, y)
            for y in range(1,6):
                self.put_obj(Floor('blue'), 2, y)
            for y in range(1,7):
                self.put_obj(Floor('blue'), 3, y)
            for y in range(self.height-5,self.height-2):
                self.put_obj(Floor('yellow'), 2, y)
            for y in range(self.height-10,self.height-6):
                self.put_obj(Floor('yellow'), 3, y)
            for x in range(self.width-self.roomsize-6,self.width-self.roomsize-1):
                self.put_obj(Floor('yellow'), x, 1)
            for x in range(self.width-self.roomsize-6,self.width-self.roomsize-1):
                self.put_obj(Floor('yellow'), x, 2)
            for x in range(self.width-self.roomsize-6,self.width-self.roomsize-1):
                self.put_obj(Floor('yellow'), x, 3)
            for x in range(self.width-4,self.width-1):
                self.put_obj(Floor('red'), x, self.roomsize+2)
            for x in range(self.width-4,self.width-1):
                self.put_obj(Floor('red'), x, self.roomsize+3)
            for x in range(self.width-self.roomsize-10,self.width-self.roomsize-5):
                self.put_obj(Floor('red'), x, self.height-3)
            for x in range(self.width-self.roomsize-10,self.width-self.roomsize-5):
                self.put_obj(Floor('red'), x, self.height-4)
            for x in range(self.width-self.roomsize-10,self.width-self.roomsize-5):
                self.put_obj(Floor('red'), x, self.height-5)
            for x in range(self.width-5,self.width-2):
                for y in range(self.height-5,self.height-2):
                    self.put_obj(Floor('blue'), x, y)
                
            # n=0
            # for i in range(self.roomsh):
            #     for j in range(self.roomsv):
            #         n_shape = self.marks[n]//5
            #         n_color = self.marks[n]%5
            #         if i==0:
            #             n+=1
            #             self.place_shape(
            #                 PATTERNS[patterns[n_shape]][:self.roomsize-2,:self.roomsize],
            #                 (i*(self.roomsize+1)+1, j*(self.roomsize+1)+1),
            #                 IDX_TO_COLOR[n_color]
            #             )
            #         elif i==self.roomsh-1:
            #             n+=1
            #             self.place_shape(
            #                 np.vstack((np.zeros((2,self.roomsize)),
            #                            PATTERNS[patterns[n_shape]][:self.roomsize-2,:self.roomsize])
            #                            ),
            #                 (i*(self.roomsize+1)+1, j*(self.roomsize+1)+1),
            #                 IDX_TO_COLOR[n_color]
            #                 )
            #         elif j==0:
            #             n+=1
            #             self.place_shape(
            #                 PATTERNS[patterns[n_shape]][:self.roomsize-2,:self.roomsize].T,
            #                 (i*(self.roomsize+1)+1, j*(self.roomsize+1)+1),
            #                 IDX_TO_COLOR[n_color]
            #                 )
            #         elif j==self.roomsv-1:
            #             n+=1
            #             self.place_shape(
            #                 np.hstack((np.zeros((self.roomsize,2)),
            #                            PATTERNS[patterns[n_shape]][:self.roomsize-2,:self.roomsize].T)),
            #                 (i*(self.roomsize+1)+1, j*(self.roomsize+1)+1),
            #                 IDX_TO_COLOR[n_color]
            #                 )

            # Colorize the rest of the floor
            # n = 0
            # for i in range(self.width):
            #     for j in range(self.height):
            #         if self.grid.get(i,j) is not None:
            #             continue
            #         else:
            #             self.grid.set(i, j, FloorCustom(floor_colors[n]))
            #             n+=1

                        

            # Place the agent in the top-left corner
            # self.place_agent()

            self.mission = (
                "avoid the real lava and get to the fake lava square"
            )
            self.regenerate=False
        else:

            # Place the agent
            if self.targetstart:
                self.agent_pos = (-1, -1)
                pos = self.place_obj(None, reject_fn=reject_nontarget_rooms)
                self.agent_pos = pos
                self.agent_dir = self._rand_int(0, 4)

            else:
                self.agent_pos = (-1, -1)
                pos = self.place_obj(None, reject_fn=reject_nonmarked_rooms)
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
        Place a shape with upper left corner at (x,y)
        """
            
        shapecoords = np.transpose(np.nonzero(shape))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])