from __future__ import annotations

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal, Goal_invisible, Floor, Wall
from minigrid.minigrid_env import MiniGridEnv
import numpy as np

from gymnasium import spaces


class FourRoomsEnv(MiniGridEnv):

    """
    ## Description

    Classic four room reinforcement learning environment. The agent must
    navigate in a maze composed of four rooms interconnected by 4 gaps in the
    walls. To obtain a reward, the agent must reach the green goal square. Both
    the agent and the goal square are randomly placed in any of the four rooms.

    ## Mission Space

    "reach the goal"

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
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-FourRooms-v0`

    """

    def __init__(self, open_all_paths: bool, subroom_size=8, agent_start_dir: int | None = None, agent_start_pos: tuple[int, int] | None = None, goal_pos=None, max_steps=100, door_poss=None, room_marks=False, visible=True, **kwargs):

        self._door_default_poss = door_poss
        self.room_marks = room_marks
        self.vis = visible

        self.agent_start_dir = self._rand_int(0, 4) if agent_start_dir is None else agent_start_dir
        self.agent_start_pos = agent_start_pos
        self.goal_pos = goal_pos

        self.open_all_paths = open_all_paths
        self.size = subroom_size * 2 + 3
        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            width=self.size,
            height=self.size,
            max_steps=max_steps,
            **kwargs,
        )
        self.action_space = spaces.Discrete(4)

    @staticmethod
    def _gen_mission():
        return "Reach the goal or wander around!"
    
    def _is_pos_valid(self, pos: tuple[int, int]) -> bool:
        """Check if a position is valid (not a wall)"""
        x, y = pos
        obj = self.grid.get(x, y)
       
        # Valid: empty cells (None) or overlappable objects (Floor)
        # Invalid: walls (can_overlap() returns False)
        is_valid = obj is None or obj.can_overlap()
        if not is_valid:
            print(f"Position {pos} is invalid (object: {obj})")

        return is_valid

    def _validate_and_set_pos(self, pos: tuple[int, int]) -> tuple[int, int] | None:
        new_pos = pos
        if not self._is_pos_valid(pos):
            print(f"Warning: {pos} is invalid. Randomizing position.")
            new_pos = None
        return new_pos

    def _gen_grid(self, width, height, regenerate=True):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)


        room_w = width // 2
        room_h = height // 2


        # Mark the rooms wtih colored tiles
        if self.room_marks:
            room1 = (1,1)
            room2 = (1,room_h+1)
            room3 = (room_w+1,1)
            room4 = (room_w+1,room_h+1)
            self.place_shape('horiz',   room1,'red')
            self.place_shape('vert',    room4,'red')
            self.place_shape('fwdslash',room3,'red')
            self.place_shape('bckslash',room2,'red')


        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * room_w
                yT = j * room_h
                xR = xL + room_w
                yB = yT + room_h

                # Vertical wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, room_h)

                    if self._door_default_poss:
                        pos = (xR, yT + 1 + self._door_default_poss[j])
                    else:
                        pos = (xR, self._rand_int(yT + 1, yB))
                        
                    self.grid.set(*pos, None)

                # Horizontal wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, room_w)

                    if self._door_default_poss:
                        pos = (xL +1 + self._door_default_poss[i+2], yB)
                    else:
                        pos = (self._rand_int(xL + 1, xR), yB)

                    # Skip door between rooms 2 (top right) and 4 (bottom right) if applicable
                    if self.open_all_paths or not (i == 1 and j == 0):
                        self.grid.set(*pos, None)

        # Validate positions now that grid is generated
        if self.agent_start_pos is not None:
            self.agent_start_pos = self._validate_and_set_pos(self.agent_start_pos)
        if self.goal_pos is not None:
            self.goal_pos = self._validate_and_set_pos(self.goal_pos)

        # Init the player start position and orientation
        if self.agent_start_pos is not None:
            i, j = self.agent_start_pos
            self.agent_pos = self.agent_start_pos
            self.grid.set(i=i, j=j, v=None)
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        if self.goal_pos is not None:
            if self.vis:
                goal = Goal()
            else:
                goal = Goal_invisible()

            self.put_obj(goal, *self.goal_pos)
            goal.init_pos = goal.cur_pos = self.goal_pos
        
        # Else randomly place goal: self.place_obj(Goal())


    def place_shape(self,shape,pos,color):
        """
        Place a 8x8 shape with lower left corner at (x,y)
        """
        shapegrid={
            'vert':np.array(
                [[1,0,1,0,1,0,1,0],
                 [1,0,1,0,1,0,1,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0],
                 [1,0,1,0,1,0,1,0],
                 [1,0,1,0,1,0,1,0]]),
            'horiz':np.array(
                [[0,0,1,1,1,1,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,1,1,1,1,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,1,1,1,1,0,0],
                 [0,0,0,0,0,0,0,0],
                 [0,0,1,1,1,1,0,0],
                 [0,0,0,0,0,0,0,0]]),
            'fwdslash':np.array(
                [[1,0,0,0,1,0,0,0],
                 [0,1,0,1,0,0,0,1],
                 [0,0,1,0,0,0,1,0],
                 [0,1,0,1,0,1,0,0],
                 [1,0,0,0,1,0,0,0],
                 [0,0,0,1,0,1,0,1],
                 [0,0,1,0,0,0,1,0],
                 [0,1,0,0,0,1,0,1]]),
            'bckslash':np.array(
                [[0,0,0,1,0,0,0,1],
                 [1,0,0,0,1,0,0,0],
                 [0,1,0,0,0,1,0,0],
                 [0,0,1,0,0,0,1,0],
                 [0,0,0,1,0,0,0,1],
                 [1,0,0,0,1,0,0,0],
                 [0,1,0,0,0,1,0,0],
                 [0,0,1,0,0,0,1,0]])
            }
            
        shapecoords = np.transpose(np.nonzero(shapegrid[shape]))+np.array(pos,dtype='int32')

        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])


class FourRoomsObjs(FourRoomsEnv):
    """
    Four rooms environment with colored shapes in each room.

    - Each room is 8x8
    - No extrinsic goal
    - Agent cannot see through walls
    """

    def __init__(self, subroom_size=8, agent_start_pos=None, goal_pos=None, max_steps=None,
                 door_poss=None, room_marks=False, visible=True, open_all_paths=True, **kwargs):
        
        # Remove see_through_walls from kwargs to ensure it defaults to False
        kwargs.pop('see_through_walls', None)

        if max_steps is None:
            max_steps = 4 * subroom_size * subroom_size

        super().__init__(
            agent_start_pos=agent_start_pos,
            goal_pos=goal_pos,
            subroom_size=subroom_size,
            max_steps=max_steps,
            door_poss=door_poss,
            room_marks=room_marks,
            visible=visible,
            open_all_paths=open_all_paths,
            **kwargs
        )

    def _gen_grid(self, width, height, regenerate=True):
        super()._gen_grid(width, height, regenerate)

        # Add shapes to each room
        room_w = width // 2
        room_h = height // 2

        for j in range(2):
            for i in range(2):
                xL = i * room_w
                yT = j * room_h
                self._place_room_shapes(i, j, xL, yT)


    def _place_room_shapes(self, room_i, room_j, xL, yT):
        """Place shapes for a specific room"""
        # Define shapes per room: (shape, offset_x, offset_y, color, scale)
        room_configs = {
            (0, 0): [  # Room 1: top-left
                ('triangleUleft', 1, 1, 'red', 1.0),
                ('plus', 6, 4, 'blue', 1.0),
                ('L', 2, 6, 'yellow', 1.0),
            ],
            (1, 0): [  # Room 2: top-right
                ('plus', 5, 1, 'red', 1.0),
                ('x', 1, 3, 'yellow', 1.0),
            ],
            (0, 1): [  # Room 3: bottom-left
                ('triangleUright', 5, 1, 'yellow', 1.0),
                ('plus', 5, 6, 'yellow', 1.0),
                ('plus', 1, 4, 'red', 1.0),
            ],
            (1, 1): [  # Room 4: bottom-right
                ('x', 3, 1, 'red', 1.0),    
                ('triangleLright', 5, 5, 'blue', 1.0),           
            ],
        }

        shapes = room_configs.get((room_i, room_j), [])
        for shape_name, offset_x, offset_y, color, scale in shapes:
            pos = (xL + offset_x, yT + offset_y)
            self.place_shape(shape_name, pos, color, scale)

    def place_shape(self, shape, pos, color, scale=1.0):
        """Place a scaled shape at position"""
        # Base 4x4 shape definitions
        base_shapes = {
            'triangleLright': np.array([
                [0,0,0,1],
                [0,0,1,1],
                [0,1,1,1],
                [1,1,1,1]]),
            'triangleUleft': np.array([
                [1,0,0,0],
                [1,1,0,0],
                [1,1,1,0],
                [1,1,1,1]]),
            'triangleUright': np.array([
                [1,1,1,1],
                [0,1,1,1],
                [0,0,1,1],
                [0,0,0,1]]),
            'plus': np.array([
                [0,1,0],
                [1,1,1],
                [0,1,0]]),
            'x':np.array(
                [[1,1,0,1,1],
                 [0,1,1,1,0],
                 [0,1,1,1,0],
                 [1,1,0,1,1],]),
            'L': np.array(
                [[0,1],
                 [1,1],]),
        }

        # Get and scale shape
        base = base_shapes.get(shape)
        if base is None:
            return  # Unknown shape

        shaped = self._scale_shape(base, scale)

        # Place Floor objects at shape coordinates
        shapecoords = np.transpose(np.nonzero(shaped)) + np.array(pos, dtype='int32')
        for coord in shapecoords:
            self.put_obj(Floor(color), coord[0], coord[1])

    def _scale_shape(self, base_shape, scale_factor):
        """Scale shape by factor (0.5, 1.0, 2.0)"""
        assert scale_factor in [0.5, 1.0, 2.0], f"Unsupported scale: {scale_factor}"

        if scale_factor == 1.0:
            return base_shape
        elif scale_factor < 1.0:
            # Downscale by selecting every nth element
            step = int(1.0 / scale_factor)
            return base_shape[::step, ::step]
        else:
            # Upscale by repeating elements
            factor = int(scale_factor)
            return np.repeat(np.repeat(base_shape, factor, axis=0), factor, axis=1)
