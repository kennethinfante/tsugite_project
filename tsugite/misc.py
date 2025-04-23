import numpy as np
from typing import List, Optional, Union, Any

class FixedSides:
    def __init__(self, pjoint: Any, side_str: Optional[str] = None, fs: Optional[List[List['FixedSide']]] = None):
        self.pjoint: Any = pjoint
        self.sides: List[List['FixedSide']]
        if side_str is not None:
            self.sides_from_string(side_str)
        elif fs is not None:
            self.sides = fs
        else:
            self.sides = [[FixedSide(2, 0)], [FixedSide(2, 1)]]
        self.update_unblocked()

    def sides_from_string(self, side_str: str) -> None:
        self.sides = []
        for tim_fss in side_str.split(":"):
            temp: List['FixedSide'] = []
            for tim_fs in tim_fss.split("."):
                axdir = tim_fs.split(",")
                ax = int(float(axdir[0]))
                dir = int(float(axdir[1]))
                temp.append(FixedSide(ax, dir))
            self.sides.append(temp)

    def update_unblocked(self) -> None:
        # List unblocked POSITIONS
        self.unblocked: List['FixedSide'] = []
        for ax in range(3):
            for dir in range(2):
                blocked = False
                if self.sides is not None:
                    for sides in self.sides:
                        for side in sides:
                            if [side.ax, side.dir] == [ax, dir]:
                                blocked = True
                                break
                if not blocked:
                    self.unblocked.append(FixedSide(ax, dir))

        # List unblocked ORIENTATIONS
        self.pjoint.rot = True
        if self.sides is not None:
            for sides in self.sides:
                # if one or more component axes are aligned with the sliding axes (sax), rotation cannot be performed
                if sides[0].ax == self.pjoint.sax:
                    self.pjoint.rot = False
                    break

class FixedSide:
    def __init__(self, ax: int, dir: int):
        self.ax: int = ax
        self.dir: int = dir

    @staticmethod
    def depth(l: Union[List, Any]) -> int:
        if isinstance(l, list):
            return 1 + max(FixedSide.depth(item) for item in l) if l else 1
        else:
            return 0

    def unique(self, other_sides: Union[List['FixedSide'], List[List['FixedSide']]]) -> bool:
        unique = True
        if FixedSide.depth(other_sides) == 1:
            for side in other_sides:
                if self.ax == side.ax and self.dir == side.dir:
                    unique = False
                    break
        elif FixedSide.depth(other_sides) == 2:
            for sides in other_sides:
                for side in sides:
                    if self.ax == side.ax and self.dir == side.dir:
                        unique = False
                        break
        return unique
