'''
This file is part of Mim.
Mim is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.
Mim is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with Mim.
If not, see <https://www.gnu.org/licenses/>.
'''
"""
Scoring Function

March 28, 2021
Authors: C3 Lab
"""

from abc import ABC
import pandas as pd
from mim_core.structs.Step import Step

class ScoringFunction(ABC):
    """
    An abstract class for scoring answers/evidence found by the search components.
    """

    def __init__(self):
        pass

    def evaluate_results(self,
                         step: Step) -> pd.DataFrame:
        pass
