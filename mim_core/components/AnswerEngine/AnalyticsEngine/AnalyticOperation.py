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
Analytic Operation

August 31, 2021
Authors: C3 Lab
"""

import pandas as pd
from abc import ABC
from mim_core.structs.Step import Step

class AnalyticOperation(ABC):
    """
    An abstract base class for building out operations to be carried out within the AnalyticsEngine.
    """

    def __init__(self):
        pass

    def execute(self, step: Step) -> pd.DataFrame:
        pass
