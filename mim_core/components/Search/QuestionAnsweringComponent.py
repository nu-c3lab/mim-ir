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
Question Answering Component

September 9, 2021
Authors: C3 Lab
"""

import pandas as pd
from abc import ABC
from typing import Dict

from mim_core.structs.Step import Step

class QuestionAnsweringComponent(ABC):
    """
    An abstract class that specifies an interface for answering a question using
    an associated source of knowledge for entities/evidence to the question at hand.
    """

    def __init__(self):
        self.operations = {
            "select": self.select,
            "project": self.project,
            "filter": self.filter,
            "aggregate": self.aggregate,
            "arithmetic": self.arithmetic,
            "group": self.group,
            "superlative": self.superlative,
            "union": self.union,
            "comparative": self.comparative,
            "comparison": self.comparison,
            "boolean": self.boolean,
            "intersection": self.intersection,
            "discard": self.discard,
            "sort": self.sort
        }

    def answer(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def select(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def project(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def filter(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def aggregate(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def arithmetic(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def group(self, step: Step, timing: Dict) -> pd.DataFrame:
        pass

    def superlative(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def union(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def comparative(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def comparison(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def boolean(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def intersection(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def discard(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass

    def sort(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        pass
