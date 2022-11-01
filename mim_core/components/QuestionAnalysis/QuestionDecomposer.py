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
Question Decomposer

March 28, 2021
Authors: C3 Lab
"""

from abc import ABC
from typing import List
from mim_core.structs.Step import Step
from mim_core.structs.Utterance import Utterance

class QuestionDecomposer(ABC):
    """
    An abstract class from which to build neural question decomposition models.
    """

    def __init__(self,
                 model = None,
                 tokenizer = None):
        self.model = model
        self.tokenizer = tokenizer

    def decompose_question(self,
                           utterance: Utterance,
                           num_generations: int=1) -> List[List[Step]]:
        pass
