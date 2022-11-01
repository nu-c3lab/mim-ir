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
Neural Search Model

April 10, 2021
Authors: C3 Lab
"""

from abc import ABC
from typing import Tuple
from mim_core.structs.Document import Document

class NeuralQAModel(ABC):
    """
    A class that provides an interface for neural question answering models.
    """
    def __init__(self,
                 tokenizer=None,
                 model=None):
        self.tokenizer = tokenizer
        self.model = model

    def answer_question(self,
                        question: str,
                        doc: Document) -> Tuple[str, float]:
        pass

    def load_model(self):
        pass
