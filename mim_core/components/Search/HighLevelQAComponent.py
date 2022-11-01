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
High Level Question Answering Component

September 9, 2021
Authors: C3 Lab
"""

from mim_core.components.Search.QuestionAnsweringComponent import QuestionAnsweringComponent

class HighLevelQAComponent(QuestionAnsweringComponent):
    """
    An abstract class / interface that specifies any class inheriting this is meant
    to answers questions from high level decompositions.
    """
    def __init__(self):
        super().__init__()
