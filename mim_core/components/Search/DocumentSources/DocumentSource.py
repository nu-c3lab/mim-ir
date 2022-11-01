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
Document Source

March 28, 2021
Authors: C3 Lab
"""

from abc import ABC
from typing import List
from mim_core.structs.Document import Document, Passage

class DocumentSource(ABC):
    """
    An abstract base class for building out new sources of documents, particularly for neural questioning answering.
    """

    def __init__(self):
        self.name = None

    def get_documents(self,
                      query: str,
                      q_id: str = None,
                      keywords: List[str] = None) -> List[Document]:
        pass

    def get_passages(self,
                     question: str,
                     documents: List[Document] = None,
                     q_id: str = None,
                     keywords: List[str] = None) -> List[Passage]:
        pass
