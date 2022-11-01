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
Document

March 27, 2021
Authors: C3 Lab
"""

class Document(object):
    """
    A class for storing and interacting with text documents.
    """

    def __init__(self,
                 title: str,
                 content: str,
                 score: float = 1.0,
                 index: int = 0):
        self.title = title
        self.content = content
        self.score = score
        self.index = index

    def __repr__(self):
        return 'Document({})'.format(self.to_json())

    def __str__(self):
        return 'Document(Title: {} | Content: {} | Score: {} | Index {})'.format(self.title, self.content, self.score, self.index)

    def to_json(self):
        return {
            "title": self.title,
            "content": self.content,
            "score": float(self.score),
            "index": self.index
        }

class Passage(Document):
    """
    A class for storing and interacting with text passages. These passages a derived from Document objects.
    """

    def __init__(self,
                 title: str,
                 content: str,
                 document_title: str,
                 score: float = 1.0,
                 index: int = 0):
        super().__init__(title, content, score, index)
        self.document_title = document_title

    def __repr__(self):
        return 'Passage({})'.format(self.to_json())

    def __str__(self):
        return 'Passage(Title: {} | Content: {} | Document Title: {} | Score: {} | Index {})'.format(self.title, self.content, self.document_title, self.score, self.index)

    def to_json(self):
        return {
            "title": self.title,
            "content": self.content,
            "score": float(self.score),
            "index": self.index,
            "document_title": self.document_title
        }