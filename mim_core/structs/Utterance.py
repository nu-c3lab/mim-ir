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
Utterance

March 27, 2021
Authors: C3 Lab
"""

class Utterance(object):
    """
    A class for representing the utterances coming in from the user.
    """

    def __init__(self, text: str, q_id: str = None, context: str = None):
        self.text = text
        self.q_id = q_id
        self.context = context

    def __repr__(self):
        return 'Utterance({})'.format(self.to_json())

    def to_json(self):
        return {
            "text": self.text,
            "q_id": self.q_id,
            "context": self.context
        }
