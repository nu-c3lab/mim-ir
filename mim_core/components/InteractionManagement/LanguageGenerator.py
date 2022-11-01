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
Language Generator

March 27, 2021
Authors: C3 Lab
"""

from abc import ABC
import spacy
from mim_core.structs.MQR import HierarchicalMQR
from datetime import datetime
import re

class LanguageGenerator(ABC):
    """
    An abstract class for generating natural language to display to the user.
    """

    def _get_pretty_date(self, ans):
        date_match = re.search(r'\d{4}\d*-\d{2}-\d{2}', ans)
        if date_match:
            return datetime.fromisoformat(date_match.group(0)).strftime('%B %d, %Y')
        else:
            return str(ans)

    def generate_response(self, mqr: HierarchicalMQR) -> str:
        pass
