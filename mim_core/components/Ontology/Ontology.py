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
Ontology

March 28, 2021
Authors: C3 Lab
"""

from abc import ABC
from typing import List

class Ontology(ABC):
    def __init__(self):
        pass

    def extract_canonical_entities(self, text: str) -> dict:
        pass

    def extract_canonical_relationship(self, text: str, operator_type: str) -> str:
        pass

    def extract_canonical_entity_class(self, text: str) -> str:
        pass

    def canonicalize_entity(self, entity: str) -> str:
        pass

    def is_expected_type(self, entity: str, expected_types: List[str]) -> bool:
        pass
