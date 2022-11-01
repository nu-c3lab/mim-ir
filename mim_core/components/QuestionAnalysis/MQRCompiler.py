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
MQR Compiler

August 3, 2021
Authors: C3 Lab
"""

from typing import List
from mim_core.components.QuestionAnalysis.CompilerRules import CompilerRule
from mim_core.components.QuestionAnalysis.CompilerRules import SelSelIntfilNonBoolean, SelSelIntfilBoolean, \
                                                               SelSelProjIntfil, SelSelFilIntfil, FilFilIntfilBoolean, \
                                                               SelBooleanSearch, RemoveNameOfProjects

class MQRCompiler(object):
    """
    A class that will make use of ontological and semantic knowledge to detect problems in and improve MQR/plans.
    """

    def __init__(self,
                 high_level: bool,
                 custom_rules: List[CompilerRule] = None):
        self.high_level = high_level
        if custom_rules:
            self.rules = custom_rules
        else:
            self.load_rules()


    def load_rules(self) -> None:
        """
        Loads a predefined set of rules depending on whether or not
        they're meant to be applied to high or low level decompositions.
        :return: None
        """

        if self.high_level:
            # High Level Rules
            self.rules = [RemoveNameOfProjects()]
        else:
            # Low Level Rules
            self.rules = [SelSelIntfilNonBoolean(), SelSelIntfilBoolean(), SelSelProjIntfil(),
                          SelSelFilIntfil(), FilFilIntfilBoolean(), SelBooleanSearch()]
