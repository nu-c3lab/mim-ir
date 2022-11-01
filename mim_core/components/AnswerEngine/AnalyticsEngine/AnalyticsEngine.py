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
Analytics Engine

March 28, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AggregateOperation import AggregateOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.ArithmeticOperation import ArithmeticOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.BooleanOperation import BooleanOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.ComparativeOperation import ComparativeOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.ComparisonOperation import ComparisonOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.DiscardOperation import DiscardOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.GroupOperation import GroupOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.IntersectionOperation import IntersectionOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.SortOperation import SortOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.SuperlativeOperation import SuperlativeOperation
from mim_core.components.AnswerEngine.AnalyticsEngine.UnionOperation import UnionOperation

class AnalyticsEngine(object):
    """
    A class that provides the functions required to carry out analysis operations, such as those specified by QDMR.
    """

    def __init__(self):

        self.aggregate_operation = AggregateOperation()
        self.arithmetic_operation = ArithmeticOperation()
        self.boolean_operation = BooleanOperation()
        self.comparative_operation = ComparativeOperation()
        self.comparison_operation = ComparisonOperation()
        self.discard_operation = DiscardOperation()
        self.group_operation = GroupOperation()
        self.intersection_operation = IntersectionOperation()
        self.sort_operation = SortOperation()
        self.superlative_operation = SuperlativeOperation()
        self.union_operation = UnionOperation()
        self.operations = {
            "aggregate": self.aggregate_operation.execute,
            "arithmetic": self.arithmetic_operation.execute,
            "boolean": self.boolean_operation.execute,
            "comparative": self.comparative_operation.execute,
            "comparison": self.comparison_operation.execute,
            "discard": self.discard_operation.execute,
            "group": self.group_operation.execute,
            "intersection": self.intersection_operation.execute,
            "sort": self.sort_operation.execute,
            "superlative": self.superlative_operation.execute,
            "union": self.union_operation.execute
        }

    def analyze_data(self,
                     step: Step) -> pd.DataFrame:
        """
        Performs the specified analysis operation on the given step of the plan.
        :param step: The step for which to carry out the analysis.
        :return: The results dataframe of the analysis.
        """
        return self.operations[step.operator_type](step)
