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
Source Scorer

March 29, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.ScoringFunction import ScoringFunction
from mim_core.exceptions import MissingColumnError

class SourceScorer(ScoringFunction):
    """
    A class that scores results based on their source.
    """

    def __init__(self,
                 **kwargs):
        super().__init__()
        self.scoring_dict = kwargs.get('scoring_dict', {})

    def evaluate_results(self,
                        step: Step) -> pd.DataFrame:
        """
        Adds a column to the dataframe containing the weighted source score.
        :param step: The step with the dataframe of results to score.
        :return: The updated dataframe with the new score column.
        """
        # Make sure the 'source' column is present for scoring
        result = step.result
        if 'source' not in result:
            # Do any cleanup
            result[str(self.__class__.__name__)] = 0.0

            # Raise the exception
            raise MissingColumnError('source')

        if len(result) > 0:
            result[str(self.__class__.__name__)] = result.apply(lambda row : self.source_score(row['source']), axis=1)
        else:
            result[str(self.__class__.__name__)] = 0.0
        return result

    def source_score(self,
                     source_name: str) -> float:
        """
        Provides a score for each source.
        :param source_name: The name of source (e.g. CypherQueryDispatcher, SQLQueryDispatcher).
        :return: The final, weighted score.
        """
        if source_name in self.scoring_dict:
            return self.scoring_dict[source_name]
        else:
            return 1.0
