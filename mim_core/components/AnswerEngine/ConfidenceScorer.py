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
Confidence Scorer

October 9, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.ScoringFunction import ScoringFunction
from mim_core.exceptions import MissingColumnError
from mim_core.utils.result_utils import is_arg_reference

class ConfidenceScorer(ScoringFunction):
    """
    A class that scores results based on the confidence score associated with an answer.
    """

    def __init__(self,
                 **kwargs):
        super().__init__()

    def evaluate_results(self,
                        step: Step) -> pd.DataFrame:
        """
        Adds a column to the dataframe called 'ConfidenceScorer' that contains the
        weighted, cumulative confidence score for this answer. It is cumulative in the
        sense that the score is multiplied with the confidence score of corresponding
        answers/evidence from past steps.
        :param step: The step with the dataframe of results/answers to score.
        :return: The updated dataframe with the new score column.
        """

        # Make sure the 'confidence' column is present for scoring
        result = step.result
        if 'confidence' not in result:
            # Do any cleanup
            result[str(self.__class__.__name__)] = 0.0

            # Raise the exception
            raise MissingColumnError('confidence')

        if len(result) > 0:
            final_confidences = []
            for idx, row in step.result.iterrows():
                cumulative_confidence = row['confidence']

                # For each of the previous step references in the columns of the step.result dataframe:
                for col in step.result:
                    ref_id = is_arg_reference(col)
                    if ref_id:
                        # prev_step_id = "@@"+str(ref_id)+"@@"

                        # Get the previous dataframe
                        prev_step = [p for p in step.parent_steps if p.reference_id == int(ref_id)][0]

                        # Get the ConfidenceScorer value from the previous dataframe
                        prev_answer_id = row["@@"+str(ref_id)+"@@"]
                        prev_row = prev_step.result[prev_step.result.id == prev_answer_id].iloc[0]

                        cumulative_confidence = cumulative_confidence * prev_row[str(self.__class__.__name__)]

                # Store the final confidence value for this answer
                final_confidences.append(cumulative_confidence)

            # Set the ConfidenceScorer column to updated values
            result[str(self.__class__.__name__)] = final_confidences
        else:
            # Set default values for the ConfidenceScorer column when there are no answers to evaluate
            result[str(self.__class__.__name__)] = 0.0
        return result
