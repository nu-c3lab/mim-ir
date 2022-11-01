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
Union Operation

September 12, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new

class UnionOperation(AnalyticOperation):
    """
    A class for carrying out a union operation.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Performs a union on the answer values in the two provided dataframes.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the union analysis operation is as follows:
            {
                "args": ["<df_ref_1>", "<df_ref_2>"],
                "prior_step_answers": {
                        "<df_ref_1>": <Pandas DataFrame>,
                        "<df_ref_2>": <Pandas DataFrame>
                }
            }
        :return: A Pandas DataFrame containing the answer.
        """
        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Pull out the data frames to operate on
            df1_str = step.operator_args[0]
            df1 = prior_step_answers[df1_str]

            df2_str = step.operator_args[1]
            df2 = prior_step_answers[df2_str]

            # Union the two data frames, producing a dataframe containing all answers from both data frames
            df = pd.concat([df1, df2], ignore_index=True).drop_duplicates().reset_index(drop=True)

            # Reset the id's of the answers
            df['id'] = df.index
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0

            return df
        except Exception as e:
            print("Unable to perform the Union operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()
