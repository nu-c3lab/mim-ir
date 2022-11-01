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
Superlative Operation

September 12, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnhandledSubOperationTypeError

class SuperlativeOperation(AnalyticOperation):
    """
    A class for carrying out a superlative operation.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Finds the superlative in the 'answer' column of df1 based on the values in the <df_ref_2> column of df1.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the count analysis operation is as follows:
            {
                "args": ["<superlative_type_str>", "<df_ref_1>", "<df_ref_2>"],
                "prior_step_answers": {
                        "@@<df_ref_1>@@": <Pandas DataFrame>,
                        "@@<df_ref_2>@@": <Pandas DataFrame>
                }
            }
        :return: A Pandas DataFrame containing the result in the "answer" column.
        """

        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Pull out the data frames to operate on
            operation_str = step.operator_args[0]

            # Get the dataframes and references to prior steps
            df1_str = step.operator_args[1]
            df1 = prior_step_answers[df1_str]

            df2_str = step.operator_args[2]
            df2 = prior_step_answers[df2_str]

            if operation_str == 'max':
                df = df2[df2.answer == df2.answer.max()]
            elif operation_str == 'min':
                df = df2[df2.answer == df2.answer.min()]
            else:
                raise UnhandledSubOperationTypeError(operation_str)

            # Correlate the df1 id's in df2 denoted by df1_str with the actual answers from df1
            df = df.drop(columns=["answer"])
            df.rename(columns={"id": df2_str}, inplace=True)
            df1_cpy = df1.rename(columns={'id': df1_str})
            df = df.merge(df1_cpy, on=df1_str, how="inner")

            # Clean up the data frame
            df = df.reset_index()
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', df1_str, df2_str])
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0

            return df
        except Exception as e:
            print("Unable to perform the Superlative operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()
