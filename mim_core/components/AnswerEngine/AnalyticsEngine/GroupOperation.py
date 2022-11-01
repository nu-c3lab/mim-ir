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
Group Operation

September 12, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnhandledSubOperationTypeError

class GroupOperation(AnalyticOperation):
    """
    A class for carrying out a group operation.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Groups answers in df1 by the values in the column "<df_ref_2>" of df1 and performs the required operation.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the count analysis operation is as follows:
            {
                "args": ["<group_type_str>", "<df_ref_1>", "<df_ref_2>"],
                "prior_step_answers": {
                        "@@<df_ref_1>@@": <Pandas DataFrame>,
                        "@@<df_ref_2>@@": <Pandas DataFrame>
                }
            }
        Note: The dataframe denoted by "@@<df_ref_2>@@" should contain a column denoting which answer
                from "@@<df_ref_1>@@" it associated with. As a result, we should group by on the second dataframe by
                the column @@<df_ref_1>@@ and perform the desired operation on the result.
        :return: A Pandas DataFrame containing the answer.
        """

        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Get the arg that specifies the operation
            operation_str = step.operator_args[0]

            # Get the dataframes and references to prior steps
            df1_str = step.operator_args[1]
            df1 = prior_step_answers[df1_str]

            df2_str = step.operator_args[2]
            df2 = prior_step_answers[df2_str]

            # Perform the required grouping and evaluation operation
            if operation_str == 'count':
                df = df1.groupby([df2_str]).count()
            elif operation_str == 'avg':
                df = df1.groupby([df2_str]).mean()
            elif operation_str == 'min':
                df = df1.groupby([df2_str]).min()
            elif operation_str == 'max':
                df = df1.groupby([df2_str]).max()
            elif operation_str == 'sum':
                df = df1.groupby([df2_str]).sum()
            else:
                raise UnhandledSubOperationTypeError(operation_str)

            # Clean up the data frame
            df = df.reset_index()
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', df2_str])
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0

            # Return the resulting DataFrame
            return df
        except Exception as e:
            print("Unable to perform the Group operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()
