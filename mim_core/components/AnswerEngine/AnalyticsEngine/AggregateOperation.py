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
Aggregate Operation

September 12, 2021
Authors: C3 Lab
"""

import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnexpectedOperationArgsError

class AggregateOperation(AnalyticOperation):
    """
    A class for carrying out a aggregate operation.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Counts the elements in the "answer" column of the Pandas DataFrame referenced by the value in "args."
        :param step: The step for which to carry out the analysis.
        :param mqr: The MQR/plan that the step came from.
        Note: Expected form of the data for the count analysis operation is as follows:
            {
                "args": ["<aggregation_type_str>", "<df_ref_1>"],
                "prior_step_answers": {
                        "@@<df_ref_1>@@": <Pandas DataFrame>
                }
            }
        :return: A Pandas DataFrame containing the answer.
        """

        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Get the arg that specifies the operation
            operation_str = step.operator_args[0]

            # Get the first arg that points to one of the prior step answers
            df1_str = step.operator_args[1]  # next(arg for arg in data["args"] if '@@' in arg)
            df1 = prior_step_answers[df1_str]

            if operation_str == 'count':
                # Perform the count operation and extract the value
                operation_result = df1.count()['answer']
            else:
                raise UnexpectedOperationArgsError(step.operator_args)

            # Create and return the Pandas DataFrame with this answer
            df = pd.DataFrame(columns=['id', 'answer'])
            df = df.append({'id': 0, 'answer': operation_result}, ignore_index=True)
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0
            return df
        except Exception as e:
            print("Unable to perform the Aggregation operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()
