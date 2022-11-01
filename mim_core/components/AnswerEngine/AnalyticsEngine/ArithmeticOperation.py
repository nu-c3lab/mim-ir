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
Arithmetic Operation

September 12, 2021
Authors: C3 Lab
"""

import datetime
import dateparser
import pandas as pd
from dateutil import relativedelta
from mim_core.structs.Step import Step
from num_parse.NumParser import NumParser
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnexpectedOperationArgsError

class ArithmeticOperation(AnalyticOperation):
    """
    A class for carrying out a arithmetic operation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_parser = NumParser()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Performs the specified arithmetic operation on the first two values in the provided dataframes.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the sum analysis operation is as follows:
            {
                "args": ["<arithmetic_operation_str>", "<df_ref_1>", "<df_ref_2>"],
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

            # Get the arg that specifies the operation
            operation_str = step.operator_args[0]

            # Pull out the data frames to operate on
            df1_str = step.operator_args[1]
            df1 = prior_step_answers[df1_str]

            df2_str = step.operator_args[2]
            df2 = prior_step_answers[df2_str]

            # Check the parent step's expected type to determine the best parser to use
            if step.parent_steps[0].expected_answer_type[0] == 'NUMERIC':
                df1_val = self.num_parser.parse_num(df1.iloc[0]['answer'])
                df2_val = self.num_parser.parse_num(df2.iloc[0]['answer'])
            elif step.parent_steps[0].expected_answer_type[0] == 'DATE':
                df1_val = dateparser.parse(df1.iloc[0]['answer'])
                df2_val = dateparser.parse(df2.iloc[0]['answer'])
            else:
                # Default case: numeric parser
                df1_val = self.num_parser.parse_num(df1.iloc[0]['answer'])
                df2_val = self.num_parser.parse_num(df2.iloc[0]['answer'])

            # Log the values
            step.misc['df1_val'] = str(df1_val)
            step.misc['df2_val'] = str(df2_val)

            # Perform the operation on the first two elements of the answer columns in the two dataframes
            if operation_str == 'sum':
                operation_result = df1_val + df2_val
            elif operation_str == 'difference':
                if type(df1_val) is datetime.datetime and type(df2_val) is datetime.datetime:
                    operation_result = str(abs(relativedelta.relativedelta(df1_val, df2_val).years)) + " years"
                else:
                    operation_result = df1_val - df2_val
            elif operation_str == 'division':
                operation_result = df1['answer'].iloc[0] / df2['answer'].iloc[0]
            else:
                raise UnexpectedOperationArgsError(step.operator_args)

            # Create and return the Pandas DataFrame with the answer
            df = pd.DataFrame(columns=['id', 'answer'])
            df = df.append({'id': 0, 'answer': operation_result}, ignore_index=True)
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0
            return df
        except Exception as e:
            print("Unable to perform the Arithmetic operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()
