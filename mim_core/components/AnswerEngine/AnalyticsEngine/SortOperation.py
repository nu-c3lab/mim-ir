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
Sort Operation

September 12, 2021
Authors: C3 Lab
"""

import re
import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new

class SortOperation(AnalyticOperation):
    """
    A class for carrying out a Sort operation.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Performs a sort operation on the answer values in the first dataframe using the second dataframe.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the sort analysis operation is as follows:
            {
                "args": ["<df_ref_1>", "<df_ref_2 | string>"],
                "prior_step_answers": {
                        "<df_ref_1>": <Pandas DataFrame>,
                        "<df_ref_2>": <Pandas DataFrame>
                }
            }
        Note: df1 is expected to have an 'id' and 'answer' column, while df2 should have both of those plus a column
                referencing df1 (i.e. '@@1@@')
        :return: A Pandas DataFrame containing the answer.
        """
        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Pull out the data frames to operate on
            arg_0_ref = re.search(r'@@\d+@@', step.operator_args[0])
            arg_1_ref = re.search(r'@@\d+@@', step.operator_args[1])

            df1_str = arg_0_ref.group(0)
            df1 = prior_step_answers[df1_str]

            df2_str = arg_1_ref.group(0)
            df2 = prior_step_answers[df2_str]

            # Rename columns of the two dataframes and create copies of them
            answer_df2_str = 'answer' + df2_str
            df1_cpy = df1.rename(columns={'id': df1_str})
            df2_cpy = df2.rename(columns={'id': df2_str, 'answer': answer_df2_str})

            # Merge df1 and df2 into a new df (unsorted)
            df = df1_cpy.merge(df2_cpy[[df2_str, answer_df2_str, df1_str]], on=df1_str, how='inner')

            # Parse the second argument to figure out how to sort the dataframe
            operation_str = step.operator_args[1]
            operation_type = self.extract_sort_operation_type(operation_str)

            # Sort the final dataframe using the desired operation on the 'answer_df2_str' column
            df.sort_values(answer_df2_str, inplace=True, ascending=(operation_type == 'ascend'))

            # Drop the 'answer_df2_str' column from the final dataframe
            df = df.drop(columns=[answer_df2_str])

            # Clean up the dataframe
            df = df.reset_index()
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', df1_str, df2_str])
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0

            return df
        except Exception as e:
            print("Unable to perform the Sort operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def extract_sort_operation_type(self,
                                    operation_str: str) -> str:
        """
        Extracts the sort operation type from the given string.
        :param operation_str: String containing some sorting specification.
        :return: A string representing the sorting operation to perform (either 'ascend' or 'descend').
        """

        # Initialize the default sorting operation
        operation_type = 'ascend'

        ######################################################################
        # Use single keywords to determine sorting order
        ######################################################################
        # Initialize key words that indicate the sorting operation
        primary_ascending_keys = ['ascend']
        primary_descending_keys = ['descend']

        secondary_ascending_keys = ['alphabetic', 'chronological']

        for word in primary_ascending_keys:
            if word in operation_str:
                return 'ascend'

        for word in primary_descending_keys:
            if word in operation_str:
                return 'descend'

        for word in secondary_ascending_keys:
            if word in operation_str:
                return 'ascend'

        ######################################################################
        # Use order pairs of words to determine the proper sorting order
        ######################################################################
        # Initialize pairs of ordering terms (should be listed as smaller to larger values)
        ordering_pairs = [
            ('low', 'high'),
            ('least', 'greatest'),
            ('first', 'last'),
            ('young', 'old'),
            ('small', 'large')
        ]

        # Find index of each term in the second element and correlate them with the best ordering term pair
        for pair in ordering_pairs:
            first_idx = operation_str.find(pair[0])
            second_idx = operation_str.find(pair[1])
            if first_idx == -1 and second_idx == -1:
                break
            else:
                if first_idx != -1 and second_idx == -1:
                    return 'ascend'
                elif first_idx == -1 and second_idx != -1:
                    return 'descend'
                elif first_idx < second_idx:
                    return 'ascend'
                elif second_idx < first_idx:
                    return 'descend'

        return operation_type
