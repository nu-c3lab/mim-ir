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
Discard Operation

September 12, 2021
Authors: C3 Lab
"""

import re
import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnexpectedOperationArgsError

class DiscardOperation(AnalyticOperation):
    """
    A class for carrying out a discard operation.
    """

    def __init__(self, **kwargs):
        self.answer_similarity_threshold = kwargs.get('answer_similarity_threshold', 0.5)
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Performs a discard operation on the answer values in the one/two provided dataframes.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the intersection operation is as follows:
            {
                "args": ["<df_ref_1 | string> ", "<df_ref_2 | string>"],
                "prior_step_answers": {
                        "<df_ref_1>": <Pandas DataFrame>,
                        "<df_ref_2>": <Pandas DataFrame>
                }
            }
        NOTE: This is equivalent to a set difference in relational algebra.
        :return: A Pandas DataFrame containing the answer.
        """
        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            arg_0_ref = re.search(r'@@\d+@@', step.operator_args[0])
            arg_1_ref = re.search(r'@@\d+@@', step.operator_args[1])
            if arg_0_ref and arg_1_ref:
                ##############################################################################
                # Discard Case 1: Both arguments reference a dataframe
                # e.g. DISCARD['@@1@@', '@@2@@']
                ##############################################################################
                # Pull out the data frames to operate on
                df1_str = arg_0_ref.group(0)
                df1 = prior_step_answers[df1_str]

                df2_str = arg_1_ref.group(0)
                df2 = prior_step_answers[df2_str]

                # Perform the discard operation by only keeping values in df1 that don't exist in df2
                #df = df1.loc[~df1['answer'].isin(df2['answer'])]  # TODO: This line may need some debugging... Produces Pandas warning
                df = df1[~df1['answer'].apply(lambda one: any(self._intersect_words_in_strings(one, two) for two in df2['answer']))]

                # Rename the id column
                df.rename(columns={"id": df1_str}, inplace=True)

                # Clean up the data frame
                df = df.reset_index()
                df['id'] = df.index
                df = df.reindex(columns=['id', 'answer', df1_str])
                df['source'] = str(self.__class__.__name__)
                df['confidence'] = 1.0

                # Return the resulting DataFrame
                return df
            else:
                ##############################################################################
                # Discard Case 2: Only one argument references a dataframe
                # e.g. DISCARD['objects', '@@2@@']
                # e.g. DISCARD['@@1@@', 'blue']
                ##############################################################################
                # Determine which argument is the referenced dataframe and which is the string argument
                if arg_0_ref and not arg_1_ref:
                    prev_str = arg_0_ref.group(0)
                    prev_df = prior_step_answers[prev_str]
                    filter_str = step.operator_args[1]
                elif not arg_0_ref and arg_1_ref:
                    prev_str = arg_1_ref.group(0)
                    prev_df = prior_step_answers[prev_str]
                    filter_str = step.operator_args[0]
                else:
                    raise UnexpectedOperationArgsError(step.operator_args)

                # Filter out all answers that contain the filter string as a substring
                # df = prev_df[~prev_df['answer'].str.contains(filter_str)]
                df = prev_df[~prev_df['answer'].apply(lambda x: self._intersect_words_in_strings(x, filter_str))]

                # Rename the id column
                df = df.rename(columns={"id": prev_str})

                # Clean up the data frame
                df = df.reset_index()
                df['id'] = df.index
                df = df.reindex(columns=['id', 'answer', 'confidence', prev_str])
                df['source'] = str(self.__class__.__name__)
                # df['confidence'] = 1.0

                # Return the resulting DataFrame
                return df

        except Exception as e:
            print("Unable to perform the Discard operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def _intersect_words_in_strings(self,
                                    s1: str,
                                    s2: str) -> str:
        """
        Finds the words that are common to both strings and
            returns them in the order they occur in the longer of the two input strings.
        :param s1: The first string.
        :param s2: The second string.
        :return: A string of words that are common to both input strings
                    in the order they occur in the longer of the two input strings.
        """

        # Splits the strings
        s1_l = s1.split(" ")
        s2_l = s2.split(" ")

        # Create a set of words from the longer string (set is assumed to be ordered in the same way as the input list)
        if len(s1_l) < len(s2_l):
            s_longer = s2_l
            s_shorter = s1_l
        else:
            s_longer = s1_l
            s_shorter = s2_l

        # Output the words in both, joined with whitespace
        intersected = [w for w in s_shorter if w in s_longer]
        return len(intersected) >= self.answer_similarity_threshold * len(s_shorter)
