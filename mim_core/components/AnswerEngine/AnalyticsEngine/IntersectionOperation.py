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
Intersection Operation

September 12, 2021
Authors: C3 Lab
"""

import re
import pandas as pd
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new

class IntersectionOperation(AnalyticOperation):
    """
    A class for carrying out a intersection operation.
    """

    def __init__(self, **kwargs):
        self.answer_similarity_threshold = kwargs.get('answer_similarity_threshold', 0.5)
        super().__init__()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Performs an intersection on the answer values in the two/three provided dataframes.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the intersection operation is as follows:
            {
                "args": ["<df_ref_3 | string> ", "<df_ref_1>", "<df_ref_2>"],
                "prior_step_answers": {
                        "<df_ref_3>": <Pandas DataFrame>,
                        "<df_ref_1>": <Pandas DataFrame>,
                        "<df_ref_2>": <Pandas DataFrame>
                }
            }
        Note: Assumes that both df1 and df2 have columns associated with df3
        :return: A Pandas DataFrame containing the answer.
        """

        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Pull out the data frames to operate on
            df1_str = step.operator_args[1]
            df1 = prior_step_answers[df1_str]

            df2_str = step.operator_args[2]
            df2 = prior_step_answers[df2_str]

            search_result = re.search(r'@@\d+@@', step.operator_args[0])
            if search_result:
                ##############################################################################
                # Intersection Case 1: First argument is reference to another dataframe
                # e.g. INTERSECTION['@@3@@', '@@1@@', '@@2@@']
                ##############################################################################
                # Pull out the final data frame to operate on
                df3_str = search_result.group(0)
                df3 = prior_step_answers[df3_str]

                # Rename id column to df1_str and df2_str
                df1_cpy = df1.rename(columns={"id": df1_str})
                df2_cpy = df2.rename(columns={"id": df2_str})

                # Pull out the columns (df1_str, df3_str) and (df2_str, df3_str) from each of df1 and df2 and perform an inner join
                df = pd.merge(df1_cpy[[df1_str, df3_str]], df2_cpy[[df2_str, df3_str]], on=df3_str, how='inner')

                # Rename the id column of df3 to df3_str and then merge df3 into df
                df3_cpy = df3.rename(columns={"id": df3_str})
                df = df.merge(df3_cpy[[df3_str, 'answer']], on=df3_str, how='inner')

                # Clean up the data frame
                df = df.reset_index()
                df['id'] = df.index
                df = df.reindex(columns=['id', 'answer', df3_str, df1_str, df2_str])
                df['source'] = str(self.__class__.__name__)
                df['confidence'] = 1.0

                # Return the resulting DataFrame
                return df
            else:
                ##############################################################################
                # Intersection Case 2: First argument is a string referencing some class of entities
                # e.g. INTERSECTION['objects', '@@1@@', '@@2@@']
                ##############################################################################
                # TODO: May want to add an additional check that the items retrieved
                #       from the two dataframes match types in the first argument

                d = {}
                for df1_idx, df1_row in df1.iterrows():
                    for df2_idx, df2_row in df2.iterrows():
                        intersected_answer = self._intersect_words_in_strings(df1_row['answer'], df2_row['answer'])
                        if intersected_answer != '':
                            d[intersected_answer] = (df1_row['id'], df2_row['id'])

                # Create the new dataframe with these answers
                if len(d) == 0:
                    df_rows = [{'id': 0, 'answer': None, df1_str: None, df2_str: None}]
                else:
                    df_rows = [{'id': idx, 'answer': key, df1_str: d[key][0], df2_str: d[key][1]} for idx, key in enumerate(d)]
                df = pd.DataFrame(df_rows, columns=['id', 'answer', df1_str, df2_str])

                # Clean up the data frame
                df = df.reset_index()
                df['id'] = df.index
                df = df.reindex(columns=['id', 'answer', df1_str, df2_str])
                df['source'] = str(self.__class__.__name__)
                df['confidence'] = 1.0

                # Return the resulting DataFrame
                return df

        except Exception as e:
            print("Unable to perform the Intersection operation.")

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

        """
        Good test question (ids):
        -------------------------
        5aba749055429901930fa7d8
        5ab4e3065542990594ba9cb4
        5a88d745554299206df2b378
        5abe689a55429976d4830b16
        5abb30755542996cc5e49fd8
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
        return " ".join(intersected) if len(intersected) > self.answer_similarity_threshold * len(s_shorter) else ''
