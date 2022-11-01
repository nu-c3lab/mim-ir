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
Comparative Operation

September 12, 2021
Authors: C3 Lab
"""

import re
import pandas as pd
from mim_core.structs.Step import Step
from num_parse.NumParser import NumParser
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnhandledSubOperationTypeError

class ComparativeOperation(AnalyticOperation):
    """
    A class for carrying out a comparative operation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_parser = NumParser()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Compares answers in df1 by the values in the column "<df_ref_2>" of df1 and performs the required operation.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the comparative analysis operation is as follows:
            {
                "args": ["<df_ref_1>", "<df_ref_2>", "<comparator string (may contain reference to previous answers)>"],
                "prior_step_answers": {
                        "@@<df_ref_1>@@": <Pandas DataFrame>,
                        "@@<df_ref_2>@@": <Pandas DataFrame>,
                        "@@<df_ref_3>@@": <Pandas DataFrame>
                }
            }
        Comparison types implemented: numeric
        Comparison types NOT implemented (not exhaustive): dates, containment, starts/ends with, ...
        :return: A Pandas DataFrame containing the answer.
        """

        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            # Get the arg that specifies the operation and parse out the required operation type and object
            operation_str = step.operator_args[2]
            operation_type, operation_object = self.extract_comparative_operation_type(operation_str)

            # if still no operation identified, throw error
            if operation_type is None:
                raise UnhandledSubOperationTypeError(operation_str)

            # Get the dataframes and references to prior steps
            df1_str = step.operator_args[0]
            df1 = prior_step_answers[df1_str]

            df2_str = step.operator_args[1]
            df2 = prior_step_answers[df2_str]

            df3 = None
            df3_str = None

            # parse the comparison object
            # first, check if the comparison object has a reference (e.g. X where Y is same as Z)
            if '@@' in operation_str:
                ##############################################################################
                # Comparative Case 1: Comparison string contains reference to 3rd dataframe
                ##############################################################################
                df3_str = re.findall('@@[\d]+@@', operation_object)
                df3 = prior_step_answers[df3_str]

                try:
                    if operation_type == 'equality':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) == df3['answer'].map(lambda n: self.num_parser.parse_num(n))][df1_str]]
                    elif operation_type == 'greater than':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) > df3['answer'].map(lambda n: self.num_parser.parse_num(n))][df1_str]]
                    elif operation_type == 'greater equality':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) >= df3['answer'].map(lambda n: self.num_parser.parse_num(n))][df1_str]]
                    elif operation_type == 'less than':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) < df3['answer'].map(lambda n: self.num_parser.parse_num(n))][df1_str]]
                    elif operation_type == 'less equality':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) <= df3['answer'].map(lambda n: self.num_parser.parse_num(n))][df1_str]]
                    else:
                        raise UnhandledSubOperationTypeError(operation_type)
                except ValueError:
                    if operation_type != 'equality':
                        # No support for inequality operations on objects that are not fully numeric
                        raise UnhandledSubOperationTypeError(operation_type)
                    df = df1.iloc[df2[df2['answer'] == df3['answer']][df1_str]]
            else:
                ##############################################################################
                # Comparative Case 2: Comparison string does not reference 3rd dataframe
                ##############################################################################
                parsed_object = None
                numeric_object = False
                # search for numerical words in the answer
                try:
                    parsed_object = self.num_parser.parse_num(operation_object)
                    numeric_object = True
                except ValueError:

                    # no numerical objects found - we can only perform equality operations if this is the case
                    # important: this means that we can't perform comparative operations on objects like dates yet
                    if parsed_object is None:
                        if operation_type != 'equality':
                            raise UnhandledSubOperationTypeError('inequality operation on objects that are not fully numeric')
                        parsed_object = operation_object.strip()

                if numeric_object:
                    if operation_type == 'equality':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) == parsed_object][df1_str]]
                    elif operation_type == 'greater than':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) > parsed_object][df1_str]]
                    elif operation_type == 'greater equality':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) >= parsed_object][df1_str]]
                    elif operation_type == 'less than':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) < parsed_object][df1_str]]
                    elif operation_type == 'less equality':
                        df = df1.iloc[df2[df2['answer'].map(lambda n: self.num_parser.parse_num(n)) <= parsed_object][df1_str]]
                    else:
                        raise UnhandledSubOperationTypeError(operation_type)
                else:
                    if operation_type == 'equality':
                        df = df1.iloc[df2[df2['answer'] == parsed_object][df1_str]]
                    else:
                        raise UnhandledSubOperationTypeError(operation_type)

            # Correlate the ids of the earlier steps with the answers for the current step
            # Join df1 based on the answer column to df
            df1_cpy = df1.rename(columns={"id": df1_str})
            df = df.merge(df1_cpy[[df1_str, 'answer']], on='answer', how='left')

            # Join df2 based on the df1_str column to df
            df2_cpy = df2.rename(columns={"id": df2_str})
            df = df.merge(df2_cpy[[df1_str, df2_str]], on=df1_str, how='left')

            # If it exists, join df3 on df2_str column to df
            if df3 and df3_str:
                df3_cpy = df3.rename(columns={"id": df3_str})
                df = df.merge(df3_cpy[[df2_str, df3_str]], on=df2_str, how='left')

            # Clean up the data frame
            df = df.reset_index()
            df['id'] = df.index
            df = df.reindex(
                columns=['id', 'answer', df1_str, df2_str] if not df3_str else ['id', 'answer', df1_str, df2_str,
                                                                                df3_str])
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0

            # Return the resulting DataFrame
            return df

        except Exception as e:
            print("Unable to perform the Comparative operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def extract_comparative_operation_type(self,
                                           operation_str: str) -> (str, str):
        """
        Extracts the comparison operation type and the object against which a comparison is made.
        :param operation_str: String containing some comparison statement (e.g. 'is more than 3')
        :return: Two strings representing the comparison operation type and the object against which a comparison is made.
        """

        # we distinguish primary and secondary equality, because words like "is" might be present in a comparison but
        # only signal equality given the lack of other comparison keywords. E.g. "X where Y is larger than 500" should
        # be treated as a greater than comparison, despite having "is".
        # However, we want to catch equality cases where we have steps like "X where Y is 3"
        primary_equality_keys = ['same as ', 'equal ', 'equals ']
        secondary_equality_keys = ['is ', 'are ', 'was ']

        greater_than_keys = ['larger than ', 'greater than ', 'higher than ', 'more than ']
        greater_equality_keys = ['at least ']
        less_than_keys = ['lower than ', 'smaller than ', 'less than ']
        less_equality_keys = ['at most ']

        comparison_types = {
            'equality': primary_equality_keys,
            'greater than': greater_than_keys,
            'greater equality': greater_equality_keys,
            'less than': less_than_keys,
            'less equality': less_equality_keys
        }

        containment_keys = ['contain ', 'contains ', 'include ', 'includes ', 'has ', 'have ']
        starts_with_keys = ['start with ', 'starts with', 'begin with', 'begins with']
        ends_with_keys = ['end with ', 'ends with']
        unimplemented_keys = containment_keys + starts_with_keys + ends_with_keys

        # containment, starts with, and ends with have yet to be implemented
        for u_key in unimplemented_keys:
            if u_key in operation_str:
                raise UnhandledSubOperationTypeError(u_key)  # No support for containment, starts with, or ends with operations

        operation_type = None
        operation_object = None
        # check for primary keys for each comparison type
        for t in comparison_types.keys():
            keys = comparison_types[t]
            for k in keys:
                if k in operation_str:
                    operation_type = t
                    operation_object = operation_str.split(k)[-1]
                    break
            if operation_type is not None:
                break

        # if still haven't identified comparison type, check secondary equality
        if operation_type is None:
            for k in secondary_equality_keys:
                if k in operation_str:
                    operation_type = 'equality'
                    operation_object = operation_str.split(k)[-1]
                    break
        return operation_type, operation_object
