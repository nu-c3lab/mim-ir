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
Boolean Operation

September 12, 2021
Authors: C3 Lab
"""

import re
import dateparser
import pandas as pd
from mim_core.structs.Step import Step
from num_parse.NumParser import NumParser
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnexpectedOperationArgsError, UnhandledSubOperationTypeError

class BooleanOperation(AnalyticOperation):
    """
    A class for carrying out a boolean operation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.num_parser = NumParser()

    def execute(self,
                step: Step) -> pd.DataFrame:
        """
        Performs a boolean compares of answers in df1 with the values in the column "<df_ref_2>" of df1 (or using just
        the info from the boolean comparison string in the case where only one dataframe is being operated on). This
        function will essentially return a single boolean answer based on some condition being satisfied.
        :param step: The step for which to carry out the analysis.
        Note: Expected form of the data for the boolean analysis operation is as follows:
            {
                "args": ["<df_ref_1>", "<df_ref_2>", "<boolean comparison string>"],
                "prior_step_answers": {
                        "@@<df_ref_1>@@": <Pandas DataFrame>,
                        "@@<df_ref_2>@@": <Pandas DataFrame>,
                }
            }
        :return: A Pandas DataFrame containing the answer.
        """

        try:
            # Get the dataframes containing the answers from previous steps that are going to be compared
            prior_step_answers = get_prev_step_answers_new(step)

            if len(step.operator_args) == 2:
                # Get the arg that specifies the operation and parse out the required operation type and object
                operation_str = step.operator_args[1]
                operation_type, operation_object = self._extract_boolean_operation_type(operation_str)

                # if still no operation identified, throw error
                if operation_type is None:
                    raise UnhandledSubOperationTypeError(operation_str)

                # Get the dataframes and references to prior steps
                df1_str = step.operator_args[0]
                df1 = prior_step_answers[df1_str]

                df2_str = None
                df2 = None

                # Check if there is a reference to a past answer in the boolean comparison string
                if '@@' in operation_object:
                    ##############################################################################
                    # Boolean Case 1: Boolean conditional string references a dataframe
                    # e.g. BOOLEAN['@@1@@', 'is same as @@2@@']
                    # e.g. BOOLEAN['@@1@@', 'is there any @@REF@@']
                    ##############################################################################
                    ref_result = re.search('@@(\d+|REF)+@@', operation_object)
                    if ref_result and 'REF' not in ref_result.group(0) and df1_str not in ref_result.group(0):
                        ##############################################################################
                        # Boolean Case 1a: Boolean conditional string references a second dataframe
                        ##############################################################################
                        df2_str = ref_result.group(0)
                        df2 = prior_step_answers[df2_str]

                        # Get the top answer from each dataframe to compare
                        df1_answer = df1['answer'][0]
                        df2_answer = df2['answer'][0]

                        # Parse the answers into a more structured form, if possible
                        if step.parent_steps[0].expected_answer_type[0] == 'NUMERIC':
                            try:
                                df1_val = self.num_parser.parse_num(df1_answer)
                                df2_val = self.num_parser.parse_num(df2_answer)
                            except ValueError:
                                df1_val = df1_answer
                                df2_val = df2_answer
                        elif step.parent_steps[0].expected_answer_type[0] == 'DATE':
                            try:
                                df1_val = dateparser.parse(df1_answer)
                                df2_val = dateparser.parse(df2_answer)
                            except ValueError:
                                df1_val = df1_answer
                                df2_val = df2_answer
                        else:
                            df1_val = df1_answer
                            df2_val = df2_answer

                        # Log the values
                        step.misc['df1_val'] = str(df1_val)
                        step.misc['df2_val'] = str(df2_val)

                        # Perform the boolean comparison for numeric/datetime answers
                        if operation_type == 'equality':
                            operation_result = df1_val == df2_val
                        elif operation_type == 'greater than':
                            operation_result = df1_val > df2_val
                        elif operation_type == 'greater equality':
                            operation_result = df1_val >= df2_val
                        elif operation_type == 'less than':
                            operation_result = df1_val < df2_val
                        elif operation_type == 'less equality':
                            operation_result = df1_val <= df2_val
                        else:
                            raise UnhandledSubOperationTypeError(operation_type)

                        # Put the result in a dataframe
                        df = pd.DataFrame(columns=['id', 'answer'])
                        df = df.append({'id': 0, 'answer': operation_result}, ignore_index=True)

                    else:
                        ##############################################################################
                        # Boolean Case 1a: Boolean conditional string references a single dataframe
                        ##############################################################################
                        if operation_type == 'existence':
                            # Check if an answer exists
                            operation_result = len(df1.index) > 0
                            df = pd.DataFrame(columns=['id', 'answer'])
                            df = df.append({'id': 0, 'answer': operation_result}, ignore_index=True)

                        else:
                            raise UnhandledSubOperationTypeError(operation_type)

                    # Create and return the Pandas DataFrame with this answer
                    df['source'] = str(self.__class__.__name__)
                    df['confidence'] = 1.0
                    return df

                else:
                    ##############################################################################
                    # Boolean Case 2: Boolean conditional string references no dataframe
                    # e.g. BOOLEAN['@@1@@', 'is at least one']
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
                            df = df1[df1['answer'].map(lambda n: self.num_parser.parse_num(n)) == parsed_object]
                        elif operation_type == 'greater than':
                            df = df1[df1['answer'].map(lambda n: self.num_parser.parse_num(n)) > parsed_object]
                        elif operation_type == 'greater equality':
                            df = df1[df1['answer'].map(lambda n: self.num_parser.parse_num(n)) >= parsed_object]
                        elif operation_type == 'less than':
                            df = df1[df1['answer'].map(lambda n: self.num_parser.parse_num(n)) < parsed_object]
                        elif operation_type == 'less equality':
                            df = df1[df1['answer'].map(lambda n: self.num_parser.parse_num(n)) <= parsed_object]
                        elif operation_type == 'existence':
                            df = df1
                        else:
                            raise UnhandledSubOperationTypeError(operation_type)
                    else:
                        if operation_type == 'equality':
                            df = df1[df1['answer'] == parsed_object]            # TODO: Add NumParse?
                        elif operation_type == 'existence':
                            df = df1
                        else:
                            raise UnhandledSubOperationTypeError(operation_type)

                    # Check if the boolean conditional had any answers that satisfied it
                    operation_result = len(df.index) > 0

                    # Create and return the Pandas DataFrame with this answer
                    df = pd.DataFrame(columns=['id', 'answer'])
                    df = df.append({'id': 0, 'answer': operation_result}, ignore_index=True)
                    df['source'] = str(self.__class__.__name__)
                    df['confidence'] = 1.0
                    return df

            elif len(step.operator_args) == 4:
                ##############################################################################
                # Boolean Case 3: Logical operation being performed on two dataframes
                # e.g. BOOLEAN['logical_and', 'true', '@@1@@', '@@2@@']
                ##############################################################################
                # Get the type of logical operation to perform and the value it should have
                operation_type = step.operator_args[0]
                operation_value = step.operator_args[1]

                # Get the dataframes and references to prior steps
                df1_str = step.operator_args[2]
                df1 = prior_step_answers[df1_str]

                df2_str = step.operator_args[3]
                df2 = prior_step_answers[df2_str]

                # Get the first values in each of the two dataframes
                df1_bool = df1['answer'][0]
                df2_bool = df2['answer'][0]

                # Perform the operation
                if operation_type == 'logical_and':
                    operation_result = df1_bool and df2_bool if operation_value == 'true' else not (
                            df1_bool and df2_bool)
                elif operation_type == 'logical_or':
                    operation_result = df1_bool or df2_bool if operation_value == 'true' else not (df1_bool or df2_bool)
                else:
                    raise UnhandledSubOperationTypeError(operation_type)

                # Create and return the Pandas DataFrame with this answer
                df = pd.DataFrame(columns=['id', 'answer'])
                df = df.append({'id': 0, 'answer': operation_result}, ignore_index=True)
                df['source'] = str(self.__class__.__name__)
                df['confidence'] = 1.0
                return df
            else:
                # Unexpected number of arguments to the Boolean operation
                raise UnexpectedOperationArgsError(step.operator_args)
        except Exception as e:
            print("Unable to perform the Boolean operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def _extract_boolean_operation_type(self,
                                        operation_str: str) -> (str, str):
        """
        Extracts the boolean operation type and the object against which a boolean comparison is made.
        :param operation_str: String containing some boolean comparison statement (e.g. 'is more than 3')
        :return: Two strings representing the boolean operation type and the object against which a boolean comparison is made (if any).
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

        existence_keys = ['is any', 'are any', 'is there any', 'are there any']

        comparison_types = {
            'equality': primary_equality_keys,
            'greater than': greater_than_keys,
            'greater equality': greater_equality_keys,
            'less than': less_than_keys,
            'less equality': less_equality_keys,
            'existence': existence_keys
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
