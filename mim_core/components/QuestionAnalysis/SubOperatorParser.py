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
Sub-Operator Parser

June 29, 2021
Authors: C3 Lab
"""

import re
from mim_core.structs.Step import Step
from num_parse.NumParser import NumParser


class SubOperatorParser(object):
    """
    A class for determining the sub-operation types of generated steps.
    """

    def __init__(self):
        self.num_parser = NumParser()

    def parse_subtype(self,
                      step: Step) -> str:
        """
        Determines the subtype of the operation to be performed in the given step.
        :param step: The step for which to determine the subtype.
        :return: A string representing the operation subtype.
        """
        if step.operator_type.lower() == 'select':
            return self._determine_select_sub_operation(step)
        elif step.operator_type.lower() == 'intersection':
            return self._determine_intersection_sub_operation(step)
        elif step.operator_type.lower() == 'discard':
            return self._determine_discard_sub_operation(step)
        elif step.operator_type.lower() == 'boolean':
            return self._determine_boolean_sub_operation(step)
        else:
            return step.operator_type

    def _determine_select_sub_operation(self, step: Step) -> str:
        """
        Determines what the sub-operation of a given select step is.
        :param step: The select step.
        :return: The sub-operation type as a string.
        """
        if step.entity_class is None:
            return 'select-single'
        else:
            return 'select-class'

    def _determine_intersection_sub_operation(self, step: Step) -> str:
        """
        Determines what the sub-operation of a given intersection step is.
        :param step: The intersection step.
        :return: The sub-operation type as a string.
        """
        if not re.search(r'@@\d+@@', step.operator_args[0]):
            return 'intersection-filter'
        else:
            return 'intersection'

    def _determine_discard_sub_operation(self, step: Step) -> str:
        """
        Determines what the sub-operation of a given discard step is.
        :param step: The discard step.
        :return: The sub-operation type as a string.
        """
        if (not re.search(r'@@\d+@@', step.operator_args[0])) or (not re.search(r'@@\d+@@', step.operator_args[1])):
            return 'discard-filter'
        else:
            return 'discard'

    def _determine_boolean_sub_operation(self, step: Step) -> str:
        """
        Determines what the boolean sub-operation of a given boolean step is.
        :param step: The boolean step.
        :return: The sub-operation type as a string.
        """
        if len(step.operator_args) == 4:
            return 'boolean-truth-eval'
        elif len(step.operator_args) == 2:
            # Parse the operation string in the second argument
            operation_type, operation_object = self._extract_boolean_operation_type(step.operator_args[1])

            # Check if the operation is search (NO references to dataframe; operation_object is NOT numeric/datetime )
            if operation_object and (not self._is_numeric(operation_object)) and (not re.search('@@(\d+|REF)+@@', operation_object)):
                return 'boolean-search'
            # Check if the operation is existence
            elif operation_type and operation_type == 'existence':
                return 'boolean-existence'
            # Check if the operation is inequality
            elif operation_type and operation_type in ['equality', 'greater than', 'greater equality', 'less than', 'less equality']:
                return 'boolean-inequality'
            else:
                # TODO: Figure out what to do with unknown subtypes
                return 'boolean'
        elif len(step.operator_args) == 1 and (not re.search('@@(\d+|REF)+@@', step.operator_args[0])):
            # Check if there is a single boolean string, which represents some kind of search operation
            return 'boolean-search'
        else:
            # TODO: Figure out what to do with unhandled number of arguments
            return 'boolean'

    def _is_numeric(self, operation_object: str) -> bool:
        # TODO: Put functionality like this is a "utils" script
        try:
            # search for numerical words in the answer
            _ = self.num_parser.parse_num(operation_object)
            return True
        except ValueError:
            # search for a numerical digit in the answer
            for token in operation_object.split():
                try:
                    _ = float(token)
                    return True
                except ValueError:
                    pass
        return False

    def _extract_boolean_operation_type(self, operation_str: str) -> (str, str):
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
