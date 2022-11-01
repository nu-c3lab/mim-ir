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
Comparison Operation

August 31, 2021
Authors: C3 Lab
"""

import dateparser
import pandas as pd
from typing import List
from operator import attrgetter
from itertools import chain
from num_parse.NumParser import NumParser
from mim_core.structs.Step import Step
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticOperation import AnalyticOperation
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new
from mim_core.exceptions import UnhandledSubOperationTypeError
from mim_core.components.models import get_model

class ComparisonOperation(AnalyticOperation):
    """
    A class for carrying out a comparison operation.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.spacy_model = get_model("en_core_web_trf")
        self.num_parser = NumParser()

    def _get_tree_dep(self, tok, dep):
        if list(tok.children):
            return next(chain(filter(lambda child: child.dep_ == dep, tok.children), filter(lambda tok: tok, map(lambda child: self._get_tree_dep(child, dep), tok.children))), None)
            # return any(child.dep_ == dep or self._tree_has_dep(child, dep) for child in tok.children)
        else:
            return tok if tok.dep_ == dep else None

    def get_reference_entity(self,
                             step: Step) -> List[str]:
        """
        Extracts the proper noun chunks in the qdmr of the given step.
        :param step: The step to find the proper nouns in.
        :return: A list of proper noun chunks.
        """
        doc = self.spacy_model(step.qdmr)
        noun_chunks = list(doc.noun_chunks)

        root = next(filter(lambda tok: tok.dep_ == 'ROOT', doc))
        if step.operator_type == 'boolean':
            subject_head = self._get_tree_dep(root, 'nsubj')
            if subject_head:
                toks = list(subject_head.subtree)
                return doc[toks[0].i:toks[-1].i + 1].text
            else:
                return noun_chunks[0].text
        else:
            object_head = self._get_tree_dep(root, 'pobj')
            if object_head:
                toks = list(object_head.subtree)
                return doc[toks[0].i:toks[-1].i + 1].text
            else:
                return noun_chunks[-1].text

    def get_root_step(self,
                      df_str: str,
                      comparison_step: Step) -> Step:
        """
        Get the root step corresponding to the step with the given df_str ref
        :param df_str:
        :param comparison_step:
        :return:
        """
        step_ref_id = int(df_str.strip('@@'))
        comparison_parents = comparison_step.get_parent_steps()

        df_str_step = None      # The step corresponding to the df_str ref
        for p in comparison_parents:
            if p.reference_id == step_ref_id:
                df_str_step = p
                break

        parents = df_str_step.get_ancestor_steps()
        if parents:
            # Return the ancestor with the smallest reference id
            return min(parents, key=attrgetter('reference_id'))
        else:
            # If there are non parents, then the root step is simply the step preceding the comparison step
            return df_str_step

    def get_root_answers(self,
                         df_str: str,
                         step: Step) -> pd.DataFrame:
        """
        Gets the answer(s) dataframe in the root reference.
        :param df_str:
        :param step: The comparison step.
        :return:
        """
        # Get the reference id of the root step
        root_step = self.get_root_step(df_str, step)

        # Get the subject/object of that root step (and use that as the answer)
        return pd.DataFrame([{"answer": self.get_reference_entity(root_step)}], columns=['answer'])

    def execute(self, step: Step) -> pd.DataFrame:
        """
        The main function for carrying out the comparison operation.
        :param step: The comparison step to execute.
        :return: The result of the comparison operation.
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

            # Perform the operation
            if operation_str == 'true' or operation_str == 'false':
                df = self._boolean_comparison(operation_str, df1, df2, df1_str, df2_str, step)

            elif operation_str == 'max' or operation_str == 'min' or operation_str == 'before' or operation_str == 'after':
                df = self._numeric_datetime_comparison(operation_str, df1, df2, df1_str, df2_str, step)
            else:
                raise UnhandledSubOperationTypeError(operation_str)

            # Clean up the data frame
            df['id'] = df.index
            df['source'] = str(self.__class__.__name__)
            df['confidence'] = 1.0

            # Return the resulting DataFrame
            return df

        except Exception as e:
            print("Unable to perform the Comparison [not Comparative] operation.")

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def _boolean_comparison(self, operation_str, df1, df2, df1_str, df2_str, step) -> pd.DataFrame:
        """
        Performs the comparison operation for boolean values.
        :param operation_str:
        :param df1:
        :param df2:
        :param df1_str:
        :param df2_str:
        :return:
        """
        operation_boolean = operation_str == 'true'

        df1_val = bool(df1.iloc[0]['answer'])
        df2_val = bool(df2.iloc[0]['answer'])

        df1_root_answers = create_null_result()
        df2_root_answers = create_null_result()

        # Get the answers from the root parent associated with df1, if it's true
        if df1_val == operation_boolean:
            df1_root_answers = self.get_root_answers(df1_str, step)

        # Get the answers from the root parent associated with df2, if it's true
        if df2_val == operation_boolean:
            df2_root_answers = self.get_root_answers(df2_str, step)

        # Create the final dataframe
        df = pd.concat([df1_root_answers, df2_root_answers], ignore_index=True).drop_duplicates().reset_index(
            drop=True)

        # If there is more than one row in the results, remove rows where answer column has nan
        if len(df) > 1:
            df = df.dropna(subset=['answer'])

        return df

    def _numeric_datetime_comparison(self, operation_str, df1, df2, df1_str, df2_str, step) -> pd.DataFrame:
        """
        Performs the comparison operation for numeric and datetime values.
        e.g. "which is highest of @@4@@,  @@5@@"
        :param operation_str:
        :param df1:
        :param df2:
        :param df1_str:
        :param df2_str:
        :param step:
        :return:
        """

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

        # perform the comparison
        if (df1_val > df2_val and operation_str == 'max') or (df1_val < df2_val and operation_str == 'min') \
                or (df1_val < df2_val and operation_str == 'before') or (df1_val > df2_val and operation_str == 'after'):
            df = self.get_root_answers(df1_str, step)
        elif (df2_val > df1_val and operation_str == 'max') or (df2_val < df1_val and operation_str == 'min') \
                or (df2_val < df1_val and operation_str == 'before') or (df2_val > df1_val and operation_str == 'after'):
            df = self.get_root_answers(df2_str, step)
        else:
            df = create_null_result()

        return df

