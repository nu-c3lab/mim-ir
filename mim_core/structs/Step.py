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
Step

March 27, 2021
Authors: C3 Lab
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict
from anytree import NodeMixin
from anytree.util import leftsibling

from mim_core.utils.result_utils import create_null_result, get_parent_step_refs, is_arg_reference, \
                                        is_arg_parent_reference, is_null_result
from mim_core.components.QuestionAnalysis.QDMRParser import QDMRProgramBuilder

class Step(NodeMixin):
    """
    A class for representing one of the steps to be carried out in order to answer the question.
    """

    def __init__(self,
                 qdmr: str,
                 reference_id: int,
                 q_id: str,
                 step_type: str,
                 operator_type: str = None,
                 operator_args: List[str] = None,
                 entities: Dict[str, str] = None,
                 entity_class: str = None,
                 relationship: str = None,
                 question_text: str = None,
                 expected_answer_type: List[str] = None,
                 result: pd.DataFrame = None,
                 timing: Dict = None,
                 errors: List[str] = None,
                 misc: Dict = None,
                 operator_subtype: str = None,
                 parent_steps: List['Step'] = None,
                 child_steps: List['Step'] = None,
                 is_built: bool = False):
        super().__init__()
        self.qdmr = qdmr
        self.q_id = q_id
        self.step_type = step_type
        self.reference_id = reference_id
        self.operator_type = operator_type
        self.operator_args = operator_args
        self.entities = entities
        self.entity_class = entity_class
        self.relationship = relationship
        self.question_text = question_text
        self.expected_answer_type = expected_answer_type
        self.result = result if result is not None else create_null_result()
        self.timing = timing if timing else {}
        self.errors = errors if errors else []
        self.misc = misc if misc else {}
        self.operator_subtype = operator_subtype
        self.parent_steps = parent_steps if parent_steps else []
        self.child_steps = child_steps if child_steps else []
        self.is_built = is_built

    def __repr__(self):
        return 'Step({})'.format(self.to_json())

    def to_json(self):
        return {
            "qdmr": self.qdmr,
            "q_id": self.q_id,
            "step_type": self.step_type,
            "reference_id": self.reference_id,
            "operator_type": self.operator_type,
            "operator_subtype": self.operator_subtype,
            "operator_args": self.operator_args,
            "entities": self.entities,
            "entity_class": self.entity_class,
            "question_text": self.question_text,
            "relationship": self.relationship,
            "expected_answer_type": self.expected_answer_type,
            "parent_steps": [p.reference_id for p in self.parent_steps],
            "child_steps": [c.reference_id for c in self.child_steps],
            "result": self.result.to_json(default_handler=str),
            "timing": self.timing,
            "errors": [str(e.__class__.__name__) + ": " + str(e) for e in self.errors],
            "misc": self.misc,
            "is_built": self.is_built
        }

    def update_operator_args_and_type(self) -> None:
        """
        Determines the operator args and type for this step based on the current QDMR string.
        :return: None
        """
        # Classify the operator type and extract arguments (using BREAK paper's open-source solution)
        self.operator_type, orig_operator_args = self._extract_operators(self.qdmr)

        # Perform some post-processing on the operator args to get them into the desired form
        self.operator_args = self._clean_operator_args(orig_operator_args)

    def _extract_operators(self,
                           qdmr_step: str) -> (str, List[str]):
        """
        Classifies the operator type and extracts operator arguments from a single qdmr line.
        :param qdmr_step: A single line of the qdmr decomposition
        """
        qdmr_step = 'return ' + re.sub(r'@@\d*@@', prediction_to_parse_form, qdmr_step)

        parse = QDMRProgramBuilder(qdmr_step)
        parse.build()

        return parse.operators[0], parse.steps[0].arguments

    def _clean_operator_args(self,
                             operator_args: list) -> List[str]:
        """
        Replaces #REF with the proper arg number it refers to. Converts parse form back to the reference tokens.
        :param operator_args: List of QDMR operator args.
        :return: A list of QDMR operator args in the form needed for the plan.
        """
        # TODO: This is 100% a hack to ensure the operator args work for the demo. Figure out how to make this function not needed.
        #       This method currently assumes #REF is always succeeded by a valid reference number in the list.

        # '#REF' not in operator_args[i]
        def is_only_reference_num(s: str) -> bool:
            return bool(re.search('^#\d+$', s))

        resolved_args = []
        for i in range(0, len(operator_args)):
            if '#REF' in operator_args[i]:
                pass
            elif is_only_reference_num(operator_args[i]) and '#REF' in operator_args[i-1]:
                # resolved_args.append(operator_args[i-1].replace('#REF', operator_args[i]))
                resolved_args.append(operator_args[i])
            else:
                resolved_args.append(operator_args[i])

        # Convert parse form back to the reference token form
        cleaned_args = []
        for arg in resolved_args:
            cleaned_args.append(re.sub(r'#\d+', parse_to_prediction_form, arg))

        return cleaned_args

    def get_all_prior_steps(self) -> List['Step']:
        """
        Returns all prior steps in the current level of decomposition.
        :return: A list of all Steps that occur prior to this Step.
        """

        left_sibling = leftsibling(self)
        all_prior_steps = []
        while left_sibling:
            all_prior_steps.append(left_sibling)
            left_sibling = leftsibling(left_sibling)
        return all_prior_steps

    def get_parent_steps(self) -> List['Step']:
        """
        Returns the parent steps (steps referenced by the current step) in the current level of the decomposition.
        :return: A list of Steps that are the immediate parents of the current step.
        """

        # Get the reference IDs for the parent steps (one level up)
        parent_ids = [int(x) for x in get_parent_step_refs(self)]

        # Get the prior steps
        all_prior_steps = self.get_all_prior_steps()

        # Retrieve the steps that have the matching reference IDs
        return [x for x in all_prior_steps if x.reference_id in parent_ids]

    def get_ancestor_steps(self) -> List['Step']:
        """
        Return all direct ancestor steps (steps referenced by the current step and their ancestors) in the current level of decomposition.
        :return: A list of Steps that are ancestors of the current Step. Does not necessarily return all prior steps.
        """

        ancestors = []
        not_visited = self.get_parent_steps()
        while len(not_visited) > 0:
            # Get the next parent to visit and mark it as visited (i.e. it is an ancestor)
            p_node = not_visited.pop(0)
            ancestors.append(p_node)

            # Get the parents of this node and mark them as candidates to visit if they have not yet been visited
            new_parents = p_node.get_parent_steps()
            for np in new_parents:
                if np not in not_visited and np not in ancestors:
                    not_visited.append(np)

        return ancestors

    def get_all_results(self) -> pd.DataFrame:
        """
        Get the results for this dataframe concatenated with the results from the rightmost children (recursively applied).
        IMPORTANT: Does NOT put the returned DataFrame in the form required for concatenation with its parent's result DataFrame.
        :return: A DataFrame consisting of all results for this node and its children.
        """

        if self.is_leaf:
            # If the results are null, make sure ALL the reference columns are specified for this step (e.g. @@2@@, parent_2)
            if is_null_result(self.result):
                step_str = self.qdmr
                prev_step_refs = []
                while is_arg_parent_reference(step_str):
                    ref_num_str = is_arg_parent_reference(step_str)
                    prior_step_key = "parent_" + ref_num_str
                    prev_step_refs.append(prior_step_key)
                    step_str = step_str.replace(prior_step_key, "")

                while is_arg_reference(step_str):
                    ref_num_str = is_arg_reference(step_str)
                    prior_step_key = "@@" + ref_num_str + "@@"
                    prev_step_refs.append(prior_step_key)
                    step_str = step_str.replace(prior_step_key, "")

                # Add the references to past steps the DataFrame as NaN columns
                for c in prev_step_refs:
                    self.result[c] = np.nan

            return self.result
        else:
            # Get the rightmost child
            rc = list(self.children)[-1]

            # Get all results for the rightmost child in the form of the current node
            child_results = rc.get_all_results_for_parent()

            # Concatenate the results from the child nodes with this node's results
            all_results = pd.concat([self.result, child_results],
                                    ignore_index=True).drop_duplicates().reset_index(drop=True)

            # If there is more than one row in the results, remove rows where answer column has nan
            if len(all_results) > 1:
                all_results = all_results.dropna(subset=['answer'])

            # Reset the id column
            all_results['id'] = all_results.index

            # Return all results
            return all_results

    def get_all_results_for_parent(self) -> pd.DataFrame:
        """
        Get the results for this dataframe concatenated with the results from the rightmost children (recursively applied).
        IMPORTANT: Puts the returned DataFrame in the form required for concatenation with its parent's result DataFrame.
        :return: A DataFrame consisting of all results for this node and its children.
        """

        # Get all of the results for this node
        all_results = self.get_all_results()

        # If the parent node has a references to a prior step (and is not an aggregate or arithmetic operation),  correlate the answers in this step with those parent references
        if is_arg_reference(self.parent.qdmr) and self.operator_type not in ['aggregate', 'arithmetic']:
            ancestors = self.get_ancestor_steps()
            for a in ancestors:
                df_str = "@@" + str(a.reference_id) + "@@"
                a_df = a.get_all_results().rename(columns={"id": df_str})
                all_results = all_results.merge(a_df, on=df_str, how='left', suffixes=('', "_DROP"))
                all_results.drop(all_results.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)

        # Drop any columns that are not in [id, answer, source, parent_X] (so we can concatenate this node's results with its parent's results)
        cols_to_keep = [col for col in all_results.columns if is_arg_parent_reference(col)] + ['id', 'answer', 'confidence', 'source']
        all_results = all_results[cols_to_keep]

        # Rename "parent_X" columns as "@@X@@"
        for col in all_results.columns:
            parent_ref = is_arg_parent_reference(col)
            if parent_ref:
                all_results = all_results.rename(columns={col: "@@"+parent_ref+"@@"})

        # Return all results
        return all_results

    def get_expected_type(self) -> str:
        """
        Get the expected type of the question by first checking the root, then its rightmost child,
        the the rightmost child of that child. This method assumes that the hierarchy has a depth of
        no more than 2.
        :return: The question's expected type.
        """

        if not self.expected_answer_type and self.children:
            if not self.children[-1].expected_answer_type and self.children[-1].children:
                return self.children[-1].children[-1].expected_answer_type
            return self.children[-1].expected_answer_type
        return self.expected_answer_type

    def is_low_level_retrieval_step(self) -> bool:
        """
        Checks if this step is intended to retrieve evidence from a knowledge source for low-level QA.
        :return: A boolean denoting whether this step is a low-level retrieval step.
        """

        return self.operator_type.lower() in ['select', 'project', 'filter']

    def is_high_level_retrieval_step(self) -> bool:
        """
        Checks if this step is intended to retrieve evidence from a knowledge source for high-level QA.
        :return: A boolean denoting whether this step is a high-level retrieval step.
        """

        return self.operator_type.lower() in ['select', 'project', 'filter'] \
               or self.operator_subtype.lower() in ['boolean-search', 'aggregate']


def prediction_to_parse_form(matchobj) -> str:
    """
    Helper function for converting reference tokens in model predictions to form for argument parser.
    E.g. @@1@@ -> #1
    :param matchobj: input from regex substitution call
    """
    return '#' + matchobj.group(0).replace('@@', '')

def parse_to_prediction_form(matchobj) -> str:
    """
    Helper function for converting reference tokens in model predictions to form for argument parser.
    E.g. #1 -> @@1@@
    Note: This is the inverse of prediction_to_parse_form()
    :param matchobj: input from regex substitution call
    """
    return '@@' + matchobj.group(0).replace('#', '') + '@@'
