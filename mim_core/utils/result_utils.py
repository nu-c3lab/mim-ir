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
Result Utility Functions

March 27, 2021
Authors: C3 Lab
"""

import re
import requests
import numpy as np
import pandas as pd
from typing import Dict, List
from operator import attrgetter
import gdown
import mim_core.structs as s
from mim_core.exceptions import MissingPreviousStepDataError

def create_null_result():
    """
    Creates and returns a default Pandas DataFrame containing a null answer with a standardized form.
    :return: A Pandas DataFrame containing a null answer.
    """

    df = pd.DataFrame({'id': 0,
                        'answer': np.nan,
                        'source': None,
                        'confidence': 0.0,
                        'answer_confidence': np.nan,
                        'doc_retrieval_confidence': np.nan,
                        'question_string': None},
                      index=[0])
    return df


def is_null_result(result: pd.DataFrame) -> bool:
    """
    Checks if the result dataframe contains only null answers.
    :param result: The dataframe of results.
    :return: Boolean denoting whether the result dataframe contains only null answers.
    """
    return result['answer'].isnull().all()


def download_file_from_google_drive(id: str,
                                    destination: str) -> None:
    """
    Downloads a file from Google drive at the given URL to the given destination.
    :param id: The ID of the file to download.
    :param destination: The full path to the destination where the file will be downloaded.
    :return: None
    """

    url = "https://drive.google.com/uc?id=" + id
    gdown.download(url, str(destination), quiet=False)
    return None


def is_arg_reference(step_arg: str) -> str:
    """
    Determines if the given step references a prior decomposition step.
    :param step_arg:
    :return: Str denoting the step number, or None, if the step doesn't reference a prior step.
    Note: Assumes that the step reference will be enclosed by two pairs of "@@" (e.g. step 23 -> "@@23@@")
    """
    m = re.search(r'@@\d+@@', step_arg)
    if m:
        return m.group().strip('@')
    return None


def get_parent_step_refs(step) -> List[str]:
    """
    Finds the direct parent steps (i.e. one level up) of the given step.
    :param step: The step to find the parent references for.
    :return: A list of the parent step numbers for the given step.
    """
    refs = [is_arg_reference(x) for x in step.operator_args]
    return [x for x in refs if x is not None]


def get_prev_step_answers_new(step) -> Dict:
    """
    Same as get_prev_step_answers() but without the need to use mqr. Makes use of the new Step/MQR graph structure.
    :param step: The step for which to gather past DataFrames.
    :return: A dictionary containing the answers to previous steps.
    """

    # TODO: Rename this function
    prior_step_answers = {}

    # Retrieve dataframes for steps in the current level of the plan
    for s in step.parent_steps:
        # Retrieve the previous data frame corresponding to the current parent step
        prior_step_key = "@@" + str(s.reference_id) + "@@"
        prev_df = s.get_all_results()          # TODO: Change this to s.get_all_results()? (from s.result)

        # Make sure the retrieved dataframe is not a null result
        if is_null_result(prev_df):
            raise MissingPreviousStepDataError()

        # Filter out unneeded columns (i.e. only keep id, answer, and any references to previous steps)
        cols_to_keep = [c for c in prev_df.columns if is_arg_reference(c) or c in ['id', 'answer', 'confidence', 'wikidata_id']]
        prior_step_answers[prior_step_key] = prev_df[cols_to_keep]

    # Retrieve dataframes for steps in the parent level of the plan
    step_str = step.qdmr
    while is_arg_parent_reference(step_str):
        ref_num_str = is_arg_parent_reference(step_str)
        prior_step_key = "parent_" + ref_num_str

        parent_level_ancestors = step.parent.get_ancestor_steps()
        parent_ref_step = next(filter(lambda x: x.reference_id == int(ref_num_str), parent_level_ancestors))
        prev_df = parent_ref_step.get_all_results()

        # Make sure the retrieved dataframe is not a null result
        if is_null_result(prev_df):
            raise MissingPreviousStepDataError()

        # Filter out unneeded columns (i.e. only keep id, answer)
        cols_to_keep = [c for c in prev_df.columns if c in ['id', 'answer', 'confidence', 'wikidata_id']]
        prior_step_answers[prior_step_key] = prev_df[cols_to_keep]

        # Remove the current parent reference from step_str and continue checking for other parent references
        step_str = step_str.replace(prior_step_key, "")

    return prior_step_answers


def replace_ref_id(string: str,
                   old_id: int,
                   new_id: int) -> str:
    """
    Builds a string with the old reference id replaced by the new one.
    :param string: The string in which the replacing will occur (this string is not modified).
    :param old_id: The old step reference id number.
    :param new_id: The new step reference id number.
    :return: The newly built string.
    """
    return string.replace('@@' + str(old_id) + '@@', '@@' + str(new_id) + '@@')


def renumber_steps(mqr) -> None:
    """
    Use BFS to update the reference ids of the steps in the MQR.
    :return: None
    """

    def bfs(queue: List['Step']) -> List['Step']:
        visited = []
        while queue:
            s = queue.pop(0)
            visited.append(s)
            for c in s.child_steps:
                if c not in queue and c not in visited:
                    queue.append(c)
        return visited

    not_visited = [s.reference_id for s in mqr.steps]
    visited = []
    initial_queue = [s for s in mqr.steps if not s.parent_steps]
    while len(not_visited) > 0:
        # Initialize the queue (list of nodes to start searching from)
        if initial_queue:
            queue = initial_queue
        else:
            queue = [s for s in mqr.steps if s.reference_id == not_visited[0]]

        # Traverse the nodes
        temp_visited = bfs(queue)

        # Update lists that are tracking visited and unvisited nodes
        for s in temp_visited:
            if s.reference_id not in visited:
                visited.append(s.reference_id)
            if s.reference_id in not_visited:
                not_visited.remove(s.reference_id)

    # Create the dictionary used to track which steps need to be renumbered
    renumbering = {s: i+1 for i, s in enumerate(visited)}

    # 1st Pass: Rename to correct reference number, but with different enclosing symbols (i.e. $$<num>$$).
    for key, value in renumbering.items():
        # During this pass, only replace strings (i.e. qdmr, operator_args) and NOT reference_id
        if key != value:
            # Update the step denoted by the key with the value AND update its children steps
            step = [s for s in mqr.steps if s.reference_id == key][0]
            for c in step.child_steps:
                c.qdmr = c.qdmr.replace('@@' + str(step.reference_id) + '@@', '$$$' + str(value) + '$$$')
                c.operator_args = [op.replace('@@' + str(step.reference_id) + '@@', '$$$' + str(value) + '$$$') for op in c.operator_args]

    # 2nd Pass: Rename to correct reference number, with final (and correct) enclosing symbols (i.e. @@<num>@@)
    for key, value in renumbering.items():
        if key != value:
            # Update the step denoted by the key with the value AND update its children steps
            step = [s for s in mqr.steps if s.reference_id == key][0]
            for c in step.child_steps:
                c.qdmr = c.qdmr.replace('$$$' + str(value) + '$$$', '@@' + str(value) + '@@')
                c.operator_args = [op.replace('$$$' + str(value) + '$$$', '@@' + str(value) + '@@') for op in c.operator_args]
            step.reference_id = value

    # Sort the steps according to their reference number
    mqr.steps.sort(key=attrgetter('reference_id'),reverse=False)

    return None


def is_arg_parent_reference(step_arg: str) -> str:
    """
    Determines if the given step references a step from its parent.
    :param step_arg:
    :return: Str denoting the step number, or None, if the step doesn't reference a prior step.
    """
    m = re.search(r'parent_(\d+)', step_arg)
    if m:
        return m.group(1)
    return None


def is_simple_retrieval_question(mqr) -> bool:
    return len(mqr.root.children) == 1 \
           and (mqr.root.children[0].operator_type in ['select', 'project', 'filter', 'aggregate']
                or mqr.root.children[0].operator_subtype.lower() in ['boolean-search'])

def is_complex_retrieval_question(mqr) -> bool:
    return len(mqr.root.children) > 1 \
           and not any([is_analysis_step(x) for x in mqr.root.children])

def is_analysis_step(step) -> bool:
    return step.operator_type in ['arithmetic', 'group', 'superlative', 'union', 'comparative',
                                  'comparison', 'intersection', 'discard', 'sort', 'aggregate'] \
        or step.operator_subtype in ['boolean', 'boolean-existence', 'boolean-inequality']

def is_explanation_question(mqr) -> bool:
    return mqr.root.expected_answer_type[0].lower() in ['explanation', 'description']

def is_set_operation_step(step) -> bool:
    return step.operator_type in ['intersection', 'union', 'discard']
