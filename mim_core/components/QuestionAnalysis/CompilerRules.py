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
Compiler Rules

August 6, 2021
Authors: C3 Lab
"""

import re
from abc import ABC
from copy import deepcopy
from mim_core.structs.MQR import MQR
from mim_core.structs.Step import Step
from mim_core.utils.result_utils import replace_ref_id


class CompilerRule(ABC):
    def __init__(self):
        pass

    def preconditions_met(self, mqr: MQR) -> bool:
        pass

    def apply(self, mqr: MQR) -> bool:
        pass


class SelSelIntfilNonBoolean(CompilerRule):
    """
    A compiler rule transforming SELECT, SELECT, INT-FIL ---> SELECT, SELECT, PROJECT, PROJECT, INT-FIL
    Assumes that the expected answer type of the plan is NOT boolean.
    """

    def __init__(self):
        super().__init__()
        self.inter_step = None

    def preconditions_met(self,
                          mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """

        # Get the intersection-filter step in the plan, if there is one
        intersection_filter_steps = [s for s in mqr.steps if s.operator_subtype == 'intersection-filter']
        if len(intersection_filter_steps) == 0:
            return False

        # Ensure that the expected answer type for the input question is not a boolean
        if 'boolean' in mqr.question_type:
            return False

        # Store the intersection-filter step for later use
        self.inter_step = intersection_filter_steps[0]

        # Ensure that there are only two parents of the intersection-filter step
        if len(self.inter_step.parent_steps) != 2:
            return False

        # Check if the two parents of the intersection-filter operation are both select operations
        parent_set = set([s.operator_type for s in self.inter_step.parent_steps])
        if parent_set == {'select'}:
            return True

        return False

    def apply(self,
              mqr: MQR) -> bool:
        """
        Inserts an additional two projects after the two parent select steps.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            step = self.inter_step
            select_one = step.parent_steps[0]
            select_two = step.parent_steps[1]

            # Create and insert the two project steps
            new_reference_id_one = mqr.get_available_ref_id()
            new_qdmr_one = '{} of {}'.format(step.operator_args[0], '@@' + str(select_one.reference_id) + '@@')
            new_step_one = Step(new_qdmr_one, new_reference_id_one, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_one, select_one, step)

            new_reference_id_two = mqr.get_available_ref_id()
            new_qdmr_two = '{} of {}'.format(step.operator_args[0], '@@' + str(select_two.reference_id) + '@@')
            new_step_two = Step(new_qdmr_two, new_reference_id_two, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_two, select_two, step)

            return True
        else:
            return False


class SelSelIntfilBoolean(CompilerRule):
    """
    A compiler rule transforming SELECT, SELECT, INT-FIL ---> SELECT, SELECT, PROJECT, PROJECT, INT-FIL
    Assumes that the expected answer type of the plan IS boolean.
    """

    def __init__(self):
        super().__init__()
        self.inter_step = None

    def preconditions_met(self,
                          mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """

        # Get the intersection-filter step in the plan, if there is one
        intersection_filter_steps = [s for s in mqr.steps if s.operator_subtype == 'intersection-filter']
        if len(intersection_filter_steps) == 0:
            return False

        # Ensure that the expected answer type for the input question is not a boolean
        if 'boolean' not in mqr.question_type:
            return False

        # Store the intersection-filter step for later use
        self.inter_step = intersection_filter_steps[0]

        # Ensure that there are only two parents of the intersection-filter step
        if len(self.inter_step.parent_steps) != 2:
            return False

        # Check if the two parents of the intersection-filter operation are both select operations
        parent_set = set([s.operator_type for s in self.inter_step.parent_steps])
        if parent_set == {'select'}:
            return True

        return False

    def apply(self,
              mqr: MQR) -> bool:
        """
        Inserts an additional two filters after the two parent select steps.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            step = self.inter_step
            select_one = step.parent_steps[0]
            select_two = step.parent_steps[1]

            # Create and insert the two filter steps
            new_reference_id_one = mqr.get_available_ref_id()
            new_qdmr_one = '{} that are {}'.format('@@' + str(select_one.reference_id) + '@@', step.operator_args[0])
            new_step_one = Step(new_qdmr_one, new_reference_id_one, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_one, select_one, step)

            new_reference_id_two = mqr.get_available_ref_id()
            new_qdmr_two = '{} that are {}'.format('@@' + str(select_two.reference_id) + '@@', step.operator_args[0])
            new_step_two = Step(new_qdmr_two, new_reference_id_two, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_two, select_two, step)

            return True
        else:
            return False


class SelSelProjIntfil(CompilerRule):
    """
    A compiler rule transforming SELECT, SELECT, PROJECT, INT-FIL ---> SELECT, SELECT, PROJECT, PROJECT, INT-FIL
    """

    def __init__(self):
        super().__init__()
        self.inter_step = None

    def preconditions_met(self,
                          mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """

        # Get the intersection-filter step in the plan, if there is one
        intersection_filter_steps = [s for s in mqr.steps if s.operator_subtype == 'intersection-filter']
        if len(intersection_filter_steps) == 0:
            return False

        # Store the intersection-filter step for later use
        self.inter_step = intersection_filter_steps[0]

        # Ensure that there are only two parents of the intersection-filter step
        if len(self.inter_step.parent_steps) != 2:
            return False

        # Check if the two parents of the intersection-filter operation are select and project operations
        parent_set = set([s.operator_type for s in self.inter_step.parent_steps])
        if parent_set == {'select', 'project'}:
            return True

        return False

    def apply(self,
              mqr: MQR) -> bool:
        """
        Inserts an additional project after the parent select step.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            step = self.inter_step
            select_parent = step.parent_steps[0] if step.parent_steps[0].operator_type == 'select' else step.parent_steps[1]
            project_parent = step.parent_steps[0] if step.parent_steps[0].operator_type == 'project' else step.parent_steps[1]

            # Make a new project step with no references/connections to other nodes
            new_reference_id = mqr.get_available_ref_id()
            new_qdmr = replace_ref_id(project_parent.qdmr, project_parent.parent_steps[0].reference_id,select_parent.reference_id)
            new_operator_args = [replace_ref_id(op, project_parent.parent_steps[0].reference_id, select_parent.reference_id)
                                 for op in project_parent.operator_args]
            new_project_step = Step(qdmr=new_qdmr,
                                    reference_id=new_reference_id,
                                    q_id = step.q_id,
                                    step_type = step.step_type,
                                    operator_type=project_parent.operator_type,
                                    operator_args=new_operator_args,
                                    entities=deepcopy(project_parent.entities),
                                    relationship=project_parent.relationship,
                                    question_text=project_parent.question_text,
                                    expected_answer_type=deepcopy(project_parent.expected_answer_type),
                                    operator_subtype=project_parent.operator_subtype,
                                    parent_steps=None,
                                    child_steps=None,
                                    is_built=True)

            # Insert the new project step into the plan between the parent and child
            mqr.insert_step(new_project_step, select_parent, step)

            return True
        else:
            return False


class SelSelFilIntfil(CompilerRule):
    """
    A compiler rule transforming SELECT, SELECT, FILTER, INT-FIL ---> SELECT, SELECT, FILTER, FILTER, INT-FIL
    """

    def __init__(self):
        super().__init__()
        self.inter_step = None

    def preconditions_met(self,
                          mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """

        # Get the intersection-filter step in the plan, if there is one
        intersection_filter_steps = [s for s in mqr.steps if s.operator_subtype == 'intersection-filter']
        if len(intersection_filter_steps) == 0:
            return False

        # Store the intersection-filter step for later use
        self.inter_step = intersection_filter_steps[0]

        # Ensure that there are only two parents of the intersection-filter step
        if len(self.inter_step.parent_steps) != 2:
            return False

        # Check if the two parents of the intersection-filter operation are select and filter operations
        parent_set = set([s.operator_type for s in self.inter_step.parent_steps])
        if parent_set == {'select', 'filter'}:
            return True

        return False

    def apply(self,
              mqr: MQR) -> bool:
        """
        Inserts an additional filter after the parent select step.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            step = self.inter_step
            select_parent = step.parent_steps[0] if step.parent_steps[0].operator_type == 'select' else step.parent_steps[1]
            filter_parent = step.parent_steps[0] if step.parent_steps[0].operator_type == 'filter' else step.parent_steps[1]

            # Make a new project step with no references/connections to other nodes
            new_reference_id = mqr.get_available_ref_id()
            new_qdmr = replace_ref_id(filter_parent.qdmr, filter_parent.parent_steps[0].reference_id, select_parent.reference_id)
            new_operator_args = [replace_ref_id(op, filter_parent.parent_steps[0].reference_id, select_parent.reference_id) for op in filter_parent.operator_args]
            new_filter_step = Step(qdmr=new_qdmr,
                                   reference_id=new_reference_id,
                                   q_id = step.q_id,
                                   step_type = step.step_type,
                                   operator_type=filter_parent.operator_type,
                                   operator_args=new_operator_args,
                                   entities=deepcopy(filter_parent.entities),
                                   relationship=filter_parent.relationship,
                                   question_text=filter_parent.question_text,
                                   expected_answer_type=deepcopy(filter_parent.expected_answer_type),
                                   operator_subtype=filter_parent.operator_subtype,
                                   parent_steps=None,
                                   child_steps=None)

            # Insert the new filter step into the plan between the parent and child
            mqr.insert_step(new_filter_step, select_parent, step)

            return True
        else:
            return False


class FilFilIntfilBoolean(CompilerRule):
    """
    A compiler rule transforming FILTER, FILTER, INT-FIL ---> FILTER, BOOL-EXIST, FILTER, BOOL-EXIST, BOOL-TRUTH-EVAL
    Assumes that the expected answer type of the plan IS boolean.
    """

    def __init__(self):
        super().__init__()
        self.inter_step = None

    def preconditions_met(self,
                          mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """

        # Get the intersection-filter step in the plan, if there is one
        intersection_filter_steps = [s for s in mqr.steps if s.operator_subtype == 'intersection-filter']
        if len(intersection_filter_steps) == 0:
            return False

        # Ensure that the expected answer type for the input question is not a boolean
        if 'boolean' not in mqr.question_type:
            return False

        # Store the intersection-filter step for later use
        self.inter_step = intersection_filter_steps[0]

        # Ensure that there are only two parents of the intersection-filter step
        if len(self.inter_step.parent_steps) != 2:
            return False

        # Check if the two parents of the intersection-filter operation are select and filter operations
        parent_set = set([s.operator_type for s in self.inter_step.parent_steps])
        if parent_set == {'filter'}:
            return True

        return False

    def apply(self,
              mqr: MQR) -> bool:
        """
        Inserts an additional boolean-exist steps after the filters and
        changes the intersection-filter step to a boolean-truth-eval step.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            step = self.inter_step
            filter_one = step.parent_steps[0]
            filter_two = step.parent_steps[1]

            # Create and insert the two boolean existence steps
            new_reference_id_one = mqr.get_available_ref_id()
            new_qdmr_one = 'is there any {}'.format('@@' + str(filter_one.reference_id) + '@@')
            new_step_one = Step(new_qdmr_one, new_reference_id_one, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_one, filter_one, step)

            new_reference_id_two = mqr.get_available_ref_id()
            new_qdmr_two = 'is there any {}'.format('@@' + str(filter_two.reference_id) + '@@')
            new_step_two = Step(new_qdmr_two, new_reference_id_two, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_two, filter_two, step)

            # Reconfigure the intersection-filter step to be boolean-truth-eval
            step.qdmr = "is both {} and {} true".format('@@' + str(new_reference_id_one) + '@@',
                                                        '@@' + str(new_reference_id_two) + '@@')
            step.is_built = False

            return True
        else:
            return False


class SelBooleanSearch(CompilerRule):
    """
    A compiler rule transforming SELECT, BOOLEAN-SEARCH ---> SELECT, FILTER, BOOLEAN-EXISTENCE
    """

    def __init__(self):
        super().__init__()
        self.boolean_step = None

    def preconditions_met(self,
                          mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """

        # Get the boolean-search step in the plan, if there is one
        boolean_search_steps = [s for s in mqr.steps if s.operator_subtype == 'boolean-search']
        if len(boolean_search_steps) == 0:
            return False

        # Store the intersection-filter step for later use
        self.boolean_step = boolean_search_steps[0]

        # Ensure that there is only one parents of the boolean-search step
        if len(self.boolean_step.parent_steps) != 1:
            return False

        # Check if the parent of the intersection-filter operation is a select operation
        parent_set = set([s.operator_type for s in self.boolean_step.parent_steps])
        if parent_set == {'select'}:
            return True

        return False

    def apply(self,
              mqr: MQR) -> bool:
        """
        Inserts a filter step after the parent select operation and
        changes the boolean-search operation to boolean-existence.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            step = self.boolean_step
            select_one = step.parent_steps[0]

            # Create and insert the filter step
            new_reference_id_one = mqr.get_available_ref_id()
            new_qdmr_one = '{} that are {}'.format('@@' + str(select_one.reference_id) + '@@', step.operator_args[1])
            new_step_one = Step(new_qdmr_one, new_reference_id_one, q_id=step.q_id, step_type=step.step_type)
            mqr.insert_step(new_step_one, select_one, step)

            # Modify the boolean-search step to be a boolean-existence step
            step.qdmr = "is there any {}".format('@@' + str(new_reference_id_one) + '@@')
            step.is_built = False

            return True
        else:
            return False

class RemoveNameOfProjects(CompilerRule):
    """
    A compiler rule removing useless "name of #1" project operations.
    """

    def __init__(self):
        super().__init__()
        self.name_of_steps = []

    def preconditions_met(self, mqr: MQR) -> bool:
        """
        Determines if the given MQR/plan has the proper structure and preconditions in order to apply this rule.
        :param mqr: The MQR/plan to check the structure of.
        :return: A boolean denoting whether the MQR/plan meets the preconditions.
        """
        # Get the project steps in the plan, if there are any
        project_steps = [s for s in mqr.steps if s.operator_type == 'project']
        if len(project_steps) == 0:
            return False

        # Check if the project step has the form "name of @@<ref>@@" or "the name of @@<ref>@@"
        self.name_of_steps = [s for s in project_steps if re.search('^(the )?name of @@(\d+)@@', s.qdmr)]
        if len(self.name_of_steps) == 0:
            return False
        else:
            return True

    def apply(self, mqr: MQR) -> bool:
        """
        Removes the "name of" steps.
        :param mqr: The MQR/plan to which to apply this rule.
        :return: A boolean denoting whether compilation/updates were performed to the plan.
        """

        if self.preconditions_met(mqr):
            # Remove the name_of_steps
            for s in self.name_of_steps:
                mqr.remove_step(s)

            return True
        else:
            return False