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
Mim Question Representation

March 27, 2021
Authors: C3 Lab
"""

from typing import List, Dict
from anytree import LevelOrderIter

from .Step import Step
from mim_core.utils.result_utils import replace_ref_id, renumber_steps

class MQR(object):
    """
    A class representing the plan for answering a question and which contains the requisite steps and information.
    """

    def __init__(self,
                 question_text: str,
                 steps: List[Step],
                 q_id: str,
                 timing: Dict = None,
                 errors: List = None,
                 question_type: List[str] = None):
        self.question_text = question_text
        self.steps = steps
        self.q_id = q_id
        self.timing = timing if timing else {}
        self.errors = errors if errors else []
        self.question_type = question_type if question_text else []

    def __repr__(self):
        return 'MQR({})'.format(self.to_json())

    def to_json(self):
        return {
            "question_text": self.question_text,
            "question_type": self.question_type,
            "q_id": self.q_id,
            "steps": [s.to_json() for s in self.steps],
            "timing": self.timing,
            "errors": [str(e.__class__.__name__) + ": " + str(e) for e in self.errors]
        }

    def get_available_ref_id(self) -> int:
        """
        Returns the next highest available reference id.
        :return: A reference id that is not used by another step.
        """
        return len(self.steps) + 1

    def insert_step(self,
                    new_step: Step,
                    parent_step: Step,
                    child_step: Step) -> None:
        """
        Inserts the new step between the given parent and child step. If the 'remove_existing_link' flag is set,
        any links between the parent and child will be removed and re-routed through the new_step.
        :param new_step: The step to add between the parent and child steps.
        :param parent_step: The original parent step and also the step to become the parent of the new step.
        :param child_step:The original child step and also the step to become the child of the new step.
        :return: None
        """
        # TODO: Handle the case where parent step is None
        # TODO: Handle the case where child_step is None

        # Update the child's qdmr and operator args from parent.reference_id to new_step.reference_id
        child_step.qdmr = replace_ref_id(child_step.qdmr, parent_step.reference_id, new_step.reference_id)
        child_step.operator_args = [replace_ref_id(op, parent_step.reference_id, new_step.reference_id)
                                    for op in child_step.operator_args]
        child_step.question_text = replace_ref_id(child_step.question_text, parent_step.reference_id, new_step.reference_id)

        # Remove parent from child.parent_steps
        if parent_step in child_step.parent_steps:
            child_step.parent_steps.remove(parent_step)

        # Add new_step to child.parent_steps
        child_step.parent_steps.append(new_step)

        # Add child to new_step.child_steps
        new_step.child_steps.append(child_step)

        # Add parent to new_step.parent_steps
        new_step.parent_steps.append(parent_step)

        # Remove child from parent.child_steps
        if child_step in parent_step.child_steps:
            parent_step.child_steps.remove(child_step)

        # Add new_step to parent.child_steps
        parent_step.child_steps.append(new_step)

        # Add the new_step to the MQR's list of steps
        self.steps.append(new_step)

        return None

    def remove_step(self,
                    step: Step) -> None:
        """
        Removes the given step from the MQR. Children and parents of the removed step are then linked together and renumbered.
        :param step: The step to be removed from the MQR.
        :return: None
        """
        # TODO: For now, this is assuming there is a maximum of one parent and one child of the step to be removed

        # Get the parent step (if there is one)
        parent_step = step.parent_steps[0] if step.parent_steps else None

        # Get the child step (if there is one)
        child_step = step.child_steps[0] if step.child_steps else None

        if parent_step:
            # Remove this step from parent_step.child_steps
            parent_step.child_steps.remove(step)

        if child_step:
            # Update the QDMR and operator_args that reference the step to be removed
            child_step.qdmr = replace_ref_id(child_step.qdmr, step.reference_id, parent_step.reference_id)
            child_step.operator_args = [replace_ref_id(op, step.reference_id, parent_step.reference_id)
                                        for op in child_step.operator_args]
            child_step.question_text = replace_ref_id(child_step.question_text, step.reference_id, parent_step.reference_id)

            # Remove this step from child_step.parent_steps
            child_step.parent_steps.remove(step)

            if parent_step:
                # Add parent step as parent of the child step
                child_step.parent_steps.append(parent_step)

                # Add child step as child of the parent step
                parent_step.child_steps.append(child_step)


        # Remove this step from the MQR
        self.steps.remove(step)

        # Make sure all the steps are numbered properly
        renumber_steps(self)

        return None


class HierarchicalMQR(object):
    """
    A class representing the hierarchical plan for answering a question
    and which contains the requisite steps and information.
    """

    def __init__(self,
                 root: Step,
                 q_id: str,
                 timing: Dict = None,
                 errors: List = None):
        self.root = root
        self.q_id = q_id
        self.timing = timing if timing else {}
        self.errors = errors if errors else []

    def __repr__(self):
        return 'HierarchicalMQR({})'.format(self.to_json())

    def to_json(self):
        return {
            "q_id": self.q_id,
            "steps": [s.to_json() for s in LevelOrderIter(self.root)],
            "timing": self.timing,
            "errors": [str(e.__class__.__name__) + ": " + str(e) for e in self.errors]
        }