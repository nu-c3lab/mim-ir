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
QDMR Validator

June 29, 2021
Authors: C3 Lab
"""

from typing import List
from mim_core.structs.Step import Step
from mim_core.components.QuestionAnalysis.QuestionDecomposer import QuestionDecomposer


class QDMRValidator(object):
    """
    A class for ensuring generated QDMR is structurally valid.
    """

    def __init__(self):
        pass

    def validate(self,
                 question: str,
                 candidate_plans: List[List[Step]]) -> List[Step]:
        """
        Evaluates the candidate plans and returns the most structurally valid plan.
        :param question: The original question string from which this plan was generated.
        :param candidate_plans: A list of plans (list of Steps) to evaluate.
        :return: The most structurally valid plan.
        Note: It is assumed that the candidate plans are provided in order of decreasing probability.
        """

        plan_scores = []
        for plan in candidate_plans:
            # Parse the candidate plan to determine its operation type and arguments
            try:
                for s in plan:
                    s.update_operator_args_and_type()
            except:
                plan_scores.append(float('inf'))
                continue

            num_problems = 0
            num_problems += 1 if self.has_self_reference(plan) else 0
            num_problems += 1 if self.has_extra_terminal_steps(plan) else 0
            num_problems += 1 if self.has_repeated_steps(plan) else 0

            if num_problems == 0:
                return plan
            else:
                plan_scores.append(num_problems)

        # Return the plan with the fewest problems. If there's a tie, use the earliest plan
        #   (since it is assumed the candidates are provided in order of decreasing probability).
        min_score_index = plan_scores.index(min(plan_scores))
        return candidate_plans[min_score_index]

    def has_self_reference(self,
                           plan: List[Step]) -> bool:
        """
        Checks if there are steps in the plan that reference themselves.
        :param plan: The plan to validate.
        :return: A boolean denoting whether this plan has an steps that reference themselves.
        """
        return any(['@@'+str(s.reference_id)+'@@' in s.qdmr for s in plan])

    def has_extra_terminal_steps(self,
                                 plan: List[Step]) -> bool:
        """
        Checks if there are any extra leaf steps in the plan (i.e. any ignored steps)
        :param plan: The plan to validate.
        :return: A boolean denoting whether this plan has too many terminal steps.
        """
        # parents = get_all_parent_refs(plan[-1], plan)
        parents = plan[-1].get_ancestor_steps()
        return (len(parents) + 1) != len(plan)

    def has_repeated_steps(self,
                           plan: List[Step]) -> bool:
        """
        Checks if there are any steps that get repeated in the plan.
        :param plan: The plan to validate.
        :return: A boolean denoting whether this plan has repeated steps.
        """
        step_strings = set()
        for s in plan:
            if s.qdmr in step_strings:
                return True
            else:
                step_strings.add(s.qdmr)
        return False

    def has_ignored_input(self,
                          question: str,
                          steps: List[Step]) -> bool:
        # TODO: Implement this function
        return False
