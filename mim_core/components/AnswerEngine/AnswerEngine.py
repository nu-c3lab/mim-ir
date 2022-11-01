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
Answer Engine

March 28, 2021
Authors: C3 Lab
"""

import pandas as pd
from time import time
from typing import Dict
from mim_core.structs.Step import Step
from mim_core.structs.MQR import HierarchicalMQR
from mim_core.components.AnswerEngine.AnalyticsEngine.AnalyticsEngine import AnalyticsEngine
from mim_core.components.Search.HighLevelQAComponent import HighLevelQAComponent
from mim_core.components.Search.HybridAnsweringComponent import HybridAnsweringComponent
import mim_core.utils.component_loading as cl_utils
from mim_core.utils.result_utils import is_explanation_question, is_simple_retrieval_question, is_complex_retrieval_question
from mim_core.exceptions import *
from anytree import PreOrderIter


class AnswerEngine(object):
    """
    A class that executes the plan specified by the MQR by searching
    document sources and knowledge graphs, analyzing data, and evaluating answers.
    """

    def __init__(self, **kwargs):
        self.answer_evaluator = cl_utils.load_answer_evaluator(kwargs.get('answer_evaluator'))
        self.analytics_engine = AnalyticsEngine()
        self.question_answering_components = [cl_utils.load_question_answering_component(qac) for qac in kwargs.get('question_answering_components', [])]
        self.high_level_question_answering_components = [c for c in self.question_answering_components if isinstance(c, HighLevelQAComponent)]
        self.complex_question_answering_components = [c for c in self.question_answering_components if isinstance(c, HybridAnsweringComponent)]
        self.explanation_question_answering_components = []
        for qac in self.question_answering_components:
            qac.analytics_engine = self.analytics_engine
        self.answerable_question_types = kwargs.get('answerable_question_types', ['simple-retrieval', 'complex-retrieval', 'analysis', 'explanation'])

    def execute_plan(self,
                     mqr: HierarchicalMQR) -> HierarchicalMQR:
        """
        Uses the plan specified by the MQR to answer a question.
        :param mqr: The question answering plan representation.
        :return: An executed plan/MQR.
        """

        # Initialize timing fields
        if mqr.timing:
            if "answer_engine" not in mqr.timing:
                mqr.timing["answer_engine"] = {}
            if "answer_evaluator" not in mqr.timing["answer_engine"]:
                mqr.timing["answer_engine"]["answer_evaluator"] = {"total": 0}
            for answerer in self.high_level_question_answering_components:
                mqr.timing["answer_engine"][str(answerer.__class__.__name__)] = {"total": 0}
            if "answer_merging" not in mqr.timing["answer_engine"]:
                mqr.timing["answer_engine"]["answer_merging"] = 0

        # Start timing the answer engine's execution of the MQR
        total_start_time = time()

        # Check for and handle errors coming from the QuestionAnalyzer
        for e in mqr.errors:
            if type(e) in [DecompositionPredictionError, DecompositionTypeParseError]:
                # Clean up and return to Interaction Manager
                # mqr.timing["answer_engine"]["total"] = time() - total_start_time
                # return mqr
                continue
            elif type(e) in [DecompositionReferenceError, DecompositionLeafError,
                             DecompositionTypeIOError, DecompositionSeparatorError]:
                # TODO: Add more complex logic for level 1 and 2 errors
                continue
            else:
                # There was an unexpected error
                # mqr.timing["answer_engine"]["total"] = time() - total_start_time
                # return mqr
                continue

        # Route explanation / description questions to separate answering mechanism than factoid/analysis questions
        if not self.is_answerable_question(mqr):
            pass
        elif mqr.root.expected_answer_type and mqr.root.expected_answer_type[0].lower() in ['description', 'explanation'] and self.explanation_question_answering_components:
            # Answer the explanation / description question
            self.answer_explanation_question(mqr.root, mqr.timing)
        else:
            # Find answers to each of the steps in the MQR/plan for factoid/retrieval/analysis questions
            for step in PreOrderIter(mqr.root):
                # Start timing the current step's execution time
                start_time = time()
                try:
                    if step.step_type == "complex":
                        if len(self.complex_question_answering_components) > 0:
                            if self.complex_question_answering_components[0].should_answer_with_IRRR(mqr):
                                step.misc["answered_with_irrr"] = True
                                self.complex_question_answering_components[0].answer_complex_question(step, mqr.timing)
                                break
                        continue
                    elif step.step_type == "simple":
                        self.answer_simple_question(step, mqr.timing)
                    elif step.step_type == "low":
                        self.answer_low_level_question(step, mqr.timing)
                    else:
                        raise ValueError("Unknown step_type.")      # TODO: Make a custom exception for this

                    # Measure the stop time and store the total execution time for this step
                    step.timing["total"] = time() - start_time

                    # Handle any catastrophic errors have a occurred in the subcomponents of the current step (check the error list)
                    for e in step.errors:
                        # Handle Type 3 errors
                        if type(e) in [UnexpectedOperationArgsError, UnhandledOperationTypeError, MissingColumnError,
                                       UnhandledSubOperationTypeError, MissingPreviousStepDataError, DataframeMergeError,
                                       MalformedInputError, ValueError]:
                            # # Clean up and return to Interaction Manager
                            # step.timing["total"] = time() - start_time
                            # mqr.timing["answer_engine"]["total"] = time() - total_start_time
                            # return mqr
                            continue
                        # Handle Type 1 and 2 Errors
                        elif type(e) in [MissingEntitiesError, MissingRelationshipError, MultipleRelationshipsFoundWarning, NoRelationshipFoundWarning, MissingDocsError]:
                            # if len(step.result) == 0 or is_null_result(step.result):
                            #     # Clean up and return to Interaction Manager
                            #     step.timing["total"] = time() - start_time
                            #     mqr.timing["answer_engine"]["total"] = time() - total_start_time
                            #     return mqr
                            continue
                        else:
                            # There was an unexpected error, so best clean up and return to Interaction Manager (for now)
                            mqr.timing["answer_engine"]["total"] = time() - total_start_time
                            return mqr

                except (UnhandledOperationTypeError, DataframeMergeError) as e:
                    # Log the known error
                    step.errors.append(e)

                    # Perform cleanup and return to the Interaction Manager
                    step.timing["total"] = time() - start_time
                    mqr.timing["answer_engine"]["total"] = time() - total_start_time
                    return mqr

                except Exception as e:
                    # Log the unknown error
                    step.errors.append(e)

                    # Perform any cleanup before returning to the Interaction Manager
                    step.timing["total"] = time() - start_time
                    mqr.timing["answer_engine"]["total"] = time() - total_start_time

                    return mqr

        # Get the total execution time of the answer engine
        mqr.timing["answer_engine"]["total"] = time() - total_start_time

        return mqr

    def answer_simple_question(self,
                               step: Step,
                               timing: Dict = None) -> None:
        """
        Finds the answer to the simple question contained in the given step.
        :param step: The simple question step to answer.
        :param timing: A dictionary used to track cumulative operation time.
        :return: None. The answer (if any) will be stored in step.result.
        """

        # Find answer to the current decomposition step
        if len(self.high_level_question_answering_components) > 0:
            if step.operator_type.lower() in self.high_level_question_answering_components[0].operations:
                # Loop through each of the search components
                answer_results = []
                start_idx = 0
                for answerer in self.high_level_question_answering_components:
                    # Use the current search component to find answers
                    start_time = time()
                    answer_result = answerer.answer(step, timing)
                    if timing:
                        timing["answer_engine"][str(answerer.__class__.__name__)]["total"] += time() - start_time

                    # TODO: probably not necessary to use "new_index" twice (as both the index and "id" column)
                    new_index = range(start_idx, start_idx + answer_result.shape[0])
                    answer_results.append(answer_result.assign(id=new_index).set_index(pd.Index(new_index)))
                    start_idx += answer_result.shape[0]

                # Concatenate the results found from each search component into a single dataframe
                try:
                    step.result = pd.concat(answer_results).dropna(subset=['answer'])  # This assumes all dataframes have the same columns
                except Exception as e:
                    raise DataframeMergeError

                # Use the answer evaluator to score and get the final results for this step (sorts them by final_score in descending order )
                eval_start_time = time()
                step.result = self.answer_evaluator.get_best_results(step, 0.0, timing)

                # Remove duplicate answers
                merge_start_time = time()
                # step.result = self.answer_evaluator.merge_answers(step)
                if timing:
                    timing["answer_engine"]["answer_merging"] += time() - merge_start_time
                if timing:
                    timing["answer_engine"]["answer_evaluator"]["total"] += time() - eval_start_time
            else:
                # Raise an exception since we don't know how to handle the operator type for the current step
                raise UnhandledOperationTypeError(step.operator_type)
        elif step.operator_type in self.analytics_engine.operations:
            self.answer_low_level_question(step)
        else:
            return None

    def answer_low_level_question(self,
                                  step: Step,
                                  timing: Dict = None) -> None:
        """
        Finds the answer to the low-level question contained in the given step.
        :param step: The low-level question step to answer.
        :param timing: A dictionary used to track cumulative operation time.
        :return: None. The answer (if any) will be stored in step.result.
        """

        # Find answer to the current decomposition step
        if len(self.low_level_question_answering_components):
            if step.operator_type.lower() in self.low_level_question_answering_components[0].operations:

                # Loop through each of the search components
                answer_results = []
                start_idx = 0
                for answerer in self.low_level_question_answering_components:
                    # Use the current search component to find answers
                    start_time = time()
                    answer_result = answerer.answer(step, timing)
                    if timing:
                        timing["answer_engine"][str(answerer.__class__.__name__)]["total"] += time() - start_time

                    # TODO: probably not necessary to use "new_index" twice (as both the index and "id" column)
                    new_index = range(start_idx, start_idx + answer_result.shape[0])
                    answer_results.append(answer_result.assign(id=new_index).set_index(pd.Index(new_index)))
                    start_idx += answer_result.shape[0]

                # Concatenate the results found from each search component into a single dataframe
                try:
                    step.result = pd.concat(answer_results)  # This assumes all dataframes have the same columns
                except Exception as e:
                    raise DataframeMergeError

                # Use the answer evaluator to score and get the final results for this step
                if step.is_low_level_retrieval_step():
                    start_time = time()
                    step.result = self.answer_evaluator.get_best_results(step, 0.0, timing)
                    if timing:
                        timing["answer_engine"]["answer_evaluator"] += time() - start_time
            else:
                # Raise an exception since we don't know how to handle the operator type for the current step
                raise UnhandledOperationTypeError(step.operator_type)
        else:
            return None

    def answer_explanation_question(self,
                                    step: Step,
                                    timing: Dict = None) -> None:
        """
        Find the answer to the question in the given step that requires an explanation.
        :param step: The step containing an explanation question.
        :param timing: A dictionary used to track cumulative operation time.
        :return: None. The answer (if any) will be stored in step.result.
        """

        # Init timing fields
        if "explanation_qa" not in timing["answer_engine"]:
            timing["answer_engine"]["explanation_qa"] = 0

        # Use the first explanation based QA component to provide an answer
        start_time = time()
        if len(self.explanation_question_answering_components) > 0:
            explainer = self.explanation_question_answering_components[0]
            step.result = explainer.answer(step)
        if timing:
            timing["answer_engine"]["explanation_qa"] += time() - start_time

        return None

    def is_answerable_question(self,
                               mqr: HierarchicalMQR) -> bool:
        """
        Determines if the original question asked to the system is a question we want to answer.
        :param mqr: The plan for answering a given question.
        :return: Boolean
        """

        # Get the question types of the root of the MQR
        if is_explanation_question(mqr.root):
            question_type = 'explanation'
            # print("DETECTED EXPLANATION QUESTION for {}".format(mqr.root.question_text))
        elif is_simple_retrieval_question(mqr):
            question_type = 'simple-retrieval'
            # print("DETECTED SIMPLE-RETRIEVAL QUESTION for {}".format(mqr.root.question_text))
        elif is_complex_retrieval_question(mqr):
            question_type = 'complex-retrieval'
            # print("DETECTED COMPLEX-RETRIEVAL QUESTION for {}".format(mqr.root.question_text))
        else:
            question_type = 'analysis'
            # print("DETECTED ANALYSIS QUESTION for {}".format(mqr.root.question_text))

        # Check if the question type of the user's original question is one we want to answer.
        return question_type in self.answerable_question_types
