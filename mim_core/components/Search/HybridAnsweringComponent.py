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
Hybrid Answering Component

October 12, 2022
Authors: C3 Lab
"""

import pandas as pd
from time import time
from typing import Dict
from mim_core.structs.Step import Step
from mim_core.structs.MQR import HierarchicalMQR
from mim_core.components.Search.HighLevelQAComponent import HighLevelQAComponent
from mim_core.components.Search.IRRRAnsweringComponent import IRRRAnsweringComponent
from mim_core.components.Search.HighLevelNeuralSearchComponent import HighLevelNeuralSearchComponent

class HybridAnsweringComponent(HighLevelQAComponent):
    """
    A class that provides an interface for question answering via IRRR, baseline Mim neural search, and the analytics engine.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.analytics_engine = kwargs.get('analytics_engine', None)
        self.irrr_answering_component = IRRRAnsweringComponent(**kwargs)
        self.high_level_neural_search_component = HighLevelNeuralSearchComponent(**kwargs)
        self.select_mode = kwargs.get('select_mode', 'irrr')        # OTHER OPTION IS 'mim'
        self.project_mode = kwargs.get('project_mode', 'irrr')      # OTHER OPTION IS 'mim'
        self.filter_mode = kwargs.get('filter_mode', 'irrr')        # OTHER OPTION IS 'mim'
        self.aggregate_mode = kwargs.get('aggregate_mode', 'irrr')  # OTHER OPTION IS 'mim'
        self.boolean_mode = kwargs.get('boolean_mode', 'irrr')      # OTHER OPTION IS 'mim'

    def answer(self,
               step: Step,
               timing: Dict = None) -> pd.DataFrame:
        """
        A function that wraps access to the core searching operations: select, project, filter.
        :param step: The step for which to carry out the select operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers.
        """

        # Initialize timing fields
        if timing:
            if "answer_engine" not in timing:
                timing["answer_engine"] = {}
            if str(self.__class__.__name__) not in timing["answer_engine"]:
                timing["answer_engine"][str(self.__class__.__name__)] = {"total": 0}
            if "document_retrieval" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] = 0
            if "answer_retrieval" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] = 0
            if "answer_processing" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] = 0

        if step.operator_type.lower() in ['select', 'project'] and 'BOOLEAN' in step.expected_answer_type:
            return self.boolean_search(step, timing)
        else:
            return self.operations[step.operator_type](step, timing)

    def answer_complex_question(self,
                                step: Step,
                                timing: Dict = None) -> pd.DataFrame:
        # Initialize timing fields
        if timing:
            if "answer_engine" not in timing:
                timing["answer_engine"] = {}
            if str(self.__class__.__name__) not in timing["answer_engine"]:
                timing["answer_engine"][str(self.__class__.__name__)] = {"total": 0}
            if "document_retrieval" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] = 0
            if "answer_retrieval" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] = 0
            if "answer_processing" not in timing["answer_engine"][str(self.__class__.__name__)]:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] = 0

        start_time = time()

        # Store the original num_hops of IRRR
        original_num_hops = self.irrr_answering_component.num_hops

        # Update the num_hops of IRRR to 4
        self.irrr_answering_component.num_hops = 4

        # Call IRRR
        final_answer = self.irrr_answering_component.retrieve_answer_with_irrr(step.question_text)

        # Reset num_hops of IRRR to original state
        self.irrr_answering_component.num_hops = original_num_hops

        # Process the answer
        final_answers = []
        final_answers.append({
            "answer": final_answer,
            "confidence": 1.0,
            "answer_confidence": 1.0,
            "question_string": step.question_text
        })
        if timing:
            timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

        # Store the answer in a new data frame
        df = pd.DataFrame(final_answers)
        df['id'] = df.index
        df = df.reindex(columns=['id', 'answer', 'confidence', 'answer_confidence', 'question_string'])
        df['source'] = str(self.irrr_answering_component.__class__.__name__)

        # Store the result
        step.result = df

        return df

    def should_answer_with_IRRR(self,
                                mqr: HierarchicalMQR) -> bool:
        # Check if the question was decomposed or not
        p = mqr.to_json()
        if len(p["steps"]) <= 1:
            return True

        # Check if the form is one of comparison, discard, intersection, or bridge.
        bridge_templates = [('select'),
                            ('select', 'project'),
                            ('select', 'project', 'project'),
                            ('select', 'project', 'project', 'project'),
                            ('select', 'project', 'filter'),
                            ('select', 'project', 'aggregate'),
                            ('select', 'aggregate')]
        comparison_templates = [('select', 'select', 'aggregate', 'aggregate', 'comparison'),
                                ('boolean', 'boolean', 'comparison'),
                                ('select', 'select', 'comparison')]
        discard_templates = [('select', 'discard'),
                             ('select', 'filter'),
                             ('select', 'select', 'discard')]
        intersection_templates = [('select', 'select', 'intersection'),
                                  ('select', 'filter')]

        p_simple_steps = tuple([s["operator_type"] for s in p["steps"][1:]])

        return p_simple_steps in bridge_templates or p_simple_steps in comparison_templates or p_simple_steps in discard_templates or p_simple_steps in intersection_templates

    def return_dummy_answer(self,
                            dummy: str,
                            step: Step,
                            timing: Dict = None) -> pd.DataFrame:
        # Process the answer
        final_answers = []
        final_answers.append({
            "answer": dummy,
            "confidence": 1.0,
            "answer_confidence": 1.0,
            "question_string": step.question_text
        })

        # Store the answer in a new data frame
        df = pd.DataFrame(final_answers)
        df['id'] = df.index
        df = df.reindex(columns=['id', 'answer', 'confidence', 'answer_confidence', 'question_string'])
        df['source'] = str(self.__class__.__name__)

        return df

    def select(self,
               step: Step,
               timing: Dict = None) -> pd.DataFrame:
        """
        Performs the select operation by searching for articles relevant to the entities specified by the step.
        :param step: The step for which to carry out the select operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers.
        """

        if self.select_mode == 'irrr':
            return self.irrr_answering_component.answer(step, timing)
        else:
            return self.high_level_neural_search_component.answer(step, timing)
        # return self.return_dummy_answer("select answer", step, timing)

    def project(self,
                step: Step,
                timing: Dict = None) -> pd.DataFrame:
        """
        Performs the project operation using the Roberta model.
        :param step: The step for which to carry out the project operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers and their associated subjects.
        """

        if self.project_mode == 'irrr':
            return self.irrr_answering_component.answer(step, timing)
        else:
            return self.high_level_neural_search_component.answer(step, timing)
        # return self.return_dummy_answer("project answer", step, timing)

    def filter(self,
               step: Step,
               timing: Dict = None) -> pd.DataFrame:
        """
        Performs the filter operation using the information provided by the decomposition step.
        Note: Formulates the search question as a boolean question and uses the found documents to answer it.
              Each answer from the previous step is kept if the boolean question is answered with 'yes'.
        :param step: The step for which to carry out the filter operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers that passed the filter.
        """

        if self.filter_mode == 'irrr':
            return self.irrr_answering_component.answer(step, timing)
        else:
            return self.high_level_neural_search_component.answer(step, timing)
        # return self.return_dummy_answer("filter answer", step, timing)

    def aggregate(self,
                  step: Step,
                  timing: Dict = None) -> pd.DataFrame:
        """
        Performs the aggregate operation using the information provided by the decomposition step.
        Note: This operation makes use of the same model used for select/project operations.
        :param step: The step for which to carry out the aggregate operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return:
        """

        if self.aggregate_mode == 'irrr':
            return self.irrr_answering_component.answer(step, timing)
        else:
            return self.high_level_neural_search_component.answer(step, timing)
        # return self.return_dummy_answer("100", step, timing)

    def boolean(self,
                step: Step,
                timing: Dict = None) -> pd.DataFrame:
        """
        A wrapper function for routing answering to the appropriate search or analytic operation.
        :param step: The step for which to carry out the boolean operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return:
        """
        if step.operator_subtype == 'boolean-search':
            return self.boolean_search(step, timing)
        else:
            return self.analytics_engine.analyze_data(step)

    def boolean_search(self,
                       step: Step,
                       timing: Dict = None) -> pd.DataFrame:
        """
        Performs the boolean operation using the information provided by the decomposition step.
        :param step: The step for which to carry out the boolean operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return:
        """
        if self.boolean_mode == 'irrr':
            return self.irrr_answering_component.answer(step, timing)
        else:
            return self.high_level_neural_search_component.answer(step, timing)
        # return self.return_dummy_answer(True, step, timing)

    def arithmetic(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def group(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def superlative(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def union(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def comparative(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def comparison(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def intersection(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def discard(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)

    def sort(self, step: Step, timing: Dict = None) -> pd.DataFrame:
        return self.analytics_engine.analyze_data(step)
