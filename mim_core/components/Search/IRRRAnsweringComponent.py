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
IRRR Answering Component

September 30, 2022
Authors: C3 Lab
"""

import os
import json
import uuid
import pandas as pd
import subprocess
from time import time
from typing import Dict
from mim_core.structs.Step import Step
from mim_core.components.Search.HighLevelQAComponent import HighLevelQAComponent
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new, is_arg_reference
from mim_core.exceptions import UnexpectedOperationArgsError, MissingEntitiesError, UnhandledOperationTypeError

class IRRRAnsweringComponent(HighLevelQAComponent):
    """
    A class that provides an interface for question answering via IRRR and the analytics engine.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.analytics_engine = kwargs.get('analytics_engine', None)
        self.intermediate_answer_dir = kwargs.get('intermediate_answer_dir', None)
        self.intermediate_answer_filename = kwargs.get('intermediate_answer_filename', None)
        self.intermediate_question_dir = kwargs.get('intermediate_question_dir', None)
        self.intermediate_question_filename = kwargs.get('intermediate_question_filename', None)
        self.num_hops = kwargs.get('num_hops', 1)
        self.num_passages_per_it = kwargs.get('num_passages_per_it', 30)

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

        try:
            if step.operator_type.lower() in ['select', 'project'] and 'BOOLEAN' in step.expected_answer_type:
                return self.boolean_search(step, timing)
            else:
                return self.operations[step.operator_type](step, timing)
        except:
            raise UnhandledOperationTypeError(step.operator_type)

    def retrieve_answer_with_irrr(self,
                                  question: str) -> str:

        # Convert question to JSON format, create a new QID for it

        q_id = str(uuid.uuid4())
        data = [{
            "id": q_id,
            "question": question,
            "answers": [],
            "src": "mim"
        }]
        question_dict = {"version": 1.0, "data": data}

        # Write JSON file to the specified directory
        with open(os.path.join(self.intermediate_question_dir, self.intermediate_question_filename), 'w') as f:
            json.dump(question_dict, f)

        # TODO: Run shell script for IRRR (use self.num_hops and self.num_passages_per_it)
        #command = f'#!/bin/bash ; cd ~/Documents/IRRR ; source env/bin/activate ; bash scripts/predict_dynamic_hops.sh {self.intermediate_answer_dir} {os.path.join(self.intermediate_question_dir, self.intermediate_question_filename)} {"/home/cameron/Documents/IRRR/downloads/model"} {self.num_passages_per_it} {self.num_hops} electra beerqa_wiki_doc_para'
        #output = subprocess.run(command, capture_output=True, shell=False)
        process = subprocess.Popen("/bin/bash", shell=False, universal_newlines=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        commands = f'cd ~/Documents/IRRR\nsource env/bin/activate\nbash scripts/predict_dynamic_hops.sh {self.intermediate_answer_dir} {os.path.join(self.intermediate_question_dir, self.intermediate_question_filename)} {"/home/cameron/Documents/IRRR/downloads/model"} {self.num_passages_per_it} {self.num_hops} electra beerqa_wiki_doc_para\n'
        output, error = process.communicate(commands)

        # Read the output file and grab the final answer
        with open(os.path.join(self.intermediate_answer_dir, self.intermediate_answer_filename), 'r') as f:
            answer_dict = json.load(f)

        # answer_dict = {"answer": {q_id: "yes"}} # Used for testing only

        final_answer = answer_dict["answer"][q_id]

        return final_answer

    def select(self,
               step: Step,
               timing: Dict = None) -> pd.DataFrame:
        """
        Performs the select operation by searching for articles relevant to the entities specified by the step.
        :param step: The step for which to carry out the select operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers.
        """

        try:
            final_answers = []
            step.misc["question_strings"] = []

            # Formulate the question string
            qdmr_question = self.formulate_select_question(step)
            step.misc["question_strings"].append(qdmr_question)

            # Retrieve and process the answer
            start_time = time()
            final_answer = self.retrieve_answer_with_irrr(qdmr_question)
            final_answers.append({
                    "answer": final_answer,
                    "confidence": 1.0,
                    "answer_confidence": 1.0,
                    "question_string": qdmr_question
            })
            if timing:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

            # Store the answer in a new data frame
            df = pd.DataFrame(final_answers)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'answer_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)
            return df
        except Exception as e:
            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def formulate_select_question(self,
                                   step: Step) -> str:
        """
        Produces a better formatted question string from the step's QDMR.
        :param step: The step containing the QDMR string.
        :return: A formatted, open-ended question.
        """
        if step.question_text:
            return step.question_text
        else:
            return "Which {}?".format(step.qdmr)

    def project(self,
                step: Step,
                timing: Dict = None) -> pd.DataFrame:
        """
        Performs the project operation using the Roberta model.
        :param step: The step for which to carry out the project operation.
        :param timing: A dictionary used to track cumulative operation time.
        :return: Pandas Dataframe containing the answers and their associated subjects.
        """

        try:
            # Get the entities for the project operation from the previous step
            prior_step_answers = get_prev_step_answers_new(step)

            # Prepare the data from previous steps
            try:
                df1_str = step.operator_args[0]
                df1 = prior_step_answers[df1_str]
            except Exception as e:
                raise UnexpectedOperationArgsError(step.operator_args) from e

            entities = df1['answer'].tolist()
            if len(entities) == 0:
                raise MissingEntitiesError()

            # Perform the search
            final_answers = []
            step.misc["question_strings"] = []
            for ent in entities:

                # Formulate the QDMR as a question
                qdmr_question = self.formulate_project_question(ent, step)
                step.misc["question_strings"].append(qdmr_question)

                # Retrieve and process the answer
                start_time = time()
                answer_candidate = self.retrieve_answer_with_irrr(qdmr_question)
                final_answers.append({
                    "answer": answer_candidate,
                    "subject": ent,
                    "confidence": 1.0,
                    "answer_confidence": 1.0,
                    "question_string": qdmr_question
                })
                if timing:
                    timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

            # Create a new data frame to store the results
            prev_step_str = step.operator_args[0]
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'subject', 'answer', 'confidence', 'answer_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)

            # Correlate the new answers with the original entities/answers from a prior step and drop duplicate columns from the previous answer dataframe
            answer_df1_str = "answer" + df1_str
            df1_cpy = df1.rename(columns={'id': df1_str, 'answer': answer_df1_str})
            df.rename(columns={'subject': answer_df1_str}, inplace=True)
            df = df.merge(df1_cpy, on=answer_df1_str, how='left', suffixes=('', "_DROP"))
            df.drop(df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)

            # Clean up the dataframe
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', prev_step_str, 'confidence','answer_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)

            # Return this data frame as the answer to this step
            return df

        except Exception as e:
            print("Unable to perform the PROJECT operation in {}.".format(str(self.__class__.__name__)))

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def proper_capitalize(self,
                          s: str) -> str:
        """
        Capitalize the string without lower-casing everything else.
        :param s: The string to capitalize.
        :return: The (properly) capitalized string.
        """
        return s[:1].upper() + s[1:]

    def formulate_project_question(self,
                                   entity: str,
                                   step: Step) -> str:
        """
        Produce a question string from the qdmr string via substitution.
        It will replace the qdmr step reference arg with the entity to search for.
        :param entity: The entity to place in the reference slot.
        :param step: The step containing the relationship to use.
        :return: A formatted, open-ended question.
        """

        if step.question_text:
            return step.question_text.replace(step.operator_args[0], entity)
        else:
            return "Which {}?".format(step.qdmr.replace(step.operator_args[0], entity))

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

        try:
            # Get the entities for the filter operation from the previous step
            prior_step_answers = get_prev_step_answers_new(step)

            # Prepare the data from previous steps
            try:
                df1_str = step.operator_args[0]
                df1 = prior_step_answers[df1_str]
            except Exception as e:
                raise UnexpectedOperationArgsError(step.operator_args) from e

            entities = df1['answer'].tolist()
            if len(entities) == 0:
                raise MissingEntitiesError()

            # Perform the search
            final_answers = []
            step.misc["filter_questions_and_answers"] = []
            step.misc["docs"] = []
            for ent in entities:
                # Formulate the QDMR as a question
                qdmr_question = self.formulate_filter_question(ent, step)

                # Retrieve and process the answer
                start_time = time()
                retrieved_answer = self.retrieve_answer_with_irrr(qdmr_question)
                if retrieved_answer == 'yes':
                    final_answer = True
                else:
                    final_answer = False

                if final_answer:
                    final_answers.append({
                        "answer": ent,
                        "confidence": 1.0,
                        "answer_confidence": 1.0,
                        "question_string": qdmr_question
                    })
                if timing:
                    timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                # Keep track of results
                step.misc["filter_questions_and_answers"].append({"filter_question_string": qdmr_question, "retrieved_answer": retrieved_answer, "results": final_answer})

            # Create a new data frame to store the results
            prev_step_str = step.operator_args[0]
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'answer_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)

            # Correlate the new answers with the original entities/answers from a prior step and drop duplicate columns from the previous answer dataframe
            df1_cpy = df1.rename(columns={'id': df1_str})
            df = df.merge(df1_cpy, on='answer', how='left', suffixes=('', "_DROP"))
            df.drop(df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)

            # Clean up the dataframe
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', prev_step_str, 'confidence', 'answer_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)

            # Return this data frame as the answer to this step
            return df

        except Exception as e:
            print("Unable to perform the FILTER operation in {}.".format(str(self.__class__.__name__)))

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def formulate_filter_question(self,
                                  entity: str,
                                  step: Step) -> str:
        """
        Produces a question string from the qdmr string via substitution.
        Creates a boolean question asking if the entity satisfies the filter string.
        :param entity: The entity to be filtered.
        :param step: The step containing the filter string to use.
        :return: A formatted boolean question.
        """
        if step.question_text:
            return step.question_text.replace(step.operator_args[0], entity)
        return "Is it {} {}?".format(entity, step.qdmr.replace(step.operator_args[0], ""))

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
        try:
            # Perform the search
            final_answers = []

            step.misc["question_strings"] = []
            step.misc["aggregate_answers"] = []

            # parent_step_ref_num = is_arg_reference(step.qdmr)  # Check if the arg contains a pattern like @@<number>@@
            # parent_step = mqr.steps[int(parent_step_ref_num) - 1]
            parent_steps = step.get_parent_steps()
            parent_step = parent_steps[0]

            # Case 1: The prior step references another prior step (carry out aggregate for possibly multiple entities)
            grandparent_step_ref_num = is_arg_reference(parent_step.qdmr)
            if grandparent_step_ref_num:
                # grandparent_step = mqr.steps[int(grandparent_step_ref_num) - 1]
                grandparent_step = parent_step.get_parent_steps()[0]
                # Get the entities in the grandparent step
                entities = grandparent_step.result['answer'].tolist()

                # for each of these entities in the grandparent step, ask a question for each doc source and take top answer
                for ent in entities:

                    # Formulate question for parent step
                    parent_qdmr = parent_step.qdmr.replace("@@"+str(grandparent_step_ref_num)+"@@", ent)

                    # Retrieve and process the answer
                    start_time = time()
                    answer_candidate = self.retrieve_answer_with_irrr(parent_qdmr)
                    final_answers.append({
                        "answer": answer_candidate,
                        "confidence": 1.0,
                        "answer_confidence": 1.0,
                        "question_string": parent_qdmr
                    })
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

            # Case 2: The prior step does not reference another prior step (so just carry out aggregate for single question)
            else:
                # Formulate the QDMR as a question
                # qdmr_question = step.qdmr.replace("@@" + str(prior_step_ref_num) + "@@", prior_step.qdmr)
                # TODO: Fix bug: the prior_step.qdmr sometimes contains a reference to a previous step. Need to substitute in answers from two steps up.
                qdmr_question = self.formulate_aggregate_question(step, parent_step.qdmr)
                step.misc["question_strings"].append(qdmr_question)

                # Retrieve and process the answer
                start_time = time()
                answer_candidate = self.retrieve_answer_with_irrr(qdmr_question)
                final_answers.append({
                    "answer": answer_candidate,
                    "confidence": 1.0,
                    "answer_confidence": 1.0,
                    "question_string": qdmr_question
                })
                if timing:
                    timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

            step.misc["aggregate_answers"].append(final_answers)

            # Store the answers in a new data frame
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)
            return df

        except Exception as e:
            print("Unable to perform the AGGREGATE operation in {}.".format(str(self.__class__.__name__)))

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def formulate_aggregate_question(self,
                                     current_step: Step,
                                     parent_step_qdmr: str) -> str:
        """
        Creates an aggregate question asking how many of something there are.
        :param current_step_qdmr: The current, aggregate step.
        :param parent_qdmr: The QDMR of the parent step.
        :return: A formatted aggregate question.
        """
        if 'number of' in current_step.qdmr:
            return "How many {} are there?".format(parent_step_qdmr)
        else:
            return "What is the " + current_step.qdmr.replace(current_step.operator_args[1], parent_step_qdmr) + "?"

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

        # TODO: In rare cases, there is a previous step reference that needs to be handled here.

        try:
            # Perform the search
            final_answers = []
            step.misc["boolean_question_string"] = []
            step.misc["boolean_answers"] = []
            step.misc["docs"] = []
            step.misc["passages"] = []

            # Formulate the QDMR as a question
            qdmr_question = self.formulate_boolean_question(step)
            step.misc["boolean_question_string"].append(qdmr_question)

            # Retrieve and process the answer
            start_time = time()
            retrieved_answer = self.retrieve_answer_with_irrr(qdmr_question)
            if retrieved_answer == 'yes':
                final_answer = True
            else:
                final_answer = False
            final_answers.append({
                "answer": final_answer,
                "confidence": 1.0,
                "answer_confidence": 1.0,
                "question_string": qdmr_question
            })
            if timing:
                timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

            step.misc["boolean_answers"].append(final_answers)

            # Store the answers in a new data frame
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'answer_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)
            return df

        except Exception as e:
            print("Unable to perform the BOOLEAN-SEARCH operation in {}.".format(str(self.__class__.__name__)))

            # Log the error
            step.errors.append(e)

            # Create and return a default Pandas DataFrame with a null answer
            return create_null_result()

    def formulate_boolean_question(self,
                                   step: Step) -> str:
        """
        Creates an boolean question asking if something is true or not.
        :param step: The step containing the qdmr string to use.
        :return: A formatted boolean question string.
        """
        if step.question_text:
            return step.question_text
        else:
            return "Is it {}?".format(step.qdmr)

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
