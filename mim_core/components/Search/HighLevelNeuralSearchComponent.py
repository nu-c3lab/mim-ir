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
High Level Neural Search Component
August 22, 2021
Authors: C3 Lab
"""

import dateparser
import pandas as pd
from functools import reduce
from time import time
from typing import Dict, List
from mim_core.structs.Step import Step
from num_parse.NumParser import NumParser
from mim_core.components.Search.HighLevelQAComponent import HighLevelQAComponent
from mim_core.components.models import get_model
from mim_core.utils.result_utils import create_null_result, get_prev_step_answers_new, is_arg_reference, is_set_operation_step
import mim_core.utils.component_loading as cl_utils
from mim_core.exceptions import MissingDocsError, UnexpectedOperationArgsError, MissingEntitiesError, UnhandledOperationTypeError


class HighLevelNeuralSearchComponent(HighLevelQAComponent):
    """
    A class that provides an interface for a neural network based searching mechanism using high level QDMR.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.ontology = cl_utils.load_ontology(kwargs.get('ontology', None))
        self.document_sources = [cl_utils.load_document_source(doc) for doc in kwargs.get('document_sources', [])]
        self.project_model = cl_utils.load_neural_qa_model(kwargs.get('project_model', None))
        self.search_threshold = max(kwargs.get('search_threshold', 0.0), 0.0)
        self.max_project_results = kwargs.get('max_project_results', 1)
        self.max_select_results = kwargs.get('max_select_results', 2)
        self.max_set_ops_inputs = kwargs.get('max_set_ops_inputs', 15)
        self.filter_model = cl_utils.load_neural_qa_model(kwargs.get('filter_model', None))
        self.filter_threshold = kwargs.get('filter_threshold', 0.5)
        self.analytics_engine = kwargs.get('analytics_engine', None)
        self.simple_doc_weight = kwargs.get('simple_doc_weight', 0.3)
        self.simple_answer_weight = kwargs.get('simple_answer_weight', 0.7)
        self.complex_doc_weight = kwargs.get('complex_doc_weight', 0.5)
        self.complex_answer_weight = kwargs.get('complex_answer_weight', 0.5)
        self.nlp = get_model('en_core_web_trf')
        self.num_parser = NumParser()

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

    def get_keywords(self,
                     step: Step,
                     manual_keywords: List[str] = None) -> List[str]:
        """
        Produces a list of keywords to search with.
        :param step: The step from which to derive keywords.
        :param manual_keywords: A list of keywords that will be ALWAYS added to the final list.
        :return: A list of keywords to use in improving document retrieval results.
        """

        # TODO: Get the entities that were linked to a proper noun in Wikidata (THIS IS A KLUDGE! DO BETTER!)
        keywords = [ent for ent in step.entities.keys() if ent[0].isupper()]
        # TODO: This may help in addressing the todo above. Switch to this line to test. Might also play around with removing the "and step.operator_type == 'select'" filter to see if it produces better results.
        # keywords = reduce(lambda a, low_step: a + list(low_step.entities.keys()), filter(lambda step: step.step_type == 'low' and step.operator_type == 'select', step.descendants), [])

        # Add all of the keywords specified manually
        if manual_keywords:
            keywords += manual_keywords

        step.misc["doc_keywords"] = keywords

        return keywords

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
            step.misc["select_answers"] = []
            step.misc["docs"] = []
            step.misc["passages"] = []
            for doc_source in self.document_sources:

                # Formulate the question string
                qdmr_question = self.formulate_select_question(step)
                step.misc["question_strings"].append(qdmr_question)

                # Find the proper documents associated with the current entity
                start_time = time()
                doc_retrieval_keywords = self.get_keywords(step)
                docs = doc_source.get_documents(qdmr_question, step.q_id, keywords=doc_retrieval_keywords)
                passages = doc_source.get_passages(qdmr_question, docs, q_id=step.q_id, keywords=doc_retrieval_keywords)
                if timing:
                    timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] += time() - start_time
                step.misc["docs"] = [(d.title, d.score) for d in docs]
                step.misc["passages"] = [p.to_json() for p in passages]
                if len(passages) == 0:
                    continue
                #     raise MissingDocsError()

                # Use the neural search model to find the answer to the current QDMR step
                search_results = []
                for passage in passages:
                    start_time = time()
                    results = self.project_model.answer_question(qdmr_question, passage)
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                    # Process the answers that were found
                    start_time = time()
                    for r in results:
                        for ent in self.ontology.extract_canonical_entities(r[0], single_term=True):
                            search_results.append({
                                "answer": ent,
                                "confidence": float(self.complex_answer_weight * r[1] + self.complex_doc_weight * passage.score) if step.root.step_type == 'complex' else float(self.simple_answer_weight * r[1] + self.simple_doc_weight * passage.score),
                                "answer_confidence": r[1],
                                "doc_title": passage.title,
                                "doc_retrieval_confidence": float(passage.score),
                                "question_string": qdmr_question
                            })
                step.misc["select_answers"].append(search_results)

                # Sort the answers in descending order and pick the top answers
                sorted_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)

                # Make sure we're getting a datetime or numeric answer if the following step requires one
                if any([cs.operator_type == 'arithmetic' or (cs.operator_type in ['boolean', 'comparison', 'comparative'] and step.expected_answer_type[0] in ['DATE', 'NUMERIC']) for cs in step.child_steps]):

                    valid_results = []
                    for result in sorted_results:
                        try:
                            if self.num_parser.parse_num(result['answer']) or dateparser.parse(result['answer']):
                                valid_results.append(result)
                        except Exception as e:
                            continue
                else:
                    valid_results = sorted_results

                # Perform thresholding based on the confidence scores
                good_answers = [r for r in valid_results if r["confidence"] > self.search_threshold]

                # If there are no good answers exceeding the threshold, return only the best one
                if not good_answers:
                    final_answers.append(valid_results[0])
                else:
                    num_to_return = self.max_set_ops_inputs if any([is_set_operation_step(s) for s in step.child_steps]) else self.max_select_results
                    final_answers = good_answers[0:num_to_return]

                if timing:
                    timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] += time() - start_time

            # Store the answers in a new data frame
            df = pd.DataFrame(final_answers)#, columns=['answer', 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
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
            step.misc["project_answers"] = []
            step.misc["docs"] = []
            step.misc["passages"] = []
            for ent in entities:
                search_results = []
                for doc_source in self.document_sources:
                    # Formulate the QDMR as a question
                    qdmr_question = self.formulate_project_question(ent, step)
                    step.misc["question_strings"].append(qdmr_question)

                    # Find the proper documents associated with the current entity
                    start_time = time()
                    doc_retrieval_keywords = self.get_keywords(step, [ent])
                    docs = doc_source.get_documents(qdmr_question, step.q_id, keywords=doc_retrieval_keywords)
                    passages = doc_source.get_passages(qdmr_question, docs, q_id=step.q_id, keywords=doc_retrieval_keywords)
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] += time() - start_time
                    step.misc["docs"] = [(d.title, d.score) for d in docs]
                    step.misc["passages"] = [p.to_json() for p in passages]

                    if len(passages) == 0:
                        continue
                    #     raise MissingDocsError()

                    # Use the neural search model to find the answer to the current QDMR step
                    for passage in passages:
                        start_time = time()
                        results = self.project_model.answer_question(qdmr_question, passage)
                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                        # Process the answers that were found
                        start_time = time()
                        for r in results:
                            for new_ent in self.ontology.extract_canonical_entities(r[0], single_term=True):
                                search_results.append({
                                    "answer": new_ent,
                                    "subject": ent,
                                    "confidence": float(self.complex_answer_weight * r[1] + self.complex_doc_weight * passage.score) if step.root.step_type == 'complex' else float(self.simple_answer_weight * r[1] + self.simple_doc_weight * passage.score),
                                    "answer_confidence": r[1],
                                    "doc_title": passage.title,
                                    "doc_retrieval_confidence": float(passage.score),
                                    "question_string": qdmr_question
                                })
                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] += time() - start_time

                    step.misc["project_answers"].append(search_results)

                # If no results are found, move onto the next entity
                if len(search_results) == 0:
                    continue

                # Method 3: Take the top k answers with the highest confidence that exceed the search_threshold
                # Sort the answers in descending order and pick the top answers
                sorted_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)

                # Make sure we're getting a datetime or numeric answer if the following step requires one
                if any([cs.operator_type == 'arithmetic' or (cs.operator_type in ['boolean', 'comparison', 'comparative'] and step.expected_answer_type[0] in ['DATE', 'NUMERIC']) for cs in step.child_steps]):

                    valid_results = []
                    for result in sorted_results:
                        try:
                            if self.num_parser.parse_num(result['answer']) or dateparser.parse(result['answer']):
                                valid_results.append(result)
                        except Exception as e:
                            continue
                else:
                    valid_results = sorted_results

                # Perform thresholding based on the confidence scores
                good_answers = [r for r in valid_results if r["confidence"] > self.search_threshold]

                # If there are no good answers exceeding the threshold, return only the best one
                if not good_answers:
                    final_answers.append(valid_results[0])
                else:
                    num_to_return = self.max_set_ops_inputs if any([is_set_operation_step(s) for s in step.child_steps]) else self.max_project_results
                    final_answers = good_answers[0:num_to_return]

            # Create a new data frame to store the results
            prev_step_str = step.operator_args[0]
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'subject', 'answer', 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)

            # Correlate the new answers with the original entities/answers from a prior step and drop duplicate columns from the previous answer dataframe
            answer_df1_str = "answer" + df1_str
            df1_cpy = df1.rename(columns={'id': df1_str, 'answer': answer_df1_str})
            df.rename(columns={'subject': answer_df1_str}, inplace=True)
            df = df.merge(df1_cpy, on=answer_df1_str, how='left', suffixes=('', "_DROP"))
            df.drop(df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)

            # Clean up the dataframe
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', prev_step_str, 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
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

                search_results = []
                for doc_source in self.document_sources:
                    # Find the proper documents associated with the current entity
                    start_time = time()
                    doc_retrieval_keywords = self.get_keywords(step, [ent])
                    docs = doc_source.get_documents(qdmr_question, step.q_id, keywords=doc_retrieval_keywords)
                    passages = doc_source.get_passages(qdmr_question, docs, q_id=step.q_id, keywords=doc_retrieval_keywords)
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] += time() - start_time
                    step.misc["docs"] = [(d.title, d.score) for d in docs]
                    step.misc["passages"] = [p.to_json() for p in passages]

                    if len(passages) == 0:
                        continue
                    #     raise MissingDocsError()

                    # Use the filtering model to find the answer to the current QDMR step
                    for passage in passages:
                        start_time = time()
                        result = self.filter_model.answer_question(qdmr_question, passage)
                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                        # Process the answers that were found
                        start_time = time()
                        # Update the "answer" based on the probability of "yes" in the result
                        if result[1] < 0.5:
                            answer = False
                            answer_confidence = 1 - result[1]
                        else:
                            answer = True
                            answer_confidence = result[1]

                        search_results.append({
                            "answer": answer,
                            "confidence": float(self.complex_answer_weight * answer_confidence + self.complex_doc_weight * passage.score) if step.root.step_type == 'complex' else float(self.simple_answer_weight * answer_confidence + self.simple_doc_weight * passage.score),
                            "answer_confidence": answer_confidence,
                            "doc_title": passage.title,
                            "doc_retrieval_confidence": float(passage.score),
                            "question_string": qdmr_question
                        })

                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] += time() - start_time

                # Keep track of results
                step.misc["filter_questions_and_answers"].append({"filter_question_string": qdmr_question, "results": search_results})

                # If no results are found, move onto the next entity
                if len(search_results) == 0:
                    continue

                # Sort the results in descending order based on the "confidence" param (combined answer_confidence and doc_retrieval_confidence)
                sorted_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)

                # Take the highest scoring answering
                best_result = sorted_results[0]

                if best_result["answer"]:
                    final_answers.append({
                        "answer": ent,
                        "confidence": best_result["confidence"],
                        "answer_confidence": best_result["answer_confidence"],
                        "doc_title": best_result["doc_title"],
                        "doc_retrieval_confidence": best_result["doc_retrieval_confidence"],
                        "question_string": best_result["question_string"]
                    })

            # Create a new data frame to store the results
            prev_step_str = step.operator_args[0]
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
            df['source'] = str(self.__class__.__name__)

            # Correlate the new answers with the original entities/answers from a prior step and drop duplicate columns from the previous answer dataframe
            df1_cpy = df1.rename(columns={'id': df1_str})
            df = df.merge(df1_cpy, on='answer', how='left', suffixes=('', "_DROP"))
            df.drop(df.filter(regex='_DROP$').columns.tolist(), axis=1, inplace=True)

            # Clean up the dataframe
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', prev_step_str, 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
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
            step.misc["docs"] = []
            step.misc["passages"] = []

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
                    search_results = []

                    # Formulate question for parent step
                    parent_qdmr = parent_step.qdmr.replace("@@"+str(grandparent_step_ref_num)+"@@", ent)

                    for doc_source in self.document_sources:
                        # Formulate the QDMR as a question
                        qdmr_question = self.formulate_aggregate_question(step, parent_qdmr)
                        step.misc["question_strings"].append(qdmr_question)

                        # Find the proper documents associated with the current entity
                        start_time = time()
                        doc_retrieval_keywords = self.get_keywords(step, [ent])
                        docs = doc_source.get_documents(qdmr_question, step.q_id, keywords=doc_retrieval_keywords)
                        passages = doc_source.get_passages(qdmr_question, docs, q_id=step.q_id,keywords=doc_retrieval_keywords)
                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] += time() - start_time
                        step.misc["docs"] = [(d.title, d.score) for d in docs]
                        step.misc["passages"] = [p.to_json() for p in passages]

                        if len(passages) == 0:
                            continue
                        #     raise MissingDocsError()

                        # Use the neural search model to find the answer to the current QDMR step
                        for passage in passages:
                            start_time = time()
                            results = self.project_model.answer_question(qdmr_question, passage)
                            if timing:
                                timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                            # Process the answers that were found
                            start_time = time()
                            for r in results:
                                for new_ent in self.ontology.extract_canonical_entities(r[0], single_term=True):
                                        search_results.append({
                                            "answer": new_ent,
                                            "confidence": float(self.complex_answer_weight * r[1] + self.complex_doc_weight * passage.score) if step.root.step_type == 'complex' else float(self.simple_answer_weight * r[1] + self.simple_doc_weight * passage.score),
                                            "answer_confidence": r[1],
                                            "doc_title": passage.title,
                                            "doc_retrieval_confidence": float(passage.score),
                                            "question_string": qdmr_question
                                        })
                            if timing:
                                timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] += time() - start_time
                        step.misc["aggregate_answers"].append(search_results)

                        # If no results are found, move onto the next entity
                        if len(search_results) == 0:
                            continue

                        # Method 3: Take the top k answers with the highest confidence that exceed the search_threshold
                        # Sort the answers in descending order and pick the top answers
                        sorted_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)

                        # Make sure we're getting a datetime or numeric answer if the following step requires one
                        if any([cs.operator_type == 'arithmetic' or (cs.operator_type in ['boolean', 'comparison', 'comparative'] and step.expected_answer_type[0] in ['NUMERIC']) for cs in step.child_steps]):

                            valid_results = []
                            for result in sorted_results:
                                try:
                                    if self.num_parser.parse_num(result['answer']):
                                        valid_results.append(result)
                                except Exception as e:
                                    continue
                        else:
                            valid_results = sorted_results

                        # Perform thresholding based on the confidence scores
                        good_answers = [r for r in valid_results if r["confidence"] > self.search_threshold]

                        # If there are no good answers exceeding the threshold, return only the best one
                        if not good_answers:
                            final_answers.append(valid_results[0])
                        else:
                            final_answers.append(good_answers[0])

            # Case 2: The prior step does not reference another prior step (so just carry out aggregate for single question)
            else:
                search_results = []
                for doc_source in self.document_sources:
                    # Formulate the QDMR as a question
                    # qdmr_question = step.qdmr.replace("@@" + str(prior_step_ref_num) + "@@", prior_step.qdmr)
                    # TODO: Fix bug: the prior_step.qdmr sometimes contains a reference to a previous step. Need to substitute in answers from two steps up.
                    qdmr_question = self.formulate_aggregate_question(step, parent_step.qdmr)
                    step.misc["question_strings"].append(qdmr_question)

                    # Find the proper documents associated with the current entity
                    start_time = time()
                    doc_retrieval_keywords = self.get_keywords(step)
                    docs = doc_source.get_documents(qdmr_question, step.q_id, keywords=doc_retrieval_keywords)
                    passages = doc_source.get_passages(qdmr_question, docs, q_id=step.q_id, keywords=doc_retrieval_keywords)
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] += time() - start_time
                    step.misc["docs"] = [(d.title, d.score) for d in docs]
                    step.misc["passages"] = [p.to_json() for p in passages]

                    if len(passages) == 0:
                        continue
                    #     raise MissingDocsError()

                    # Use the neural search model to find the answer to the current QDMR step
                    for passage in passages:
                        start_time = time()
                        results = self.project_model.answer_question(qdmr_question, passage)
                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                        # Process the answers that were found
                        start_time = time()
                        for r in results:
                            for new_ent in self.ontology.extract_canonical_entities(r[0], single_term=True):
                                search_results.append({
                                    "answer": new_ent,
                                    "confidence": float(self.complex_answer_weight * r[1] + self.complex_doc_weight * passage.score) if step.root.step_type == 'complex' else float(self.simple_answer_weight * r[1] + self.simple_doc_weight * passage.score),
                                    "answer_confidence": r[1],
                                    "doc_title": passage.title,
                                    "doc_retrieval_confidence": float(passage.score),
                                    "question_string": qdmr_question
                                })
                        if timing:
                            timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] += time() - start_time
                    step.misc["aggregate_answers"].append(search_results)

                    # If no results are found, move onto the next entity
                    if len(search_results) == 0:
                        continue

                    # Method 3: Take the top k answers with the highest confidence that exceed the search_threshold
                    # Sort the answers in descending order and pick the top answers
                    sorted_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)

                    # Make sure we're getting a datetime or numeric answer if the following step requires one
                    if any([cs.operator_type == 'arithmetic' or (cs.operator_type in ['boolean', 'comparison', 'comparative'] and step.expected_answer_type[0] in ['NUMERIC']) for cs in step.child_steps]):

                        valid_results = []
                        for result in sorted_results:
                            try:
                                if self.num_parser.parse_num(result['answer']):
                                    valid_results.append(result)
                            except Exception as e:
                                continue
                    else:
                        valid_results = sorted_results

                    # Perform thresholding based on the confidence scores
                    good_answers = [r for r in valid_results if r["confidence"] > self.search_threshold]

                    # If there are no good answers exceeding the threshold, return only the best one
                    if not good_answers:
                        final_answers.append(valid_results[0])
                    else:
                        final_answers.append(good_answers[0])

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

            search_results = []
            for doc_source in self.document_sources:
                # Find the proper documents associated with the current entity
                start_time = time()
                doc_retrieval_keywords = self.get_keywords(step)
                docs = doc_source.get_documents(qdmr_question, step.q_id, keywords=doc_retrieval_keywords)
                passages = doc_source.get_passages(qdmr_question, docs, q_id=step.q_id, keywords=doc_retrieval_keywords)
                if timing:
                    timing["answer_engine"][str(self.__class__.__name__)]["document_retrieval"] += time() - start_time
                step.misc["docs"] = [(d.title, d.score) for d in docs]
                step.misc["passages"] = [p.to_json() for p in passages]

                if len(passages) == 0:
                    continue
                #     raise MissingDocsError()

                for passage in passages:
                    start_time = time()
                    result = self.filter_model.answer_question(qdmr_question, passage)
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["answer_retrieval"] += time() - start_time

                    # Process the answers
                    start_time = time()
                    # Update the "answer" based on the probability of "yes" in the result
                    if result[1] < 0.5:
                        answer = False
                        answer_confidence = 1 - result[1]
                    else:
                        answer = True
                        answer_confidence = result[1]

                    search_results.append({
                        "answer": answer,
                        "confidence": float(self.complex_answer_weight * answer_confidence + self.complex_doc_weight * passage.score) if step.root.step_type == 'complex' else float(self.simple_answer_weight * answer_confidence + self.simple_doc_weight * passage.score),
                        "answer_confidence": answer_confidence,
                        "doc_title": passage.title,
                        "doc_retrieval_confidence": float(passage.score),
                        "question_string": qdmr_question
                    })
                    if timing:
                        timing["answer_engine"][str(self.__class__.__name__)]["answer_processing"] += time() - start_time

                step.misc["boolean_answers"].append(search_results)

            # Sort the results in descending order based on the "confidence" param (combined answer_confidence and doc_retrieval_confidence)
            sorted_results = sorted(search_results, key=lambda x: x["confidence"], reverse=True)

            # Take the highest scoring answering
            final_answers.append(sorted_results[0])

            # Store the answers in a new data frame
            df = pd.DataFrame(final_answers)
            if len(df) > 1:
                df.sort_values(by=['confidence'], inplace=True, ignore_index=True, ascending=False)   # Sort dataframe according to confidence (highest is first)
            df['id'] = df.index
            df = df.reindex(columns=['id', 'answer', 'confidence', 'doc_title', 'answer_confidence', 'doc_retrieval_confidence', 'question_string'])
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
