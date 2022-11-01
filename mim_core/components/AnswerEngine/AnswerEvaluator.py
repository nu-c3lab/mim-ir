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
Answer Evaluator

March 28, 2021
Authors: C3 Lab
"""

import os
from pathlib import Path
import csv
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
from time import time

from mim_core.structs.Step import Step
from mim_core.components.models import get_model
import mim_core.utils.component_loading as cl_utils
from mim_core.utils.result_utils import download_file_from_google_drive

class AnswerEvaluator(object):
    """
    A component for managing scoring and evaluation of answers/evidence.
    """

    def __init__(self, **kwargs):
        self.scoring_functions = [cl_utils.load_scoring_function(sf) for sf in kwargs.get('scoring_functions', [])]
        self.ontology = cl_utils.load_ontology(kwargs.get('ontology', None))
        self.score_combination_mode = kwargs.get('score_combination_mode', 'sum')
        self.do_output_feature_vectors = kwargs.get('do_output_feature_vectors', False)
        self.feature_vector_output_dir = kwargs.get('feature_vector_output_dir', None)
        self.output_file_name = None
        self.csv_writer = None
        self.scoring_function_names = [str(scorer.__class__.__name__) for scorer in self.scoring_functions]
        self.trf_model = get_model('en_core_web_trf')
        self.med_model = get_model('en_core_web_md')

        if self.do_output_feature_vectors:
            self.output_file_name = self._get_output_file_name()
            header = ['q_id', 'step_num']
            header += self.scoring_function_names
            header += ['answer_confidence', 'doc_retrieval_confidence', 'question', 'pred_answer']
            with open(self.output_file_name, 'w') as f:
                self.csv_writer = csv.writer(f)
                self.csv_writer.writerow(header)

        # Load the score blending regression model
        self._load_score_combination_model()

    def get_best_results(self,
                         step: Step,
                         threshold: float = None,
                         timing: Dict = None) -> pd.DataFrame:
        """
        Evaluates each of the results by applying the scoring functions
        and removing any that are less than a certain threshold.
        :param step: The step containing a result to evaluate.
        :param threshold: The final score threshold that results must exceed.
        :param timing: A dictionary used to track cumulative operation time.
        :return: The dataframe consisting of the best results.
        """

        # Initialize timing fields
        if timing:
            if "answer_engine" not in timing:
                timing["answer_engine"] = {}
            if "answer_evaluator" not in timing["answer_engine"]:
                timing["answer_engine"]["answer_evaluator"] = {"total": 0}
            for scorer in self.scoring_functions:
                if str(scorer.__class__.__name__) not in timing["answer_engine"]["answer_evaluator"]:
                    timing["answer_engine"]["answer_evaluator"][str(scorer.__class__.__name__)] = 0
            if "score_merging" not in timing["answer_engine"]["answer_evaluator"]:
                timing["answer_engine"]["answer_evaluator"]["score_merging"] = 0
            if "answer_thresholding_and_sorting" not in timing["answer_engine"]["answer_evaluator"]:
                timing["answer_engine"]["answer_evaluator"]["answer_thresholding_and_sorting"] = 0

        # Get the score using each of the specified scoring functions
        try:
            if step.is_high_level_retrieval_step():

                score_column_names = []
                for scorer in self.scoring_functions:
                    try:
                        start_time = time()
                        step.result = scorer.evaluate_results(step)
                        score_column_names.append(str(scorer.__class__.__name__))
                        if timing:
                            timing["answer_engine"]["answer_evaluator"][str(scorer.__class__.__name__)] += time() - start_time
                    except Exception as e:
                        # Log the error
                        step.errors.append(e)

                # For each answer candidate, produce a final score
                start_time = time()
                step.result['final_score'] = 0.0
                for idx, row in step.result.iterrows():
                    # Write out answer candidates, scores, and other metadata for training the score blending model
                    if self.do_output_feature_vectors:
                        output_row_data = [step.q_id, step.reference_id]
                        output_row_data += [row[name] for name in self.scoring_function_names]
                        output_row_data += [row['answer_confidence'], row['doc_retrieval_confidence'], row['question_string'], row['answer']]
                        with open(self.output_file_name, 'a') as f:
                            self.csv_writer = csv.writer(f)
                            self.csv_writer.writerow(output_row_data)

                    # Build the feature vector of scores
                    scores_vector = self.build_feature_vector(row)

                    # Combine the scores for the current answer candidate into a single score for ranking
                    step.result.at[idx, 'final_score'] = self.combine_features(scores_vector)

                # step.result['final_score'] = step.result[score_column_names].sum(axis=1)
                if timing:
                    timing["answer_engine"]["answer_evaluator"]["score_merging"] += time() - start_time

                # Apply the threshold to winnow the results
                start_time = time()
                if threshold:
                    step.result = step.result[step.result['final_score'] >= threshold]

                # Sort answers in descending order based on value of 'final_score'
                step.result.sort_values(by='final_score', ascending=False, inplace=True)
                if timing:
                    timing["answer_engine"]["answer_evaluator"]["answer_thresholding_and_sorting"] += time() - start_time
            else:
                step.result[['final_score']] = 1.0
        except Exception as e:
            # Log the unknown error
            step.errors.append(e)

        finally:
            return step.result

    # def merge_answers(self,
    #                   step: Step) -> pd.DataFrame:
    #     """
    #     Creates a Dataframe of answers where duplicate entities/concepts are merged.
    #     :param step: The step containing a result to evaluate.
    #     :return: A Dataframe without duplicate answers.
    #     """
    #
    #     # Sort the dataframe using the 'final_score' column
    #     final_results = step.result.sort_values('final_score', ascending=False)
    #
    #     expected_types = step.get_expected_type()
    #     if expected_types:
    #         deduper = Deduper(expected_types[0], nlp=self.trf_model if expected_types[0] in ['HUMAN_INDIVIDUAL', 'LOCATION'] else self.med_model)
    #         final_results = deduper.dedupe(final_results)
    #     else:
    #         final_results['answer'] = [list(self.ontology.extract_canonical_entities(ans, single_term = True).values())[0]['label'] for ans in final_results["answer"].tolist()]
    #
    #         # Remove duplicate values and keep the first one (which should be the one with the highest final score)
    #         final_results = final_results.drop_duplicates(subset=['answer'])
    #
    #     # Remove nans if there is a non-nan answer present
    #     if len(final_results) > 1:
    #         final_results = final_results.dropna(subset=['id', 'answer', 'confidence', 'source', 'final_score'])
    #
    #         # Sort answers in descending order based on value of 'final_score'
    #         final_results.sort_values(by='final_score', ascending=False, inplace=True)
    #
    #     return final_results

    def build_feature_vector(self,
                             answer_row: pd.Series) -> np.ndarray:
        """
        Constructs the feature vector of various scores to use when producing the final score for an answer.
        :param answer_row: The row containing the answer and its associated scores.
        :return: A list of float values denoting the scores.s
        """

        # TODO: This currently is only making use of the scores coming from ScoringFunctions. Will need to be updated
        #       if/when we start using columns or metadata coming from each step for training the model
        #       (e.g. answer_confidence, doc_retrieval_confidence, etc.).
        return np.array([float(answer_row[col_name]) for col_name in self.scoring_function_names])

    def combine_features(self,
                         scores_vector: np.ndarray) -> float:
        """
        Combines the features / scores for a given answer candidate into a single final score.
        :param scores_vector: The array of features / scores.
        :return: A single, final score.
        """

        if self.score_combination_mode == 'logistic_regression' and self.score_combination_model:
            # Use the score combination model to predict probability of an answer candidate being correct based on its scores
            score = self.score_combination_model.predict_proba(np.array([scores_vector]))
            final_score = score[0][1]
            return final_score
        elif self.score_combination_mode == 'sum':
            return np.sum(scores_vector)
        else:
            # Default to summing up the scores
            return np.sum(scores_vector)

    def _get_output_file_name(self) -> str:
        """
        Opens the output file that the evaluation data will be written to.
        :return: The file object that can be written to.
        """

        # Setup the output file path
        now = datetime.now()
        return os.path.join(self.feature_vector_output_dir, "answer_eval_data_" + now.strftime("%m_%d_%Y_%H_%M_%S") + ".csv")

    def _load_score_combination_model(self) -> None:
        """
        Loads a model for performing the feature / score combination.
        :return: None
        """

        if self.score_combination_mode == 'logistic_regression':
            try:
                # Ensure the model is downloaded
                model_file_name = 'finalized_model.pickle'

                if not os.path.exists(Path(__file__).parent / './models/{}'.format(model_file_name)):
                    print('Downloading score combination model...')
                    download_file_from_google_drive('1EXbGViiQzCyD2vJHQAuBIUTA4e1zo02f',
                                                    Path(__file__).parent / './models/{}'.format(model_file_name))

                self.score_combination_model = pickle.load(open(Path(__file__).parent / 'models/{}'.format(model_file_name), 'rb'))
            except:
                # TODO: Raise exception/warning here
                self.score_combination_model = None
        else:
            self.score_combination_model = None
