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
T5 High Level Question Decomposer

August 21, 2021
Authors: C3 Lab
"""

import torch
from typing import List
from transformers import T5Tokenizer, T5ForConditionalGeneration
from mim_core.structs.Step import Step
from mim_core.structs.Utterance import Utterance
from mim_core.components.QuestionAnalysis.QuestionDecomposer import QuestionDecomposer

class T5HighLevelDecomposer(QuestionDecomposer):
    """
    An class from which uses a T5 model fine-tuned on the
    high-level split of the BREAK dataset to decompose the question.
    """

    def __init__(self,
                 model = None,
                 tokenizer = None,
                 **kwargs):
        super().__init__(model, tokenizer)
        self.use_gpu = kwargs.get('use_gpu', False) and torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_gpu else 'cpu'
        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained('msterbentz/t5-base-break-high')
        self.model = self.model.to(self.device)

    def decompose_question(self,
                           utterance: Utterance,
                           num_generations: int=1) -> List[List[Step]]:
        """
        Decomposes the question into a set of candidate plans, each consisting of a series of Steps.
        :param utterance: The Utterance object containing the question to decompose.
        :param num_generations: The number of candidate plans / decompositions to generate.
        :return: A list of Steps containing the QDMR breakdown.
        """

        # Tokenize the input text
        t = self.tokenizer([utterance.text], max_length=256, return_tensors='pt', truncation=True)
        t = t.to(self.device)

        # Use the model to generate the QDMR
        qdmr_ids = self.model.generate(t['input_ids'],
                                       max_length=256,
                                       decoder_start_token_id=None,
                                       use_cache=True,
                                       num_beams=7,
                                       early_stopping=True,
                                       num_return_sequences=num_generations)

        # Parse the predictions into a list of list of strings
        candidate_plans = self._parse_predictions(qdmr_ids)

        # Convert these to a list of list of Steps
        candidate_steps = [[Step(qdmr=s, reference_id=i, q_id=utterance.q_id, step_type="simple") for i, s in enumerate(plan, start=1)] for plan in candidate_plans]

        # Perform some cleaning of the steps
        return [self._clean_steps(p) for p in candidate_steps]

    @staticmethod
    def _lmap(f, x):
        return list(map(f, x))

    def _ids_to_clean_text(self, generated_ids):
        gen_text = [self.tokenizer.decode(
            g, skip_special_tokens=True, clean_up_tokenization_spaces=True
        ) for g in generated_ids]
        return self._lmap(str.strip, gen_text)

    def _parse_predictions(self, token_ids) -> List[List[str]]:
        # Parse the tokens of the candidate plans into plain text/strings
        clean_candidates_text = self._ids_to_clean_text(token_ids)

        # Remove additional white space in the strings
        clean_candidates_text = [" ".join(c.split()) for c in clean_candidates_text]

        # Remove duplicate plans while maintaining the order of the original list
        unique_clean_candidates_text = list({k:None for k in clean_candidates_text}.keys())

        # Produce the list of step strings for each unique candidate plan
        return [c.split(' @@SEP@@ ') for c in unique_clean_candidates_text]

    def _clean_steps(self,
                     steps: List[Step]) -> List[Step]:
        """
        Perform some additional cleaning on the steps.
        :param steps: The list of steps to clean.
        :return: The cleaned list of steps.
        """
        # Remove the "return " at the start of the decomposition
        for s in steps:
            s.qdmr = s.qdmr.replace("return ", "", 1)
        return steps