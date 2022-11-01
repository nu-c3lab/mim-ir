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
TREC Answer Type Classifier

November 16, 2021
Authors: C3 Lab
"""

import torch
from typing import List
from transformers import pipeline
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

from mim_core.components.QuestionAnalysis.AnswerTypeClassifier import AnswerTypeClassifier

class TRECAnswerTypeClassifier(AnswerTypeClassifier):
    """
    A class that determines the expected answer type for a question using a model trained on a subset of
    the TREC-50 dataset and boolean questions coming from the SMART answer typing dataset.

    Categories
    -----------------
    NUMERIC
    DATE
    LOCATION
    HUMAN_GROUP
    HUMAN_INDIVIDUAL
    ENTITY
    ABBREVIATION
    DESCRIPTION
    EXPLANATION
    BOOLEAN
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.input_type = 'question'
        self.use_gpu = kwargs.get('use_gpu', False) and torch.cuda.is_available()

        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('msterbentz/distilbert-base-answer-type-10')
        self.nlp = pipeline('text-classification',
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=0 if self.use_gpu else -1)

    def classify(self,
                question: str) -> List[str]:
        """
        Determine the expected answer type(s) of the given question.
        :param question: The question to determine the expected answer type(s) for.
        :return: A list of expected answer type(s).
        """

        # Use the model to classify the input question
        results = self.nlp(question)     # Note: this returns a list of results i.e. [{'label': 'NUMERIC', 'score': 0.997}]
        return [r['label'] for r in results]
