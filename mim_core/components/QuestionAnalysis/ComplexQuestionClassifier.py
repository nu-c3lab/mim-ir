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
Complex Question Classifier (V3)

This model was trained with an additional 20000 simple questions coming from the SQuAD v2 training dataset and ~20000
questions from the high-level BREAK training dataset.

August 22, 2022
Authors: C3 Lab
"""

import re
import torch
from transformers import pipeline
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification

class ComplexQuestionClassifier():
    """
    A class that determines the complexity of a given question as either "simple" or "complex".
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.input_type = 'question'
        self.use_gpu = kwargs.get('use_gpu', False) and torch.cuda.is_available()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('msterbentz/distilbert-base-complex-question')
        self.nlp = pipeline('text-classification',
                            model=self.model,
                            tokenizer=self.tokenizer,
                            device=0 if self.use_gpu else -1)
        self.nice_labels = {
            "LABEL_0": "simple",
            "LABEL_1": "complex"
        }

        self.complex_addition_pattern = re.compile("(combined (.*?) of|total (.*?) of|sum of|how many (.*?) total)|how many (.*?) are \s*")
        self.complex_boolean_equality_pattern = re.compile("does (.*) have (.*) than (.*)|do (.*) than (.*)|is (.*) than (.*)|did (.*) than (.*)|has (.*) at least (.*)|was (.*) before (.*)|was (.*) after (.*)|does (.*) as (.*)")
        self.complex_boolean_existence_pattern = re.compile("(does|is|has) (there )*(exist |ever been )*(a|an)* (.*) (involving )*(with|longer than|more than|shorter than|smaller than|taller than|higher than) (.*)")
        self.complex_discard_pattern = re.compile("(besides|except|other than) (.*),*(what other|what|who|which other|which) (.*)|what (.*) (besides) (.*) (does|is|was) (.*)|what (.*) (besides|other than|except for) (.*)|(what|which) (.*) but is(n't| not)* (.*)")
        self.complex_intersection_pattern = re.compile("(what|which|who) (.*) both (.*) and (.*)|(who|what) (is|was) (.*) and (.*)|(who|what) (is|was) (.*), (.*), and (.*)")
        self.complex_subtraction_pattern = re.compile("how (many more|long after|much more|much longer|much taller|much shorter|many fewer) (.*)(is|was|does) (.*)|how (long after|soon after) (.*) (is|was|did) (.*)")

    def classify(self,
                question: str) -> str:
        """
        Determine the complexity of the given question..
        :param question: The question to determine the complexity for.
        :return: A string that is either "complex" or "simple".
        """

        # Check for a complex question using the rules
        if re.search(self.complex_addition_pattern, question.lower()) or \
           re.search(self.complex_boolean_equality_pattern, question.lower()) or \
           re.search(self.complex_boolean_existence_pattern, question.lower()) or \
           re.search(self.complex_discard_pattern, question.lower()) or \
           re.search(self.complex_intersection_pattern, question.lower()) or \
           re.search(self.complex_subtraction_pattern, question.lower()):
            return "complex"
        else:
            # If no match is found, use the model to classify to classify the input question
            results = self.nlp(question)
            return [self.nice_labels[r['label']] for r in results][0]
