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
RoBERTa-Base BoolQ Model

May 25, 2021
Authors: C3 Lab
"""

import torch
from mim_core.structs.Document import Document
from mim_core.components.Search.NeuralQAModel import NeuralQAModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class RobertaBaseBoolQModel(NeuralQAModel):
    """
    This search component uses a RoBERTa-base model (125M params) fine-tuned on
    BoolQ dataset to answer boolean questions.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.use_gpu = kwargs.get('use_gpu', False) and torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_gpu else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = AutoModelForSequenceClassification.from_pretrained('msterbentz/roberta-base-boolq')
        self.model = self.model.to(self.device)

    def answer_question(self,
                        question: str,
                        doc: Document) -> (str, float):
        """
        Runs the neural model in order to find the answer to the given question within the given document.
        :param question: The question to answer to be answered.
        :param doc: The context from which to derive the answer.
        :return: String representing the answer and the confidence/probability.
        """
        # TODO: Make this not truncate the input, but search the entire text for the answer

        # Get the tokenized input
        sequence = self.tokenizer.encode_plus(question, doc.content, return_tensors="pt", max_length=512)['input_ids']
        sequence = sequence.to(self.device)

        # Run the tokenized input through the model
        logits = self.model(sequence)[0]
        probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
        proba_yes = round(probabilities[1], 2)
        proba_no = round(probabilities[0], 2)

        return 'yes', proba_yes