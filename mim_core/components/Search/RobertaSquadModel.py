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
RoBERTa SQuAD Model

April 10, 2021
Authors: C3 Lab
"""

import torch
from typing import List, Tuple
from transformers import pipeline
from mim_core.structs.Document import Document
from mim_core.components.Search.NeuralQAModel import NeuralQAModel

class RobertaSquadModel(NeuralQAModel):
    """
    This search component uses a RoBERTa-base model fine-tuned on
    SQuAD 2.0 to find relevant entities/evidence from a Document.

    An example of the config.json entry for loading this model in Mim looks like the following:

        "project_model": {
            "type": "RobertaSquadModel",
            "topk": 1
        }

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.use_gpu = kwargs.get('use_gpu', False) and torch.cuda.is_available()
        self.model_name = "deepset/roberta-base-squad2"
        self.nlp = pipeline('question-answering',
                            model=self.model_name,
                            tokenizer=self.model_name,
                            device=0 if self.use_gpu else -1)
        self.topk = max(kwargs.get('topk', 1), 1)   # ensure topk is at least 1

    def answer_question(self,
                        question: str,
                        doc: Document) -> List[Tuple[str, float]]:
        """
        Runs the neural model in order to find the answer to the given question within the given document.
        :param question: The question to answer to be answered.
        :param doc: The context from which to derive the answer.
        :return: String representing the answer (can be "" if there is no answer) and the confidence score.
        """

        # Search for the the top k answers
        answers = self.nlp({"question": question, "context": doc.content}, topk=self.topk)

        # Format the answers based on the results
        if type(answers) == list:
            return [(x['answer'], x['score']) for x in answers]
        elif type(answers) == dict:
            return [(answers['answer'], answers['score'])]
        else:
            raise ValueError('Unexpected return type from RobertaSquadModel.')
