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

import torch
from torch import nn
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util, CrossEncoder

from mim_core.structs.Document import Passage

class PassageRanker:

    def __init__(self, **kwargs):
        self.use_gpu = kwargs.get('use_gpu', False) and torch.cuda.is_available()
        self.device = 'cuda:0' if self.use_gpu else 'cpu'
        self.ranking_mode = kwargs.get('ranking_mode', None)
        if self.ranking_mode in ['general_sentence', 'multiqa_sentence']:
            self.sentence_embedder = self._load_sentence_embedder()
        if self.ranking_mode in ['msmarco_cross', 'qnli_cross']:
            self.cross_encoder = self._load_cross_encoder()
        # TODO: Add param to use cumulative/multiplicative score vs. using only the reranked score on the passage???

    def rerank_passages_sentence_embedding(self,
                                           question: str,
                                           passages: List[Passage]) -> List[Passage]:
        """
        Re-ranks the given set of passages based on relevance to the given question using a sentence embedder that
        encodes queries and passages separately and scores them using cosine similarity..
        :param question: A question string to rank the passages with.
        :param passages: A list of Passage objects.
        :return: The list of passages with updated scores sorted in descending order.
        """

        question_embedding = self.sentence_embedder.encode(question, convert_to_tensor=True).to('cpu')

        # Get the embeddings for each of the documents as a Tensor of floats
        document_embeddings = self.sentence_embedder.encode([x.content for x in passages], convert_to_tensor=True).to('cpu')

        top_k = len(passages)
        cos_scores = util.pytorch_cos_sim(question_embedding, document_embeddings)[0]
        top_embeddings = torch.topk(cos_scores, k=top_k)

        for score, idx in zip(top_embeddings[0], top_embeddings[1]):
            passages[idx].score *= float(score)  # Update the score of the result

        # Normalize the scores for the final document outputs
        max_score = max([x.score for x in passages])
        for doc in passages:
            doc.score = doc.score / max_score if max_score != 0 else 0.0

        # Sort the results by final score
        passages.sort(key=lambda x: x.score, reverse=True)

        return passages

    def rerank_passages_bm25(self,
                             question: str,
                             passages: List[Passage]) -> List[Passage]:
        """
        Re-ranks the given set of passages based on relevance to the given question using the Okapi BM25 algorithm.
        :param question: A question string to rank the passages with.
        :param passages: A list of Passage objects.
        :return: The list of passages with updated scores sorted in descending order.
        """

        # TODO: Preprocess the passages

        # Build the BM25 corpus
        corpus = [p.content.lower() for p in passages]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)

        # Retrieve and score documents in the corpus based on the input question
        tokenized_question = question.lower().split(" ")
        doc_scores = bm25.get_scores(tokenized_question)

        # Update the score of the passages
        reranked_passages = [passages[idx] for idx, score in enumerate(doc_scores)]
        for idx, p in enumerate(reranked_passages):
            p.score *= doc_scores[idx]

        # Sort the reranked passages by descending order of score
        reranked_passages.sort(key=lambda x: x.score, reverse=True)

        return reranked_passages

    def rerank_passages_cross_encoding(self,
                                       question: str,
                                       passages: List[Passage]) -> List[Passage]:

        """
        Re-ranks the given set of passages based on relevance to the given question using a cross encoder model.
        :param question: A question string to rank the passages with.
        :param passages: A list of Passage objects.
        :return: The list of passages with updated scores sorted in descending order.
        """

        # Generate an embedding for the question
        model_inputs = [[question, d.content] for d in passages]

        scores = self.cross_encoder.predict(model_inputs, activation_fct=nn.Sigmoid())

        # Sort the scores in decreasing order
        results = [{'input': inp, 'score': score, 'doc_idx': idx} for inp, score, idx in
                   zip(model_inputs, scores, range(0, len(model_inputs)))]
        results = sorted(results, key=lambda x: x['score'], reverse=True)

        # Update the docs with the scores from top_results
        for r in results:
            passages[r['doc_idx']].score *= r['score']    # TODO: May want to change how we produce final doc retrieval score here

        # Get the Documents with the highest scores
        docs = [passages[d['doc_idx']] for d in results]      # TODO: May want to normalize the scores here again

        return docs

    def _load_sentence_embedder(self) -> SentenceTransformer:
        """
        Loads the document embedder to use for sentence embedding based retrieval.
        :return: A sentence embedding model that can be used to index and retrieve semantically similar sentences.
        """

        if self.ranking_mode == 'general_sentence':
            return SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        if self.ranking_mode == 'multiqa_sentence':
            return SentenceTransformer('multi-qa-mpnet-base-cos-v1', device=self.device)

    def _load_cross_encoder(self) -> CrossEncoder:
        """
        Loads the document cross encoder to use for cross-encoding based retrieval.
        :return: A CrossEncoder model that can be used to index and retrieve semantically similar documents.
        """

        if self.ranking_mode == 'qnli_cross':
            return CrossEncoder('cross-encoder/qnli-electra-base', device=self.device)
        if self.ranking_mode == 'msmarco_cross':
            return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=self.device)
