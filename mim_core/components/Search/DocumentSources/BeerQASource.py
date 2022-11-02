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
BeerQA Document Source

April 19, 2022
Authors: C3 Lab
"""

import re
import json
import torch
from typing import List, Dict
from bs4 import BeautifulSoup
from urllib.parse import unquote
from elasticsearch import Elasticsearch
from elasticsearch.client.indices import IndicesClient
from mim_core.structs.Document import Document, Passage
from mim_core.components.models import get_model
from mim_core.components.Search.DocumentSources.DocumentSource import DocumentSource
from mim_core.components.Search.DocumentSources.PassageRanker import PassageRanker

core_title_matcher = re.compile('([^()]+[^\s()])(?:\s*\(.+\))?')
core_title_filter = lambda x: core_title_matcher.match(x).group(1) if core_title_matcher.match(x) else x

class BeerQASource(DocumentSource):
    """
    A class for providing a connection to documents/articles from Wikipedia.

    Example config entry:

        {
            "type": "BeerQA",
            "index_name": "beerqa_wiki_doc_para",
            "max_documents": 3,
            "max_passages": 10,
            "passage_ranking_mode": "bm25"
        }

    """

    def __init__(self, **kwargs):
        super().__init__()
        self.creds_username = kwargs.get('creds_username', '')
        self.creds_password = kwargs.get('creds_password', '')
        self.creds_uri = kwargs.get('creds_uri', None)
        if not self.creds_uri:
            raise ValueError("URI is required to configure BeerQASource.")
        self.document_client = Elasticsearch(self.creds_uri, http_auth=(self.creds_username, self.creds_password) if self.creds_password else None)
        self.index_client = IndicesClient(self.document_client)
        self.index_name = kwargs.get('index_name', 'beerqa_wiki_doc_para')
        self.max_docs = kwargs.get('max_documents', 5)
        self.max_passages = kwargs.get('max_passages', 10)
        self.generate_keywords = kwargs.get('generate_keywords', False)     # generate keywords if none are supplied by the user
        self.spacy_model = get_model(kwargs.get('spacy_model', 'en_core_web_trf'))
        self.passage_ranking_mode = kwargs.get('passage_ranking_mode', None)
        self.passage_reranker = PassageRanker(**{'use_gpu': kwargs.get('use_gpu', False) and torch.cuda.is_available(),
                                                 'ranking_mode': self.passage_ranking_mode})


    def build_query(self,
                    question: str,
                    keywords: List[str] = None) -> Dict:
        """
        Builds the dictionary used for querying the ElasticSearch index.
        :param question: The question to search the passage content with and embed for additional searching.
        :param keywords: A list of keywords that are used for searching passage titles.
        :return:
        NOTE: According to ElasticSearch docs, scoring based on vectors happens after an initial set of docs are returned.
                It is not used as an initial filter over the full document set.
        """

        # return self._passage_query_constructor(question, topn=50)
        # return self._doc_query_constructor(question, topn=10)
        # return {
        #     "multi_match": {
        #         "query": question,
        #         "fields": ['title^1.25', 'title_unescape^1.25', 'text', 'doc_text']
        #     }
        # }
        query = {
            "bool":
                {
                    "should": [
                        {"match": {"text": question}},
                        {"match": {"doc_text": question}}
                    ],
                    "minimum_should_match": 1
                }
        }

        if keywords:
            keyword_query = {
                "bool": {
                    "should": [ ],
                    # "minimum_should_match": 1
                }
            }
            for keyword in keywords:
                keyword_query["bool"]["should"].append({'match': {'title_unescape': {"query": keyword, "boost": 1.5}}})
                keyword_query["bool"]["should"].append({'match': {'title': {"query": keyword, "boost": 1.5}}})

            query['bool']['should'].append(keyword_query)
        else:
            query['bool']['should'].append({'match': {'title_unescape': {"query": question, "boost": 1.25}}})
            query['bool']['should'].append({'match': {'title': {"query": question, "boost": 1.25}}})

        return query

    def get_documents(self,
                      query: str,
                      q_id: str = None,
                      keywords: List[str] = None) -> List[Document]:
        """
        Searches Wikipedia for articles relevant to the query.
        :param query: The query string to search Wikipedia with.
        :param q_id: A question id for use in testing datasets that require specific documents for a given question.
        :param keywords: A list of keywords that are used for searching article/passage titles.
        :return: A list of documents containing the title and content of relevant Wikipedia articles.
        """

        # Clean up the query string
        query = self.remove_noun_possessives(query)

        if not keywords and self.generate_keywords:
            keywords = self.extract_keywords(query)

        # Get the initial set of documents using BM25
        res = self.document_client.search(index=self.index_name,
                                                        query=self.build_query(query,
                                                                               keywords=keywords),
                                                        size=max(30, self.max_docs),
                                                        request_timeout=60)

        res = [self._extract_one(x) for x in res['hits']['hits']]
        res = self.rerank_with_query(query, res)

        # Convert the results to Document objects and return them
        max_score = max([x['_score'] for x in res])
        return [Document(title=x['title'],
                         content=x['text'],
                         score=x['_score'] / max_score) for x in res][0:self.max_docs]

    def get_passages(self,
                     question: str,
                     documents: List[Document] = None,
                     q_id: str = None,
                     keywords: List[str] = None) -> List[Passage]:
        """
        Gets a list of Passages from the provided Documents.
        Note: This method will also search Documents if none are provided.
        :param question: The query string to search/rerank with.
        :param documents: A list of Documents to pull out the passages from.
        :param q_id: A question id for use in testing datasets that require specific documents for a given question.
        :param keywords: A list of keywords that are used for searching article/passage titles.
        :return: A list of Passages.
        """

        # Make sure we have documents to extract the passages from
        if not documents:
            documents = self.get_documents(question, q_id=q_id, keywords=keywords)

        # Get the paragraphs of the articles that were retrieved
        doc_passages = []
        for d in documents:
            doc_passages.extend(self.extract_passages(d))

        # Rerank the passages
        if self.passage_ranking_mode == 'bm25':
            doc_passages = self.passage_reranker.rerank_passages_bm25(question, doc_passages)
        elif self.passage_ranking_mode in ['general_sentence', 'multiqa_sentence']:
            doc_passages = self.passage_reranker.rerank_passages_sentence_embedding(question, doc_passages)
        elif self.passage_ranking_mode in ['msmarco_cross', 'qnli_cross']:
            doc_passages = self.passage_reranker.rerank_passages_cross_encoding(question, doc_passages)

        # Return the specified number of passages
        return doc_passages[:self.max_passages]

    def extract_hyperlinks(self,
                           passage: str) -> (str, List[str]):
        """
        Extracts the hyperlinks from a given Wikipedia passage.
        :param passage: A passage that may contain hyperlinks.
        :return: A tuple containing the passage without links and the list of links contained within.
        """
        # Replace left and right brackets with actual tags
        p = passage.replace('&lt;', '<')
        p = p.replace('&gt;', '>')

        soup = BeautifulSoup(p, "html.parser")
        links = [link.get('href') for link in soup.find_all("a")]
        links = [unquote(link) for link in links if link]
        text = soup.get_text()

        return text, links

    def extract_passages(self,
                         doc: Document) -> List[Passage]:
        """
        Converts the given document representing a full Wikipedia article into paragraph documents.
        :param doc: A full Wikipedia article.
        :return: A list of paragraphs from that article.
        """

        def is_title(tokens: List[str]) -> bool:
            return len(tokens) < 7

        # Init list to return
        passage_docs = []

        # Split into paragraphs
        passages = doc.content.split('\n')
        current_title = doc.title

        for p_idx, p in enumerate(passages):

            # Skip whitespace "passages"
            if p == '' or p is None or p.isspace():
                continue

            # Tokenize the current passage   # TODO: use something more robust for this?
            p_tokens = p.split(' ')

            # Check if the current passage is a title   # TODO: Make this check more robust?
            if is_title(p_tokens):
                current_title = doc.title + ' - ' + p
                continue

            # Remove hyperlinks
            passage_content, passage_links = self.extract_hyperlinks(p)

            # Grab the current passage
            passage_docs.append(Passage(title=current_title, content=passage_content, document_title=doc.title, score=doc.score))

        # Add the index of the passage to the Document objects
        for idx, passage in enumerate(passage_docs):
            passage.index = idx

        return passage_docs

    def extract_keywords(self,
                         question: str) -> List[str]:
        spacy_parse = self.spacy_model(question)
        entities = []
        for chunk in spacy_parse.noun_chunks:
            for token in chunk:
                if token.pos_ == "PROPN":
                    entities.append(chunk.text)
                    break
        if len(entities) == 0:
            for chunk in spacy_parse.noun_chunks:
                for token in chunk:
                    if token.pos_ == "NOUN":
                        entities.append(chunk.text)
                        break

        return entities

    def remove_noun_possessives(self,
                                question: str) -> str:
        """
        Removes possessive indicators from the given string.
        Example: "What is this year's model?" -> "What is this year model?"
        :param question: The string to remove the possessive markers from.
        :return: The processed string.
        """

        s = question
        if "'s " in s:
            s = s.replace("'s ", " ")
        if "s' " in s:
            s = s.replace("s' ", "s ")
        if s[-2:] == "'s" or s[-2:] == "s'":    # End of the string has possessive
            s = s[:-2]
        return s

    ########################################################################
    # The following code was taken from the BeerQA repo
    ########################################################################

    def _passage_query_constructor(self, query, topn=50, exclude_title=False):
        fields = ["text"] if exclude_title else ["title^1.25", "title_unescape^1.25", "text"]
        return {
                "bool": {
                    "must": [
                        {"multi_match": {
                            "query": query,
                            "fields": fields,
                        }},
                    ],
                    "should": [
                        {"has_parent": {
                            "parent_type": "doc",
                            "score": True,
                            "query": {
                                "multi_match": {
                                    "query": query,
                                    "fields": [x if x != 'text' else 'doc_text' for x in fields],
                                    "boost": 0.2
                                },
                            }
                        }}
                    ],
                    "filter": [
                        {"term": {
                            "doctype": "para"
                        }}
                    ],
                }
            }

    def _doc_query_constructor(self, query, topn=10, title=None, match_phrase=False, title_only=True):
        query_type = 'match_phrase' if match_phrase else 'match'
        query_field = 'title' if title_only else 'doc_text'
        return {
                "bool": {
                    "must": [
                        {query_type: {query_field: query}}
                    ],
                    "filter": [
                        {"term": {"doctype": "doc"}}
                    ]
                }
            }

    def rerank_with_query(self, query, results):
        def score_boost(item, query):
            score = item['_score']
            core_title = core_title_filter(item['title_unescape'])
            if query.startswith('The ') or query.startswith('the '):
                query1 = query[4:]
            else:
                query1 = query
            if query == item['title_unescape'] or query1 == item['title_unescape']:
                score *= 1.5
            elif query.lower() == item['title_unescape'].lower() or query1.lower() == item['title_unescape'].lower():
                score *= 1.2
            elif item['title'].lower() in query:
                score *= 1.1
            elif query == core_title or query1 == core_title:
                score *= 1.2
            elif query.lower() == core_title.lower() or query1.lower() == core_title.lower():
                score *= 1.1
            elif core_title.lower() in query.lower():
                score *= 1.05

            item['_score'] = score
            return item

        return list(sorted([score_boost(item, query) for item in results], key=lambda item: -item['_score']))

    def _extract_one(self, item, lazy=False):
        res = {k: item['_source'][k] if k != 'text' else item['_source'].get(k, item['_source'].get('doc_text', None))
               for k in ['id', 'url', 'title', 'text', 'title_unescape', 'docid']}
        res['_score'] = item['_score']
        res['data_object'] = item['_source']['original_json'] if lazy else json.loads(item['_source']['original_json'])

        return res
