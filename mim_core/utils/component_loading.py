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
Component Loading Utilities

A set of utility functions for loading Mim components.

Authors: C3 Lab
"""

# TODO: May want to throw errors when malformed configuration is attempted to be loaded since resulting/default behavior will likely be undesired

from typing import Dict

from mim_core.components.InteractionManagement.LanguageGenerator import LanguageGenerator
from mim_core.components.InteractionManagement.HotpotQAGenerator import HotpotQAGenerator

from mim_core.components.QuestionAnalysis.QuestionDecomposer import QuestionDecomposer
from mim_core.components.QuestionAnalysis.T5HighLevelDecomposer import T5HighLevelDecomposer

from mim_core.components.QuestionAnalysis.QuestionGenerator import QuestionGenerator
from mim_core.components.QuestionAnalysis.RuleBasedQuestionGenerator import RuleBasedQuestionGenerator

from mim_core.components.Ontology.Ontology import Ontology
from mim_core.components.Ontology.WikidataOntology import WikidataOntology

from mim_core.components.QuestionAnalysis.AnswerTypeClassifier import AnswerTypeClassifier
from mim_core.components.QuestionAnalysis.TRECAnswerTypeClassifier import TRECAnswerTypeClassifier

from mim_core.components.AnswerEngine.AnswerEvaluator import AnswerEvaluator
from mim_core.components.AnswerEngine.SourceScorer import SourceScorer
from mim_core.components.AnswerEngine.ScoringFunction import ScoringFunction
from mim_core.components.AnswerEngine.ConfidenceScorer import ConfidenceScorer

from mim_core.components.Search.QuestionAnsweringComponent import QuestionAnsweringComponent
from mim_core.components.Search.HighLevelNeuralSearchComponent import HighLevelNeuralSearchComponent
from mim_core.components.Search.IRRRAnsweringComponent import IRRRAnsweringComponent
from mim_core.components.Search.HybridAnsweringComponent import HybridAnsweringComponent
from mim_core.components.Search.DocumentSources.DocumentSource import DocumentSource
from mim_core.components.Search.DocumentSources.BeerQASource import BeerQASource
from mim_core.components.Search.NeuralQAModel import NeuralQAModel
from mim_core.components.Search.RobertaSquadModel import RobertaSquadModel
from mim_core.components.Search.RobertaBaseBoolQModel import RobertaBaseBoolQModel


def load_generator(config: Dict[str, str]) -> LanguageGenerator:
    """
    Loads a language generator specified by the configuration.
    :param config: A dictionary containing the specs for the language generator.
    :return: The language generator.
    """
    if config is None or 'type' not in config:
        return HotpotQAGenerator()
    elif config['type'] == 'HotpotQAGenerator':
        return HotpotQAGenerator(**config)
    else:
        return HotpotQAGenerator()

def load_high_level_question_decomposer(config: Dict[str, str]) -> QuestionDecomposer:
    """
    Loads the high level question decomposer specified by the configuration. Default: T5HighLevelDecomposer
    :param config: A dictionary containing the specs for the question decomposer.
    :return: The question decomposer.
    """
    if config is None or 'type' not in config:
        return T5HighLevelDecomposer()
    elif config['type'] == 'T5HighLevelDecomposer':
        return T5HighLevelDecomposer(**config)
    else:
        return T5HighLevelDecomposer()

def load_question_generator(config: Dict[str, str]) -> QuestionGenerator:
    """
    Loads the question generator specified by the configuration. Default: Rule-based generator
    :param config: A dictionary containing the specs for the question generator.
    :return: The question generator.
    """
    if config is None or 'type' not in config:
        return RuleBasedQuestionGenerator()
    elif config['type'] == 'RuleBasedQuestionGenerator':
        return RuleBasedQuestionGenerator()
    else:
        return RuleBasedQuestionGenerator()

def load_ontology(config: Dict[str, str]) -> Ontology:
    """
    Loads the ontology specified by the configuration.
    :param config: A dictionary containing the specs for the ontology.
    :return: The ontology.
    """
    if config is None or 'type' not in config:
        return WikidataOntology()
    elif config['type'] == 'WikidataOntology':
        return WikidataOntology(**config)
    else:
        return WikidataOntology()

def load_answer_type_classifier(config: Dict[str, str]) -> AnswerTypeClassifier:
    """
    Loads the answer type classifier specified by the configuration.
    :param config: A dictionary containing the specs for the answer type classifier.
    :return: The answer type classifier.
    """
    if config is None or 'type' not in config:
        return TRECAnswerTypeClassifier()
    elif config['type'] == 'TRECAnswerTypeClassifier':
        return TRECAnswerTypeClassifier(**config)
    else:
        return TRECAnswerTypeClassifier()

def load_answer_evaluator(config: Dict[str, str]) -> AnswerEvaluator:
    """
    Loads the answer evaluator specified by the configuration.
    :param config: A dictionary containing the specs for the answer evaluator.
    :return: The answer evaluator.
    """
    if config is None or 'type' not in config:
        return AnswerEvaluator()
    elif config['type'] == 'AnswerEvaluator':
        return AnswerEvaluator(**config)
    else:
        return AnswerEvaluator()

def load_scoring_function(config: Dict[str, str]) -> ScoringFunction:
    """
    Loads the scoring function specified by the configuration.
    :param config: A dictionary containing the specs for the scoring function.
    :return: The scoring function.
    """
    if config is None or 'type' not in config:
        return None
    elif config['type'] == 'SourceScorer':
        return SourceScorer(**config)
    elif config['type'] == 'ConfidenceScorer':
        return ConfidenceScorer(**config)
    else:
        return None

def load_question_answering_component(config: Dict[str, str]) -> QuestionAnsweringComponent:
    """
    Loads the search component specified by the configuration.
    :param config: A dictionary containing the specs for the search component.
    :return: The search component.
    """
    if config is None or 'type' not in config:
        return None
    elif config['type'] == "HighLevelNeuralSearchComponent":
        return HighLevelNeuralSearchComponent(**config)
    elif config['type'] == "IRRRAnsweringComponent":
        return IRRRAnsweringComponent(**config)
    elif config['type'] == "HybridAnsweringComponent":
        return HybridAnsweringComponent(**config)
    else:
        return None

def load_document_source(config: Dict[str, str]) -> DocumentSource:
    """
    Loads the document source specified by the configuration.
    :param config: A dictionary containing the specs for the document source.
    :return: The document source.
    """
    if config is None or 'type' not in config:
        return None
    elif config['type'] == 'BeerQASource':
        return BeerQASource(**config)
    else:
        return None

def load_neural_qa_model(config: Dict[str, str]) -> NeuralQAModel:
    """
    Loads the neural QA model specified by the configuration.
    :param config: A dictionary containing the specs for the neural QA model.
    :return: The neural QA model.
    """
    if config is None or 'type' not in config:
        return None
    elif config['type'] == 'RobertaSquadModel':
        return RobertaSquadModel(**config)
    elif config['type'] == 'RobertaBaseBoolQModel':
        return RobertaBaseBoolQModel(**config)
    else:
        return None
