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
Mim

March 27, 2021
Authors: C3 Lab
"""

import json
from typing import List, Dict, Union

from mim_core.structs.MQR import HierarchicalMQR
from mim_core.components.QuestionAnalysis.QuestionAnalyzer import QuestionAnalyzer
from mim_core.components.InteractionManagement.InteractionManager import InteractionManager
from mim_core.components.AnswerEngine.AnswerEngine import AnswerEngine


class Mim(object):
    """
    A class for containing and managing all of the Mim components.
    """

    def __init__(self, config: str):
        self.load_config(config)

    def answer_question(self,
                        question_text: str,
                        q_id: str = None) -> (str, HierarchicalMQR):
        """
        Uses the set of specified components to answer the given question.
        :param question_text: The string containing the question.
        :param q_id: The id of the question. Used as part of system evaluation paradigms.
        :return: String containing Mim's response/answer to the question.
        """

        # Produce a formatted Utterance by using the ConversationManager
        utterance = self.interaction_manager.analyze_utterance(question_text, q_id)

        # Analyze the question to produce the MQR / question answering plan
        mqr = self.question_analyzer.analyze_question(utterance)

        # Send the MQR to the Answer Engine to be executed
        mqr = self.answer_engine.execute_plan(mqr)

        # Produce output of the desired form using the ConversationManager's LanguageGenerator
        return self.interaction_manager.produce_response(mqr)

    def load_config(self,
                        conf: Union[str, dict]) -> None:
        """
        Uses a configuration list to set up this instance of Mim.
        :param config: A list of components with their type and their arguments,
                        or a string denoting the location of a json file containing this list of dicts.
        :return: None
        """

        if isinstance(conf, str):
            with open(conf) as json_config:
                config = json.load(json_config)
        elif isinstance(conf, dict):
            config = conf
        else:
            print("Unable to load configuration. The provided configuration is of an unknown type.")
            return None

        if 'interaction_manager' in config:
            self.interaction_manager = InteractionManager(**config['interaction_manager'])
        else:
            print("Missing interaction_manager definition in the configuration. See the example config for usage.")
            return None

        print("finished loading interaction manager")

        if 'question_analyzer' in config:
            self.question_analyzer = QuestionAnalyzer(**config['question_analyzer'])
        else:
            print("Missing question_analyzer definition in the configuration. See the example config for usage.")
            return None

        print("finished loading question analyzer")

        if 'answer_engine' in config:
            self.answer_engine = AnswerEngine(**config['answer_engine'])
        else:
            print("Missing answer_engine definition in the configuration. See the example config for usage.")
            return None

        print("finished loading answer engine")
