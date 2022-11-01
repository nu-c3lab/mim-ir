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
Conversation Manager

March 27, 2021
Authors: C3 Lab
"""

from mim_core.structs.MQR import HierarchicalMQR
from mim_core.structs.Utterance import Utterance
import mim_core.utils.component_loading as cl_utils

class InteractionManager(object):
    """
    A class used to manage the input from a user and provide output to them.
    """

    def __init__(self, **kwargs):
        self.language_generator = cl_utils.load_generator(kwargs.get('language_generator'))
        self.print_plan = kwargs.get('print_plan', False)

    def analyze_utterance(self,
                          question_text: str,
                          q_id: str = None) -> Utterance:
        """
        Determines what to do with the input utterance to the system.
        :param question_text: The string containing the question.
        :param q_id: The id associated with the question. Used in system evaluation settings.
        :return: Utterance object to be further analyzed by the system.
        """
        return Utterance(text=question_text, q_id=q_id)

    def produce_response(self,
                         mqr: HierarchicalMQR) -> (str, HierarchicalMQR):
        """
        Uses the language generator to determine what output is appropriate to send back to the user.
        :param mqr: The plan from which to produce the final output.
        :return: A string with the output text response.
        """
        # The following two lines are for testing purposes. Remove when done.
        if self.print_plan:
            import json
            print(json.dumps(mqr.to_json(), indent=2))

        return self.language_generator.generate_response(mqr), mqr

