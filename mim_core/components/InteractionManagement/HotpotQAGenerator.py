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
HotpotQA Language Generator

April 6, 2021
Authors: C3 Lab
"""

import numpy as np
from mim_core.structs.MQR import HierarchicalMQR
from mim_core.utils.result_utils import is_null_result
from mim_core.components.InteractionManagement.LanguageGenerator import LanguageGenerator

class HotpotQAGenerator(LanguageGenerator):
    """
    Generates language in a way that aligns with the expected output for evaluation on the HotpotQA dataset.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.single_entity_response = kwargs.get('single_entity_response', False)

    def generate_response(self, mqr: HierarchicalMQR) -> str:
        """
        Generates output text to send out of the system.
        :param mqr: The plan from which to generate the response.
        :return: The output text as a string.
        """
        final_result = mqr.root.get_all_results()

        # Process answers (convert booleans to yes/no answers or take the text span as the response)
        if is_null_result(final_result):
            return "no"
        elif (isinstance(final_result['answer'].iloc[0], bool) or isinstance(final_result['answer'].iloc[0], np.bool_)) and final_result['answer'].iloc[0]:
            response = 'yes'
        elif (isinstance(final_result['answer'].iloc[0], bool) or isinstance(final_result['answer'].iloc[0], np.bool_)) and not final_result['answer'].iloc[0]:
            response = 'no'
        else:
            if self.single_entity_response and len(final_result) > 0:
                response = final_result['answer'].astype(str).iloc[0]
            else:
                response = ", ".join(self._get_pretty_date(a) for a in final_result['answer'].astype(str).to_list())

        return response
