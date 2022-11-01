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

import re
import spacy
from pathlib import Path
from os import path
from typing import List  # edit?

from mim_core.structs.Step import Step
from mim_core.components.QuestionAnalysis.QuestionGenerator import QuestionGenerator
from mim_core.components.QuestionAnalysis.QDMRParser import QDMRProgramBuilder
from mim_core.components.models import get_model


class RuleBasedQuestionGenerator(QuestionGenerator):
    """
    A class for building semantically-informed questions corresponding to decomposition steps within MQR objects.
    """

    def __init__(self):
        self.nlp = get_model("en_core_web_sm")
        if not path.exists(Path(__file__).parent / 'data/human_roots.txt'):
            raise FileNotFoundError("Could not locate human words file. "
                                    "Please ensure that it's in the QuestionAnalysis folder.")
        with open(Path(__file__).parent / 'data/human_roots.txt', encoding='utf-8') as f:
            self.human_roots = f.readline().split()

    def generate_question(self,
                          steps: List[Step]) -> List[str]:
        """
        Generates a question from an input QDMR.
        :param steps: A list of Step objects containing QDMR.
        :return: List[string or None] representing the questions that were generated.
        """

        # Join QDMR steps on '@@SEP@@' special token, used as a step separator during training
        decomposition_steps = [step.qdmr for step in steps]
        step_parse = QDMRProgramBuilder(self.steps_to_allen(decomposition_steps))
        decomposition_operators = step_parse.get_operators()

        for step_index, step_text in enumerate(decomposition_steps):
            step_operator = decomposition_operators[step_index]
            return self.generate_question_from_string(step_text, step_operator)

    def generate_question_from_string(self,
                                      step_text: str,
                                      step_operator: str) -> List[str]:
        """
        Generates a question from a given QDMR step string and the step operator.
        :param step_text: A single step from which to generate a question.
        :param step_operator: The QDMR operator for this step.
        :return: A list of question candidates.
        """

        rule_based_questions = []
        question_generated = None

        # Tokenize and normalize the input
        original_step_split = step_text.strip().split()
        step_text = step_text.strip().lower()

        # Do an initial check to see if the first word of the input is a question word
        question_words = ['who', 'what', 'when', 'where', 'how', 'why','which',
                          'has', 'have', 'had',
                          'is', 'are', 'was', 'were', 'will',
                          'did', 'does']

        if len(original_step_split) > 0:
            first_word = original_step_split[0].lower()
            if first_word in question_words:
                question_generated = step_text.capitalize()
                question_generated += "?" if step_text[-1] != "?" else ""

        # Parse the step text
        step_doc = self.nlp(step_text)
        step_text = ' ' + step_text     # Note: the extra ' ' at the start messes up Spacy's parser

        if question_generated is None and ' which ' in step_text:
            question_generated = self.generate_for_which(step_text, step_doc)

        if question_generated is None and ' who ' in step_text and step_operator != 'filter':
            question_generated = self.generate_for_who(step_text, step_doc)
        if question_generated is None and ' where ' in step_text:
            question_generated = self.generate_for_where(step_text, step_doc)
        if question_generated is None and step_text.startswith(' if '):
            question_generated = self.generate_for_if(step_text, step_doc)
        if question_generated is None:
            question_generated = self.generate_for_else(step_text)
        if question_generated is None:
            if step_operator in ['select', 'project']:
                question_generated = self.generate_for_select_project(step_text, step_doc)
            elif step_operator == 'filter':
                split_check = step_text.split()
                if split_check[1] == 'who':
                    split_check[1] = 'that'
                    step_text = ' ' + ' '.join(split_check)
                    step_doc = self.nlp(step_text)
                    question_generated = self.generate_for_filter(step_text, step_doc, who_bool=True)
                else:
                    question_generated = self.generate_for_filter(step_text, step_doc)
        # post-process
        if question_generated is not None:
            question_generated = self.postprocess_question(question_generated, original_step_split)
        rule_based_questions.append(question_generated)
        return rule_based_questions

    @staticmethod
    def generate_for_which(step, doc):
        dep_list = [token.dep_ for token in doc]
        if step.startswith(' which '):
            return step[1:].capitalize() + '?'
        which_index = [token.text for token in doc].index('which')
        first_nsubj = dep_list.index('nsubj') if 'nsubj' in dep_list else len(dep_list)
        if doc[which_index].dep_ == 'nsubj' and doc[which_index].head.pos_ in ['AUX', 'VERB']:
            if first_nsubj == which_index or doc[first_nsubj].pos_ not in ['DET', 'PRON']:
                if doc[1].pos_ == 'DET':
                    start_index = 2
                else:
                    start_index = 1

                return ' '.join(['Which'] + [token.text for token in doc[start_index:which_index]]
                                + [token.text for token in doc[which_index + 1:]]) + '?'
        elif doc[which_index].dep_ == 'nsubjpass' and first_nsubj >= which_index:
            if doc[1].pos_ == 'DET':
                start_index = 2
            else:
                start_index = 1
            return ' '.join(['Which'] + [token.text for token in doc[start_index:which_index]]
                            + [token.text for token in doc[which_index + 1:]]) + '?'
        return None

    @staticmethod
    def generate_for_who(step, doc):
        who_index = [token.text for token in doc].index('who')
        if step.startswith(' who ') or step.startswith(' for who '):
            return step[1:].capitalize() + '?'
        elif doc[who_index].dep_ in ['nsubj', 'nsubjpass']:
            # TODO: need check here that 'who' is first nsubj/nsubjpass
            if re.match(r"^ #\d+ who", step):
                # TODO: can predict is/are from 'who's head tag -- unimplemented
                return ' '.join(['Who is'] + [token.text for token in doc[:who_index]]
                                + ['that'] + [token.text for token in doc[who_index + 1:]]) + '?'
            else:
                aux_verb = None
                for token in doc:
                    if token.dep_ in ['ROOT', 'nsubj']:
                        if token.tag_ in ['NN', 'NNP']:
                            aux_verb = 'is'
                        elif token.tag_ in ['NNS', 'NNPS']:
                            aux_verb = 'are'
                        break
                if aux_verb is not None:
                    det = ''
                    if doc[1].tag_ != 'DT':
                        det = 'the'

                    return ' '.join(['Who', aux_verb, det] +
                                    [token.text for token in doc[:who_index]] +
                                    ['that'] + [token.text for token in doc[who_index + 1:]]) + '?'
        return None

    @staticmethod
    def generate_for_where(step, doc):
        where_index = [token.text for token in doc].index('where')
        if step.startswith(' where '):
            return step[1:].capitalize() + '?'
        elif ' #' not in step:
            root_index = [token.dep_ for token in doc].index('ROOT')
            if not any([token.dep_ == 'prep' for token in doc[root_index:where_index]]):
                root_tag = doc[root_index].tag_
                aux_verb = None
                if root_tag in ['NN', 'NNP']:
                    aux_verb = 'is'
                elif root_tag in ['NNS', 'NNPS']:
                    aux_verb = 'are'

                det = ''
                if doc[1].tag_ != 'DT':
                    det = 'the'
                if aux_verb is not None:
                    return ' '.join(['Where', aux_verb, det] + [token.text for token in doc[:where_index]] +
                                    ['that'] + [token.text for token in doc[where_index + 1:]]) + '?'
        return None

    @staticmethod
    def generate_for_if(step, doc):
        aux_list = ['is', 'are', 'was', 'were', 'will', 'did', 'does']
        split_step = step.split()
        if step.startswith(' if #'):
            # special handling here since QDMR references often confuse spacy's pos tagging
            verb = split_step[2]
            if verb in aux_list:
                return ' '.join([verb] + split_step[1:2] + split_step[3:]).capitalize() + '?'
            else:
                verb_lemma = doc[4].lemma_
                return ' '.join(['Does'] + split_step[1:2] + [verb_lemma] + split_step[3:]).capitalize() + '?'
        else:
            verb_index_in_doc = None
            for ix in range(len(doc)):
                if doc[ix].pos_ in ['VERB', 'AUX']:
                    verb_index_in_doc = ix
                    break
            root_index_in_doc = None
            for ix in range(len(doc)):
                if doc[ix].dep_ == 'ROOT':
                    root_index_in_doc = ix
                    break

            if verb_index_in_doc is not None:
                verb = doc[verb_index_in_doc].text
                if doc[verb_index_in_doc].dep_ in ['aux', 'auxpass'] or verb in aux_list:
                    verb_index_in_split = split_step.index(verb)
                    return ' '.join([verb] + split_step[1:verb_index_in_split] +
                                    split_step[verb_index_in_split + 1:]).capitalize() + '?'
                else:
                    verb = doc[root_index_in_doc].text
                    verb_index_in_split = split_step.index(verb)
                    if verb in aux_list:
                        return ' '.join([verb] + split_step[1:verb_index_in_split] +
                                        split_step[verb_index_in_split + 1:]).capitalize() + '?'
                    else:
                        # tricky to choose verb tense here, since QDMRs often mess up word tense.
                        # e.g. "if kim kardashian produce #1" -- produce is present-tense, but should either be
                        # "produced" or "produces" given the singular subject. We're missing info here --
                        # the questions of whether she "produced" or "produces" a show are different.
                        # rules:
                        # - prepend "Does"/"do" if step verb is 3rd person singular present (VBZ)
                        # - prepend "Did" if step verb is past tense (VBD)
                        # - else, prepend "did" (lossy here)
                        verb_tag = doc[root_index_in_doc].tag_
                        verb_lemma = doc[root_index_in_doc].lemma_
                        if verb_tag == 'VBZ':
                            plural = False
                            for ix in range(len(doc)):
                                if doc[ix].dep_ == 'nsubj':
                                    plural = doc[ix].tag_ in ['NNS', 'NNPS']
                                    break
                            question_start = 'Do' if plural else 'Does'
                            return ' '.join([question_start] + split_step[1:verb_index_in_split] +
                                            [verb_lemma] + split_step[verb_index_in_split + 1:]) + '?'
                        else:  # includes verb_tag == 'VBD'
                            return ' '.join(['Did'] + split_step[1:verb_index_in_split] +
                                            [verb_lemma] + split_step[verb_index_in_split + 1:]) + '?'
            else:
                # TODO: some examples fall through the cracks due to Spacy pos-tagging errors or QDMR errors.
                # e.g. if hyder ali demolish #1 (demolish tagged as NNP)
                # solution: replace 'if' with 'Did'
                return 'Did ' + ' '.join(split_step[1:]) + '?'

    @staticmethod
    def generate_for_else(step):
        else_list = ['is', 'are', 'was', 'were', 'will', 'did', 'does', 'when', 'why', 'how']
        if any([step.startswith(' ' + x + ' ') for x in else_list]) or ' what ' in step:
            return step[1:].capitalize() + '?'
        return None

    def generate_for_select_project(self, step, doc):

        def get_root(doc):
            # Try using the 'ROOT' term
            for token in doc:
                if token.dep_ == 'ROOT' and token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']:
                    return token

            # Try using the 'nsubj' term
            for token in doc:
                if token.dep_ == 'nsubj' and token.tag_ in ['NN', 'NNS', 'NNP', 'NNPS']:
                    return token

            return None

        root_token = get_root(doc)

        if root_token is not None:
            root_text, root_tag = root_token.text, root_token.tag_
            entities = [ent.text for ent in doc.ents]
            is_entity = any([root_text in e for e in entities])

            # Get the words for the final question
            det = 'the' if (len(doc) > 1 and doc[1].dep_ != 'det' and "'s" not in step and not is_entity) else ''
            question_word = 'Who' if (root_text in self.human_roots) else 'What'
            aux = 'are' if (root_tag in ['NNS', 'NNPS']) else 'is'

            return ' '.join([question_word, aux, det] + step.split()).replace('  ', ' ') + '?'

        # if we're here, likely an issue with Spacy being able to pos-tag the step. Default: "What is the" + step
        return 'What is the' + step + '?'

    @staticmethod
    def generate_for_filter(step, doc, who_bool=False):
        aux_list = ['is', 'are', 'was', 'were', 'will', 'did', 'does', 'has', 'have', 'had']
        aux_singluar_dict = {
            'is': 'is',
            'are': 'is',
            'was': 'was',
            'were': 'was',
            'will': 'will',
            'did': 'did',
            'does': 'does'
        }
        split_step = step.split()
        try:
            if split_step[2] in aux_list:
                # Old form without BoolQ
                # return ' '.join(['Which'] + split_step[0:1] + split_step[2:]) + '?'

                # new form for BoolQ
                aux = split_step[2]
                if aux in aux_singluar_dict.keys():
                    aux = aux_singluar_dict[aux].capitalize()
                    if doc[5].pos_ in ['VERB', 'ADV', 'DET', 'ADP']:
                        return ' '.join([aux] + split_step[0:1] + split_step[3:]) + '?'
                    return ' '.join([aux] + split_step[0:1] + ['a'] + split_step[3:]) + '?'
                elif aux in ['has', 'have']:
                    if split_step[3] == 'had':
                        return ' '.join(['Has'] + split_step[0:1] + split_step[3:]) + '?'
                    if who_bool:
                        if doc[5].pos_ in ['VERB', 'AUX', 'ADV']:
                            return ' '.join(['Has'] + split_step[0:1] + split_step[3:]) + '?'
                        else:
                            return ' '.join(['Has'] + split_step[0:1] + ['had'] + split_step[3:]) + '?'
                    return ' '.join(['Does'] + split_step[0:1] + ['have'] + split_step[3:]) + '?'
                else:  # aux == 'had'
                    return ' '.join(['Did'] + split_step[0:1] + ['have'] + split_step[3:]) + '?'

        except IndexError:
            # QDMR error if this occurs
            # e.g. "#1 for"
            return 'Is it' + step + '?'

        wdt = split_step[1]
        first_verb_index = None
        for ix in range(len(doc)):
            if doc[ix].pos_ in ['VERB', 'AUX']:
                first_verb_index = ix
                break

        if first_verb_index is not None:
            is_root = (doc[first_verb_index].dep_ == 'ROOT')
            verb = doc[first_verb_index].text
            try:
                split_step_verb_index = split_step.index(verb)
            except ValueError:
                return 'Is it' + step + '?'
            if is_root and doc[first_verb_index].pos_ == 'VERB':
                verb_lemma = doc[first_verb_index].lemma_
                if doc[-1].dep_ == 'prep' or split_step_verb_index == len(split_step) - 1:
                    if verb != verb_lemma:
                        for token_ix in range(len(split_step)):
                            if split_step[token_ix] == verb:
                                split_step[token_ix] = verb_lemma
                    return ' '.join(['Did'] + split_step[2:] + split_step[0:1]) + '?'

                return 'Did' + step.replace(' ' + wdt, '').replace(verb, verb_lemma) + '?'
            else:
                try:
                    split_step_verb_index = split_step.index(verb)
                except ValueError:
                    # Spacy pos-tagging has failed/gone haywire.
                    # e.g. verb is identified as "d" in #2 with nces id 063879006526
                    return 'Is it' + step + '?'
                if wdt in ['that', 'where']:
                    try:
                        # check we have to do to prevent spacy pos-tagging errors :|
                        _ = float(doc[-1].text[-1:])
                        spacy_error_check = False
                    except Exception as e:
                        spacy_error_check = True
                    if doc[-1].dep_ == 'prep' and spacy_error_check:
                        aux_verb = None
                        for token in doc:
                            if token.pos_ == 'AUX':
                                aux_verb = token.text
                        if aux_verb is not None:
                            aux_loc = split_step.index(aux_verb)
                            if 'where' in split_step:
                                return 'Is it' + step + '?'
                            elif 'that' in split_step:
                                that_loc = split_step.index('that')
                                return aux_verb.capitalize() + ' ' + ' '.join(split_step[that_loc + 1:aux_loc] + split_step[aux_loc + 1:] + split_step[:that_loc]) + '?'
                        else:
                            if verb != doc[first_verb_index].lemma_:
                                for token_ix in range(len(split_step)):
                                    if split_step[token_ix] == verb:
                                        split_step[token_ix] = doc[first_verb_index].lemma_
                            return ' '.join(['Did'] + split_step[2:] + split_step[0:1]) + '?'

                    if doc[first_verb_index].pos_ == 'AUX':
                        if who_bool:
                            return ' '.join([verb.capitalize()] + split_step[0:1] + split_step[split_step_verb_index + 1:]) + '?'
                        return ' '.join([verb.capitalize()] + split_step[2:split_step_verb_index] + ['of'] + split_step[0:1] + split_step[split_step_verb_index + 1:]) + '?'
                    else:
                        verb_tag = doc[first_verb_index].tag_
                        if verb_tag == 'VBZ':
                            if verb != doc[first_verb_index].lemma_:
                                for token_ix in range(len(split_step)):
                                    if split_step[token_ix] == verb:
                                        split_step[token_ix] = doc[first_verb_index].lemma_
                            return ' '.join(['Does'] + split_step[0:1] + split_step[2:]) + '?'
                        elif verb_tag in ['VBD', 'VBN']:
                            if verb != doc[first_verb_index].lemma_:
                                for token_ix in range(len(split_step)):
                                    if split_step[token_ix] == verb:
                                        split_step[token_ix] = doc[first_verb_index].lemma_
                            return ' '.join(['Did'] + split_step[0:1] + split_step[2:]) + '?'
                        elif verb_tag == 'MD':
                            for token_ix in range(len(split_step)):
                                if split_step[token_ix] == 'that':
                                    split_step[token_ix] = 'something that'
                                elif split_step[token_ix] == 'where':
                                    split_step[token_ix] = 'something where'
                            return ' '.join(['Is'] + split_step) + '?'
                        else:
                            if split_step_verb_index == 2:
                                return ' '.join(['Does'] + split_step[0:1] + split_step[split_step_verb_index:]) + '?'
                            else:
                                if split_step[1] == 'where':
                                    return ' '.join(['Is'] + split_step) + '?'
                                return ' '.join(['Is it'] + split_step) + '?'
                elif wdt == 'in' and split_step[2] == 'which':
                    return 'Is' + step.replace(' in which ', ' where ') + '?'
                elif wdt == verb:
                    if doc[first_verb_index].pos_ == 'AUX':
                        return ' '.join([verb.capitalize()] + split_step[0:1] + split_step[2:]) + '?'
                    else:
                        return 'Was' + step + '?'
                elif wdt in ['whose', 'which', 'with']:
                    if wdt == 'which':
                        step = step.replace(' which ', ' that ')
                    # TODO: could do this better. example:
                    # step: #1 that turkish people originate from
                    # current generation: Is it #1 that turkish people originate from?
                    # better generation: Do turkish people originate from #1?
                    return 'Is it' + step + '?'
                else:
                    return 'Is' + step + '?'
                    
        # no verb detected -- either a Spacy pos-tagging error or QDMR lacks a verb.
        # former example: #1 that exports to belgium (exports tagged as NNS)
        # latter example: #1 with religious organization leader treorchy noddfa
        return 'Is it' + step + '?'

    @staticmethod
    def pound_to_at(step):
        """
        Replaces # with @@ in decomposition step strings.
        :param step: String corresponding to a single step.
        :return: String with new format.
        """
        references = re.findall(r'#\d+ ', step)
        for reference in references:
            replacement = '@@' + reference.replace('#', '') + '@@ '
            step.replace(reference, replacement)
        return step

    @staticmethod
    def steps_to_allen(decomposition_list):
        """
        Formats a decomposition step list in the form of the AllenAI parser.
        :param decomposition: List of strings corresponding to decomposition.
        :return: String with new format.
        """
        s = 'return ' + ' ;return '.join(decomposition_list)
        return s.replace(' @@', ' #').replace('@@', '')

    @staticmethod
    def postprocess_question(question, original_step_tokens):
        question = ' ' + question.replace('?', ' ?')
        prevent_check_index = 0
        for ix, token in enumerate(original_step_tokens):
            if ix < prevent_check_index:
                continue
            longest_substring_match = [token]
            for jx in range(ix + 1, len(original_step_tokens) - 1):
                if ' '.join(longest_substring_match + [original_step_tokens[jx]]).lower() in question:
                    prevent_check_index = jx + 1
                    longest_substring_match.append(original_step_tokens[jx])
                else:
                    break
            longest_substring_match = ' ' + ' '.join(longest_substring_match) + ' '
            if longest_substring_match.lower() != longest_substring_match:
                question = question.replace(longest_substring_match.lower(), longest_substring_match)
        while '  ' in question:
            question = question.replace('  ', ' ')
        question = question.replace(' # ', ' #')
        question = question.replace(" 's ", "'s ")
        question = question.replace(' , ', ', ')
        question = question.replace(' ?', '?')
        question = question.replace(' the the ', ' the ')

        # Ensure the last character of the question string is a question mark
        if question[-1] != '?':
            question += '?'

        return question.strip()
