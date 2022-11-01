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
Question Analyzer

March 28, 2021
Authors: C3 Lab
"""

from time import time
from typing import Dict, List
import json
import requests
from anytree import LevelOrderIter

from mim_core.structs.MQR import MQR, HierarchicalMQR
from mim_core.structs.Step import Step
from mim_core.structs.Utterance import Utterance
from mim_core.components.QuestionAnalysis.MQRCompiler import MQRCompiler
from mim_core.components.QuestionAnalysis.QDMRValidator import QDMRValidator
from mim_core.components.QuestionAnalysis.SubOperatorParser import SubOperatorParser
from mim_core.components.QuestionAnalysis.ComplexQuestionClassifier import ComplexQuestionClassifier
from mim_core.exceptions import MultipleRelationshipsFoundWarning, NoRelationshipFoundWarning, \
                                DecompositionPredictionError
from mim_core.utils.result_utils import get_parent_step_refs, renumber_steps, is_arg_reference
import mim_core.utils.component_loading as cl_utils
import spacy_canonicalizer.DatabaseConnection


class QuestionAnalyzer(object):
    """
    A class for building the MQR by decomposing the question, extracting and linking entities, and managing
    any other analysis that would be useful for answering the question.
    """

    def __init__(self, **kwargs):
        self.do_low_level_decomposition = kwargs.get('do_low_level_decomposition', False)
        self.high_level_question_decomposer = cl_utils.load_high_level_question_decomposer(kwargs.get('high_level_question_decomposer', None))
        self.low_level_question_decomposer = None
        self.ontology = cl_utils.load_ontology(kwargs.get('ontology', None))
        self.low_level_mqr_compiler = MQRCompiler(high_level=False)
        self.high_level_compiler = MQRCompiler(high_level=True)
        self.qdmr_validator = QDMRValidator()
        self.suboperator_parser = SubOperatorParser()
        self.question_generator = cl_utils.load_question_generator(kwargs.get('question_generator', None))
        self.answer_type_classifier = cl_utils.load_answer_type_classifier(kwargs.get('answer_type_classifier', None))
        self.entity_extraction_method = kwargs.get('entity_extraction_method', 'ontology')
        self.relation_extraction_method = kwargs.get('relation_extraction_method', 'ontology')
        self.complex_question_classifier = ComplexQuestionClassifier(use_gpu=kwargs.get('use_gpu_for_complex_question_classification', False))
        self.ambiguity_resolution_method = kwargs.get('ambiguity_resolution_method', None)

    def analyze_question(self,
                         utterance: Utterance) -> HierarchicalMQR:
        """
        Generates a plan for answering the question provided in the Utterance.
        :param utterance: Contains the question to be answered.
        :return: MQR containing the plan for answering the question.
        """
        # Initialize timing fields
        total_start_time = time()
        timing = {
            "question_analyzer": {
                "complex_question_classification": 0
            }
        }

        # Check if the incoming utterance is a simple or a complex question
        start_time = time()
        step_type = self.complex_question_classifier.classify(utterance.text)
        if timing:
            timing["question_analyzer"]["complex_question_classification"] += time() - start_time

        # Initialize the root of the hierarchical MQR
        root = Step(qdmr=utterance.text, reference_id=1, q_id=utterance.q_id, step_type=step_type)
        root = self.build_step(root, [], timing)
        queue = [root]
        while len(queue) > 0:
            # Get the next question node to analyze
            q = queue.pop(0)

            if q.step_type.lower() == 'complex' and self.high_level_question_decomposer:
                try:
                    high_nodes = self.complex_question_analysis(q, utterance.q_id, timing)
                    for n in high_nodes:
                        n.parent = q
                        queue.append(n)
                except Exception as e:
                    q.errors.append(e)
                    continue
            elif q.step_type.lower() == 'simple' and q.is_high_level_retrieval_step() and self.low_level_question_decomposer:
                try:
                    low_nodes = self.simple_question_analysis(q, utterance.q_id, timing)
                    for n in low_nodes:
                        n.parent = q
                        queue.append(n)
                except Exception as e:
                    q.errors.append(e)
                    continue
            else:
                continue
        
        self.update_entities(root)

        timing["question_analyzer"]["total"] = time() - total_start_time

        return HierarchicalMQR(root,
                               q_id=utterance.q_id,
                               timing=timing)

    def update_entities(self,
                        root: Step) -> None:
        for step in LevelOrderIter(root):
            if step.parent:
                step.entities = dict(filter(lambda ent_item: ent_item[0] in step.qdmr, step.parent.entities.items())) or step.entities
            if step.operator_type=='select' and step.step_type=='low':
                final_entities = {}
                for span, ents in step.entities.items():
                    if len(ents) == 1:
                        final_entities[span] = ents[0]
                    elif self.ambiguity_resolution_method == "command_line":
                        choice_list = "\n" + "\n".join(f"({i+1}) {ent['label']} - {ent['description']}" for i, ent in enumerate(ents))
                        choice = int(input(f'Which "{span}" did you mean?{choice_list}' + '\n')) - 1
                        final_entities[span] = ents[int(choice)]
                    else:
                        final_entities[span] = min(ents, key=lambda ent: ent['id'])
                step.entities = final_entities
            if step.parent:
                for span in step.parent.entities:
                    if span in step.entities:
                        step.parent.entities[span] = step.entities[span]
        for step in LevelOrderIter(root):
            for span, ents in step.entities.items():
                if type(ents) == list:
                    step.entities[span] = min(ents, key=lambda ent: ent['id'])

    def falcon_extraction(self,
                          question: str) -> Dict:
        """
        Uses the Falcon 2.0 API to extract entities and relationships from a given question string.
        :param question: The question string from which to extract entities and/or relationships.
        """
        results = json.loads(requests.request(method='post',
                                              url='https://labs.tib.eu/falcon/falcon2/api?mode=long',
                                              data=f'{{"text":"{question}"}}'.encode('utf-8'),
                                              headers={"content-type": "application/json", "Accept-Charset": "UTF-8"}).text)
        with self.ontology.db.wikidata.Session() as session:
            return {"entities": {entity[0]: [{'id': entity[1].strip('<>').rpartition('/')[2].lstrip('Q'), 'label': self.ontology.db.wikidata.get_entities_by_id([entity[1].strip('<>').rpartition('/')[2].lstrip('Q')], session)[0].name}] for entity in results["entities_wikidata"]},
                    "relations": [{'id': relation[1].strip('<>').rpartition('/')[2].lstrip('P'), 'label': self.ontology.db.wikidata.get_properties_by_id([relation[1].strip('<>').rpartition('/')[2].lstrip('P')], session)[0].property_name, 'inverted': False} for relation in results["relations_wikidata"]]}

    def opentapioca_extraction(self,
                               question: str) -> Dict:
        """
        Uses the Open Tapioca API to extract entities from a given question string.
        :param question: The question string from which to extract entities.
        """
        results = json.loads(requests.request(method='post',
                                              url=f'https://opentapioca.org/api/annotate?query={question.encode("utf-8")}',
                                              # data=f'{{"query":"{question}"}}'.encode('utf-8'),
                                              headers={"content-type": "application/json", "Accept-Charset": "UTF-8"}).text)
        return {"entities": {results["text"][annotation["start"]:annotation["end"]]: [{'id': annotation["tags"][0]["id"].lstrip('Q'), 'label': annotation["tags"][0]["label"][0]}] for annotation in results["annotations"]}}

    def complex_question_analysis(self,
                                  step: Step,
                                  q_id: str,
                                  timing: Dict = None) -> List[Step]:
        """
        Carries out the operations needed to breakdown a complex question into a series of simpler question steps.
        :param step: The Step containing the complex question.
        :param timing: A dictionary used to track cumulative operation time.
        :return: A list of Steps representing a plan for answering this question.
        """

        # Do some initial question analysis of the complex question
        if not step.expected_answer_type and not step.question_text:
            step.question_text = step.qdmr
            step.expected_answer_type = self.answer_type_classifier.classify(step.question_text)

        # Build the initial plan
        initial_plan = self.build_plan(step.qdmr, q_id, high_level=True, timing=timing)

        # Compile the plan and return the final plan
        final_plan = self.compile_plan(initial_plan, self.high_level_compiler, timing)

        return final_plan.steps

    def simple_question_analysis(self,
                                 step: Step,
                                 q_id: str,
                                 timing: Dict = None) -> List[Step]:
        """
        Carries out the operations needed to breakdown a simple question into a series of low-level steps.
        :param step: The Step containing the simple question.
        :param timing: A dictionary used to track cumulative operation time.
        :return: A list of Steps representing a plan for answering this question.
        """

        # Build the initial plan
        initial_plan = self.build_plan(step.qdmr, q_id, high_level=False, timing=timing)

        # Compile the plan and return the final plan
        final_plan = self.compile_plan(initial_plan, self.low_level_mqr_compiler, timing)

        return final_plan.steps

    def low_level_decomposition(self,
                                question: str,
                                q_id: str) -> List[List[Step]]:
        """
        A wrapper function for performing low-level decomposition that is more robust than plugging
        the question into the decomposition model.
        :param question: The question string to decompose.
        :param q_id: The id of the question, if it came from a dataset.
        :return: A list of candidate plans/decompositions.
        """

        # TODO: Add more robust simple question decomposition

        # Replace any references to prior steps with a parent_X reference (e.g. @@3@@ --> parent_3)
        ref_num = is_arg_reference(question)
        if ref_num:
            final_question = question.replace('@@' + ref_num + '@@', 'parent_' + ref_num)
        else:
            final_question = question

        return self.low_level_question_decomposer.decompose_question(Utterance(final_question, q_id), num_generations=1)

    def build_plan(self,
                   question: str,
                   q_id: str,
                   high_level: bool,
                   timing: Dict = None) -> MQR:
        """
        Builds the initial plan from a given utterance.
        :param question: The question for which to build a plan for answering.
        :param q_id: The id of the question, if it came from a dataset.
        :param high_level: A flag for specifying the type of question decomposition to use.
        :param timing: A dictionary used to track cumulative operation time.
        :return: MQR containing a plan for answering the question.
        """

        # Classify the question type as being boolean or non-boolean
        start_time = time()
        question_type = self.classify_question_boolean(question)
        timing["question_analyzer"]["question_classification"] = time() - start_time

        # Perform the question decomposition
        start_time = time()
        if high_level:
            candidate_plans = self.high_level_question_decomposer.decompose_question(Utterance(question, q_id), num_generations=1)
        else:
            candidate_plans = self.low_level_decomposition(question, q_id)
        timing["question_analyzer"]["question_decomposition"] = time() - start_time

        # Throw an error if no candidate plans could be generated/parsed
        if not candidate_plans:
            raise DecompositionPredictionError()

        # Validate the QDMR and pick the most structurally valid plan among the candidates
        start_time = time()
        valid_steps = self.qdmr_validator.validate(question, candidate_plans)
        timing["question_analyzer"]["qdmr_validation"] = time() - start_time

        # Analyze and finish build each step
        finished_steps = [self.build_step(s, valid_steps, timing) for s in valid_steps]

        # Close entity linker database connection (because the library seems to have neglected to do so itself).
        # Ideally in the future we will instead want to fork that library and add our own functionalities.
        spacy_canonicalizer.DatabaseConnection.wikidata_instance.conn.close()
        spacy_canonicalizer.DatabaseConnection.wikidata_instance = None

        # Return the final plan
        return MQR(question, finished_steps, q_id, timing=timing, question_type=question_type)

    def build_step(self,
                   step: Step,
                   all_steps: List[Step],
                   timing: Dict = None) -> Step:
        """
        Finishes building out the given step.
        :param step: A step with at least the qdmr field initialized to analyze and build.
        :param all_steps: A list of the Steps in the plan so far.
        :param timing: A dictionary used to track cumulative operation time.
        :return: A completely built and analyzed step.
        """
        # Ensure the qdmr field is initialized (not "" or None) and can be analyzed properly
        if not step.qdmr:
            # TODO: Implement a warning for the case where no qdmr field is present
            return step

        # Initialize timing fields
        if timing:
            if "question_analyzer" not in timing:
                timing["question_analyzer"] = {}
            if "complex_question_classification" not in timing["question_analyzer"]:
                timing["question_analyzer"]["complex_question_classification"] = 0
            if "falcon_extraction" not in timing["question_analyzer"]:
                timing["question_analyzer"]["falcon_extraction"] = 0
            if "qdmr_operator_parsing" not in timing["question_analyzer"]:
                timing["question_analyzer"]["qdmr_operator_parsing"] = 0
            if "qdmr_graph_linking" not in timing["question_analyzer"]:
                timing["question_analyzer"]["qdmr_graph_linking"] = 0
            if "canonical_entity_extraction" not in timing["question_analyzer"]:
                timing["question_analyzer"]["canonical_entity_extraction"] = 0
            if "entity_class_extraction" not in timing["question_analyzer"]:
                timing["question_analyzer"]["entity_class_extraction"] = 0
            if "suboperator_parsing" not in timing["question_analyzer"]:
                timing["question_analyzer"]["suboperator_parsing"] = 0
            if "question_generation" not in timing["question_analyzer"]:
                timing["question_analyzer"]["question_generation"] = 0
            if "canonical_relation_extraction" not in timing["question_analyzer"]:
                timing["question_analyzer"]["canonical_relation_extraction"] = 0
            if "answer_type_classification" not in timing["question_analyzer"]:
                timing["question_analyzer"]["answer_type_classification"] = 0

        start_time = time()
        if 'falcon' in [self.entity_extraction_method, self.relation_extraction_method]:
            falcon_results = self.falcon_extraction(step.qdmr)
        if timing:
            timing["question_analyzer"]["falcon_extraction"] += time() - start_time
        if self.entity_extraction_method == 'opentapioca':
            opentapioca_results = self.opentapioca_extraction(step.qdmr)

        # Initialize/update the operators args and operator type of the step
        start_time = time()
        step.update_operator_args_and_type()
        if timing:
            timing["question_analyzer"]["qdmr_operator_parsing"] += time() - start_time

        # Link the graph nodes based on their references
        start_time = time()
        self.link_to_graph(step, all_steps)
        if timing:
            timing["question_analyzer"]["qdmr_graph_linking"] += time() - start_time

        # Canonicalize the entities in the decomposition steps
        start_time = time()
        
        if self.entity_extraction_method == 'falcon' and falcon_results['entities']:
            step.entities = falcon_results['entities']
        elif self.entity_extraction_method == 'opentapioca' and opentapioca_results['entities']:
            step.entities = opentapioca_results['entities']
        else:
            step.entities = self.ontology.extract_canonical_entities(step.qdmr, single_term = step.operator_type=='select' and step.step_type=='low')
        # step.entities = falcon_results['entities'] if self.entity_extraction_method == 'falcon' and falcon_results['entities'] \
        #         else self.ontology.extract_canonical_entities(step.qdmr, single_term = step.operator_type=='select' and step.step_type=='low')
        if timing:
            timing["question_analyzer"]["canonical_entity_extraction"] += time() - start_time

        # Use the cleaned operator arguments to determine the class of entities being asked about, if any
        start_time = time()
        step.entity_class = self.ontology.extract_canonical_entity_class(step.operator_args[0]) if step.operator_type.lower() == 'select' else None
        if timing:
            timing["question_analyzer"]["entity_class_extraction"] += time() - start_time

        # Parse out the suboperation type
        start_time = time()
        step.operator_subtype = self.suboperator_parser.parse_subtype(step)
        if timing:
            timing["question_analyzer"]["suboperator_parsing"] += time() - start_time

        # Generate the question this step is trying to answer
        start_time = time()
        # parent_steps = [all_steps[int(i)-1] for i in get_all_parent_refs(step, all_steps)]
        # step.question_text = self.question_generator.generate_question(parent_steps + [step])[0]
        # TODO: type checking for BARTGenerator -- this only holds for RuleBasedQuestionGenerator
        try:
            step.question_text = self.question_generator.generate_question([step])[0]
        except Exception as e:
            step.question_text = ""
            step.errors.append(e)
        if step.question_text is None:
            # unsure how to handle if step is outside of rule-based coverage -- use empty string or just output step text?
            step.question_text = ""  # or step.question_text = step.qdmr
        if timing:
            timing["question_analyzer"]["question_generation"] += time() - start_time

        # Determine the expected answer type for this step of the plan
        start_time = time()
        step.expected_answer_type = self.answer_type_classifier.classify(step.question_text if step.question_text and self.answer_type_classifier.input_type == 'question' else step.qdmr)
        if timing:
            timing["question_analyzer"]["answer_type_classification"] += time() - start_time

        # Canoncalize the relationship in the steps
        start_time = time()
        relationships = falcon_results['relations'] + self.ontology.extract_canonical_relationship(step.qdmr, step.operator_type, step.entities) if self.relation_extraction_method == 'falcon' \
                        else self.ontology.extract_canonical_relationship(step.qdmr, step.operator_type, step.entities)

        # Don't care about relationships if this is the select step
        if step.operator_type != 'select':
            if len(relationships) > 1:
                step.errors.append(MultipleRelationshipsFoundWarning(relationships))
            elif len(relationships) == 0:
                step.errors.append(NoRelationshipFoundWarning())

        step.relationship = relationships[0] if len(relationships) else None
        if timing:
            timing["question_analyzer"]["canonical_relation_extraction"] += time() - start_time

        # Set this step's flag that it has been built
        step.is_built = True

        return step

    def link_to_graph(self,
                      step: Step,
                      all_steps: List[Step]) -> None:
        """
        Links the parents and children of the given step to the other steps of the QDMR graph.
        :param step: The step to link to the graph.
        :param all_steps: The existing graph of QDMR steps.
        :return: None
        """

        # Connect the child to the parents
        parent_refs = set(get_parent_step_refs(step))
        if step.reference_id in parent_refs:    # Make sure the step is not a parent/child of itself
            parent_refs.remove(step.reference_id)
        step.parent_steps = [all_steps[int(i)-1] for i in parent_refs]

        # Connect the parents to the child
        for p in step.parent_steps:
            if step not in p.child_steps:
                p.child_steps.append(step)

        return None

    def classify_question_boolean(self,
                                  question_text: str) -> List[str]:
        """
        A very simple answer type classifier to determine if the given question expects a boolean or not.
        :param question_text: The question for which to determine if the expected answer type is boolean.
        :return: The expected answer type of the given question.
        """

        # TODO: Get rid of this once a more robust answer classifier is built
        if question_text.lower().split()[0] in ['are', 'were', 'is']:
            return ['boolean']
        else:
            return ['entity']

    def compile_plan(self,
                     mqr: MQR,
                     compiler: MQRCompiler,
                     timing: Dict = None) -> MQR:
        """
        Apples rules to the given plan to improve the likelihood it results in a correct answer.
        :param mqr: The MQR/plan to compile and improve.
        :param compiler: The compiler to use for compilling the given MQR/plan.
        :param timing: A dictionary used to track cumulative operation time.
        :return: The improved MQR/plan.
        """

        # Init timing fields
        if timing and "plan_compiling" not in timing["question_analyzer"]:
            timing["question_analyzer"]["plan_compiling"] = 0

        start_time = time()

        changes_made = True
        while changes_made:
            for rule in compiler.rules:
                # Apply the current rule to the MQR/plan
                changes_made = rule.apply(mqr)

                # Finish building any steps that were generated by the compilation operations
                mqr.steps = [self.build_step(s, mqr.steps) if not s.is_built else s for s in mqr.steps]

            # Reorder operation based on their reference numbers
            renumber_steps(mqr)

        if timing:
            timing["question_analyzer"]["plan_compiling"] += time() - start_time

        return mqr
