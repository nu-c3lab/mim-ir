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
Wikidata Ontology

March 28, 2021
Authors: C3 Lab
"""

import re
import spacy
from mim_core.components.Ontology.Ontology import Ontology
import mim_core.components.Ontology.RelationshipPatterns as relationship_patterns
from mim_core.components.models import get_model
from spacy_canonicalizer.DatabaseConnection import get_wikidata_instance
from jumbodb import Jumbo
from functools import reduce
from json import loads
from typing import List

class WikidataOntology(Ontology):
    # TODO: For wikidata ontology (and other modules that get reused in different components) probably should have some kind of mechanism
    #       (presumably in the componint loading module) that only loads one instance, and if an instance already exists, reuses that instance
    def __init__(self, **kwargs):
        super().__init__()
        self.spacy_model = get_model(kwargs.get('spacy_model', 'en_core_web_sm'))
        # self.spacy_model = spacy.load(kwargs.get('spacy_model', 'en_core_web_sm'))
        # self.spacy_model.add_pipe("entityLinker", last=True)
        # self.spacy_small = spacy.load('en_core_web_sm')
        self.spacy_small = get_model('en_core_web_sm') # Always using small model here for consistency in parse patterns
        self.spacy_medium = get_model('en_core_web_md')
        self.relationship_canonicalizations = relationship_patterns.canonicalizations

        # We sort alias keys by length descending because we want to perform a greedy (longest string) search
        self.sorted_aliases = sorted(self.relationship_canonicalizations.keys(), key=len, reverse=True)
        self.filter_patterns = relationship_patterns.filter_patterns
        self.project_patterns = relationship_patterns.project_patterns
        self.db = Jumbo(True)

    def replace_in_list(self, lst, replace_string, start, end):
        for i in range(start, end):
            lst.pop(start)
        lst.insert(start, replace_string)
        return (end - start) - 1

    def group_entities(self, lst, entities):
        tokens_removed = 0
        for entity in entities:
            if re.fullmatch(r'@@\d+@@', entity.text):
                tokens_removed += self.replace_in_list(lst, 'ARGUMENT', entity.start - tokens_removed, entity.end - tokens_removed)
            else:
                tokens_removed += self.replace_in_list(lst, 'ENTITY', entity.start - tokens_removed, entity.end - tokens_removed)
        return lst

    def extract_canonical_entities(self, text: str, single_term: bool=False, expected_types: list=None) -> dict:
        """
        Extracts entities from the given text and attempts to find their canonical form
        :param text: String from which to extract entities
        :return: A dictionary mapping entities as they exist in the original text to the canonical form.
        """
        canonicalizer_db = get_wikidata_instance()
        canonicalizer_db.cache["chain"].clear()
        parsed = self.spacy_model(text, component_cfg={"entityLinker": {"single_term": single_term, "expected_types": expected_types, "nlp": self.spacy_medium}})
        #return {entity.span.text:{"id": entity.identifier, "label": entity.label, "description": entity.description, "span": entity.span} for entity in parsed._.linkedEntities.entities}

        # Group entities into a dictionary by span text
        return reduce(lambda a, entity:
                          dict([(k,v) for k,v in a.items() if k != entity.span.text] + [(entity.span.text, a[entity.span.text] + [{"id": entity.identifier, "label": entity.label, "description": entity.description}])]) if entity.span.text in a
                          else dict([(k,v) for k,v in a.items()] + [(entity.span.text, [{"id": entity.identifier, "label": entity.label, "description": entity.description}])]),
                      parsed._.linkedEntities.entities,
                      {})

    def extract_canonical_relationship(self, text: str, operator_type: str, entities: dict, filter_entities: List[int] = None) -> str:
        """
        Extracts relationships from the given text and attempts to find their canonical form.
        :param text: String from which to extract entities
        :return: A string representing the canonical relationship in the input text.
        """

        canonical_relationships = []
        if operator_type in ['filter', 'project']:
            # Plan A: Use POS pattern matching to get relationship span
            parsed = self.spacy_small(text.replace('  ', ' ').strip())
            parts_of_speech = [token.pos_ for token in parsed]
            tokens = [token.text for token in parsed]
            pattern = ' '.join(self.group_entities(parts_of_speech, parsed.ents))
            grouped_tokens = self.group_entities(tokens, parsed.ents)

            if operator_type == 'filter':
                if re.fullmatch(r'@@\d+@@\s+in\s+\d{4}\d*', parsed.text):
                    canonical_relationships = [{"id": 585, "label": 'point in time', "inverted": False}]
                elif pattern in self.filter_patterns:
                    span = self.filter_patterns[pattern]
                    candidate_relationship = ' '.join([tok.lemma_ for tok in self.spacy_small(' '.join(grouped_tokens[span[0]:span[1]]))])
                    with self.db.wikidata.Session() as session:
                        if filter_entities:
                            canonical_relationships = [{"id": property.property_id, "label": property.property_name, "inverted": property.inverted} for property in self.db.wikidata.get_canonical_properties_from_entity(filter_entities, candidate_relationship, session, self.spacy_medium)]
                        else:
                            canonical_relationships = [{"id": property.property_id, "label": property.property_name, "inverted": property.inverted} for property in self.db.wikidata.get_canonical_properties(candidate_relationship, session, self.spacy_medium)]
            elif operator_type == 'project':
                if pattern in self.project_patterns:
                    span = self.project_patterns[pattern]
                    candidate_relationship = ' '.join([tok.lemma_ for tok in self.spacy_small(' '.join(grouped_tokens[span[0]:span[1]]))])
                    with self.db.wikidata.Session() as session:
                        if filter_entities:
                            canonical_relationships = [{"id": property.property_id, "label": property.property_name, "inverted": property.inverted} for property in self.db.wikidata.get_canonical_properties_from_entity(filter_entities, candidate_relationship, session, self.spacy_medium)]
                        else:
                            canonical_relationships = [{"id": property.property_id, "label": property.property_name, "inverted": property.inverted} for property in self.db.wikidata.get_canonical_properties(candidate_relationship, session, self.spacy_medium)]

        if len(canonical_relationships) == 0:
            # Plan B: Search up the relationship using it's corresponding entity label (if found) - particularly useful for inverse relationships.
            with self.db.wikidata.Session() as session:
                if filter_entities:
                    canonical_relationships = [{"id": property.property_id, "label": property.property_name, "inverted": property.inverted} for ents in entities.values() for entity in (ents if type(ents) == list else [ents]) for property in self.db.wikidata.get_canonical_properties_from_entity(filter_entities, entity['label'], session, self.spacy_medium)]
                else:
                    canonical_relationships = [{"id": property.property_id, "label": property.property_name, "inverted": property.inverted} for ents in entities.values() for entity in (ents if type(ents) == list else [ents]) for property in self.db.wikidata.get_canonical_properties(entity['label'], session, self.spacy_medium)]

        if len(canonical_relationships) == 0:
            # Plan C: See if the text contains any of the known relationship values (or their known aliases).
            canonical_relationships = [{"id": None, "label": self.relationship_canonicalizations[alias], "inverted": False} for alias in self.sorted_aliases if alias in text]

        return canonical_relationships

    def extract_canonical_entity_class(self, entity: str) -> str:
        """
        Determines if first operator argument in the given list is a class of entities and, if so, returns the class.
        :param operator_args: A cleaned QDMR operator arg.
        :return: Either a string denoting the canonicalized entity class or None.
        """

        # "Canonicalize" this argument by mapping it to the top search result from Wikidata
        doc = self.spacy_model(entity, component_cfg={"entityLinker": {"single_term": True}})
        ents = sorted(doc._.linkedEntities, key=lambda ent: len(ent.get_span()), reverse=True)
        # This entity is a "class" if it is plural or if there exists entities that have it as a superclass
        if any(sent.root.tag_ == 'NNS' for sent in doc.sents): # or (ents and ents[0].wikidata_instance.get_instances_of(ents[0].get_id(), count=1)):
            return {"id": ents[0].get_id(), "label": ents[0].get_label()}
        return None

    def canonicalize_entity(self, entity: str) -> str:
        """
        Canonicalizes the single, proper noun given as input.
        :param entity: A string containing a single entity that is a proper noun
        :return: A string representing the canonical version of this entity.
        """
        # Get only the proper noun from the entity text
        doc = self.spacy_model(entity, component_cfg={"entityLinker": {"single_term": True}})
        proper_noun_str = " ".join([token.text for token in doc if token.pos_ == 'PROPN'])

        # Extract and link the canonical
        canon_entities = self.extract_canonical_entities(proper_noun_str)

        if len(canon_entities) == 0:
            return entity
        else:
            return " ".join([ent['label'] for ent in canon_entities.values()])

    def get_entities_of_type(self, entity: str, max_depth=10) -> List[str]:
        """
        Retrieves all instances of the class entity (canonicalized from the given string).
        :param entity: Entity of which to retrieve categories.
        :param max_depth: Maximum depth to traverse up the entity parent hierarchy to retrieve categories.
        :return: List of entity tuples as (id, name).
        """
        canonicalizer_db = get_wikidata_instance()
        canonicalizer_db.cache["chain"].clear()
        doc = self.spacy_model(entity, component_cfg={"entityLinker": {"single_term": True}})
        parent_entity_ids = reduce(lambda a, ent: a + ent.get_subclasses(max_depth), doc._.linkedEntities, [])
        return self.db.wikidata.get_instances(parent_entity_ids)

    def get_entity_categories(self, entity: str, max_depth=10) -> List[str]:
        """
        Retrieves all parent categories of the entities canonicalized from the given string.
        :param entity: Entity of which to retrieve categories.
        :param max_depth: Maximum depth to traverse up the entity parent hierarchy to retrieve categories.
        :return: List of entity tuples as (id, name).
        """
        canonicalizer_db = get_wikidata_instance()
        canonicalizer_db.cache["chain"].clear()
        doc = self.spacy_model(entity, component_cfg={"entityLinker": {"single_term": True}})
        entity_ids = reduce(lambda a, ent: a + ent.get_categories(max_depth), doc._.linkedEntities, [])
        with self.db.wikidata.Session() as session:
            return self.db.wikidata.get_entities_by_id(entity_ids, session, source='wikidata')

    def is_expected_type(self, entity: str, expected_types: List[str]) -> bool:
        """

        :param entity:
        :param expected_types:
        :return:
        """
        entity_cats = self.get_entity_categories(entity, max_depth=10)
        entity_types = [cat.name for cat in entity_cats]
        return any(ex_type in entity_types for ex_type in expected_types)

    def get_root_paths(self, entity, include_instance_of=False):
        # TODO: Replace this code with spacy-canonicalizer chain stuff?
        if 'subclass of' in entity.relationships:
            return [path + [entity.id] for rel in entity.relationships['subclass of'] if rel.graph == 'wikidata' for path in self.get_root_paths(rel.object)]
        elif include_instance_of and 'instance of' in entity.relationships:
            return [path + [entity.id] for rel in entity.relationships['instance of'] if rel.graph == 'wikidata' for path in self.get_root_paths(rel.object)]
        else:
            return [[entity.id]]

    def longest_common_path(self, path_set_1, path_set_2):
        def longest_intersection(patha, pathb):
            for idx in range(min(len(patha), len(pathb))):
                if patha[idx] != pathb[idx]:
                    if idx > 0:
                        # second element (path length) cannot simply be the length of the path, because it needs to be the shortest path to the root
                        return (patha[idx-1], self.get_depth(patha[idx-1]))
                    else:
                        return (None, 0)
            # If the loop completed, the shorter path is a subpath of the other
            if len(patha) < len(pathb):
                return (patha[-1], self.get_depth(patha[-1]))
            else:
                return (pathb[-1], self.get_depth(pathb[-1]))

        # for path1 in path_set_1:
        #     for path2 in path_set_2:
        #         print(path1, path2, longest_intersection(path1, path2))

        return max([longest_intersection(path1, path2) for path1 in path_set_1 for path2 in path_set_2], key=lambda x: x[1])

    def least_common_ancestor(self, entity1_id, entity2_id):
        with self.db.wikidata.Session() as session:
            entity1 = self.db.wikidata.get_entities_by_id([entity1_id], session, source='wikidata')[0]
            entity2 = self.db.wikidata.get_entities_by_id([entity2_id], session, source='wikidata')[0]

            entity1_paths = self.get_root_paths(entity1, include_instance_of=True)
            # print('-------------------', entity1.name, '-------------------')
            # print('longest of', len(entity1_paths), max(entity1_paths, key=lambda x: len(x)))
            entity2_paths = self.get_root_paths(entity2, include_instance_of=True)
            # print('-------------------', entity2.name, '-------------------')
            # print('longest of', len(entity2_paths), max(entity2_paths, key=lambda x: len(x)))
            return self.longest_common_path(entity1_paths, entity2_paths)

           #if entity1 == entity2:
           #    return entity1
           #if 'subclass of' in entity1.relationships:
           #    queue1 = [rel.object for rel in entity1.relationships['subclass of'] if rel.graph=='wikidata']
           #elif 'instance of' in entity1.relationships:
           #    queue1 = [rel.object for rel in entity1.relationships['instance of'] if rel.graph=='wikidata']
           #else:
           #    queue1 = []
           #if 'subclass of' in entity2.relationships:
           #    queue2 = [rel.object for rel in entity2.relationships['subclass of'] if rel.graph=='wikidata']
           #elif 'instance of' in entity2.relationships:
           #    queue2 = [rel.object for rel in entity2.relationships['instance of'] if rel.graph=='wikidata']
           #else:
           #    queue2 = []
           #explored1 = [entity1] + queue1
           #explored2 = [entity2] + queue2

           #while len(queue1) or len(queue2):
           #    if len(queue1):
           #        anc1 = queue1.pop(0)
           #        if anc1 in explored2:
           #            return anc1
           #        new_nodes = list(filter(lambda ent: ent not in explored1, map(lambda rel: rel.object, anc1.relationships['subclass of']))) if 'subclass of' in anc1.relationships else []
           #        queue1 += new_nodes
           #        explored1 += new_nodes
           #    if len(queue2):
           #        anc2 = queue2.pop(0)
           #        if anc2 in explored1:
           #            return anc2
           #        new_nodes = list(filter(lambda ent: ent not in explored2, map(lambda rel: rel.object, anc2.relationships['subclass of']))) if 'subclass of' in anc2.relationships else []
           #        queue2 += new_nodes
           #        explored2 += new_nodes

    def get_depth(self, entity_id):
        with self.db.wikidata.Session() as session:
            entity = self.db.wikidata.get_entities_by_id([entity_id], session, source='wikidata')[0]
            if 'subclass of' in entity.relationships:
                queue = list(map(lambda rel: (rel.object, 2), entity.relationships['subclass of']))
            elif 'instance of' in entity.relationships and entity.id != 35120:
                queue = list(map(lambda rel: (rel.object, 2), entity.relationships['instance of']))
            else:
                return 1 # Starting with 1 instead of zero because we are assuming a 'virtual' root node that connects all Wikidata nodes without a parent class
            explored = [(entity, 1)] + queue
            while len(queue):
                anc = queue.pop(0)
                if 'subclass of' in anc[0].relationships:
                    new_nodes = list(filter(lambda ent: ent[0] not in map(lambda x: x[0], explored), map(lambda rel: (rel.object, anc[1] + 1), anc[0].relationships['subclass of'])))
                    queue += new_nodes
                    explored += new_nodes
                else:
                    return anc[1]
            # Should never get here
            return None
