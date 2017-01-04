from __future__ import unicode_literals

from collections import defaultdict
import util
import logging
from sklearn.externals import joblib


class Gazetteer:
    """
    This class holds the following  fields, which are extracted and exported to file.

      entity_count: Total entities in the file

      pop_dict: This is a dictionary with the entity name as a key and the popularity score as the
        value. If there are more than one entity with the same name, the popularity is the maximum
        value across all duplicate entities.

      index: This contains the inverted index, which maps terms and n-grams to the set of documents
        which contain them

      entities: This is simply a list of all entities

      numtypes: The set of nested numeric types for this facet
     """
    def __init__(self, domain, facet, exclude_ngrams=False):
        """
        Args:
            domain (str): The domain that this gazetteer is used
            facet (str): The name of the facet that this gazetteer is used
            exclude_ngrams (bool): The boolean flat whether to exclude ngrams
        """
        self.domain = domain
        self.facet = facet
        self.entity_count = 0
        self.exclude_ngrams = exclude_ngrams

        self.pop_dic = defaultdict(int)
        self.index = defaultdict(set)
        self.entities = []
        self.numtypes = set()
        self.max_ngram = 1

    def to_dict(self):
        """
        Returns:
            dict
        """
        return {
            "domain_name": self.domain,
            "facet_name": self.facet,
            "total_entities": self.entity_count,
            "pop_dict": self.pop_dic,
            "index": self.index,
            "entities": self.entities,
            "numtypes": self.numtypes
        }

    def to_file(self, filename):
        """
        Args:
            filename (str)
        """
        try:
            joblib.dump(self.to_dict(), filename)
        except IOError as ex:
            logging.info('Exception writing to {0}: {1}'.format(filename, ex))

    def _update_entity(self, entity, popularity, keep_max=True):
        """
        Updates all gazetteer data with an entity and its popularity.

        Args:
            entity (str): A normalized entity name.
            popularity (float): The entity's popularity value.
            keep_max (bool): If True, if the entity is already in the pop_dict, then set the
                popularity to the max of popularity and the value in the pop_dict.
                Otherwise, overwrite it.
        """
        # Only update the relevant data structures when the entity isn't
        # already in the gazetteer. Update the popularity either way.
        if self.pop_dic[entity] == 0:
            self.entities.append(entity)
            if not self.exclude_ngrams:
                for ngram in util.iter_ngrams(entity.split(), max_length=self.max_ngram):
                    self.index[ngram].add(self.entity_count)
            self.entity_count += 1

        if keep_max:
            self.pop_dic[entity] = max(self.pop_dic[entity], popularity)
        else:
            self.pop_dic[entity] = popularity

    def update_with_entity_data_file(self, filename, popularity_cutoff, normalizer):
        """
        Updates this gazetteer with data from an entity data file.

        Args:
            filename (str): The filename of the entity data file.
            popularity_cutoff (float): A threshold at which entities with
                popularity below this value are ignored.
            normalizer (function): A function that normalizes text.
        """
        logging.info("Loading entity data from '{}'".format(filename))
        line_count = 0
        entities_added = 0
        for pop, entity in util.read_tsv_lines(filename, 2):
            pop = 0 if pop == 'null' else float(pop)
            line_count += 1
            entity = normalizer(entity)
            self.numtypes.update(util.get_nested_numeric_types(entity))
            if pop > popularity_cutoff:
                self._update_entity(entity, float(pop))
                entities_added += 1

        logging.info('{}/{} entities in entity data file exceeded popularity '
                     "cutoff and were added to the gazetteer".format(
                         entities_added, line_count))

    def update_with_entity_map(self, entity_map, numeric_types, normalizer):
        """
        Updates this gazetteer with data from entity map.

        Args:
            entity_map (dict)
            numeric_types (dict)
            normalizer (function)
        """
        for mapping_key, mapping_value in entity_map.items():
            if mapping_key == '*':
                continue
            mapping_key = normalizer(mapping_key)
            self._update_entity(mapping_key, 1)
            self.numtypes.update(util.get_nested_numeric_types(mapping_key))
        self.numtypes.update(numeric_types)
        logging.info("Nested numeric types for '{}' facet: {}".format(
            self.facet, map(str, self.numtypes)))

    def update_with_synonyms_file(self, filename, normalizer, update_if_missing_canonical=True):
        """
        Updates this gazetteer with data from a synonyms file.

        Args:
            filename (str): The filename of the entity data file.
            normalizer (function): A function that normalizes text.
            update_if_missing_canonical (bool): Bool indicator that added synonyms not in dict
        """
        logging.info('Loading synonyms from {}'.format(filename))
        line_count = 0
        synonyms_added = 0
        missing_canonicals = 0
        min_popularity = 0
        if len(self.pop_dic) > 0:
            min_popularity = min(list(self.pop_dic.values()))
        for canonical, synonym in util.read_tsv_lines(filename, 2):
            line_count += 1
            canonical = normalizer(canonical)
            synonym = normalizer(synonym)

            if update_if_missing_canonical or canonical in self.pop_dic:
                self._update_entity(entity=synonym,
                                    popularity=self.pop_dic.get(canonical, min_popularity))
                synonyms_added += 1
            if canonical not in self.pop_dic:
                missing_canonicals += 1
                logging.debug(u"Synonym '{}' for entity '{}' not in gazetteer"
                              .format(synonym, canonical))
        logging.info('Added {}/{} synonyms from file into gazetteer'
                     .format(synonyms_added, line_count))
        if update_if_missing_canonical and missing_canonicals:
            logging.info('Loaded {} synonyms where the canonical name is not '
                         'in the gazetteer'.format(missing_canonicals))
