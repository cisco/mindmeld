# -*- coding: utf-8 -*-
import codecs
from collections import defaultdict
import logging
import os

from sklearn.externals import joblib

logger = logging.getLogger(__name__)


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

      sys_types: The set of nested numeric types for this entity
     """

    def __init__(self, name, exclude_ngrams=False):
        """
        Args:
            domain (str): The domain that this gazetteer is used
            entity_type (str): The name of the entity that this gazetteer is used
            exclude_ngrams (bool): The boolean flat whether to exclude ngrams
        """
        self.name = name
        self.exclude_ngrams = exclude_ngrams
        self.max_ngram = 1

        self.entity_count = 0
        self.pop_dict = defaultdict(int)
        self.index = defaultdict(set)
        self.entities = []
        self.sys_types = set()

    def to_dict(self):
        """
        Returns: dict
        """
        return {
            'name': self.name,
            'total_entities': self.entity_count,
            'pop_dict': self.pop_dict,
            'index': self.index,
            'entities': self.entities,
            'sys_types': self.sys_types
        }

    def from_dict(self, serialized_gaz):
        """De-serializes gaz object from a dictionary using deep copy ops

        Args:
            serialized_gaz (dict): The serialized gaz object
        """
        for key, value in serialized_gaz.items():
            # We only shallow copy lists and dicts here since we do not have nested
            # data structures in this container, only 1-levels dictionaries and lists,
            # so the references only need to be copies. For all other types, like strings,
            # they can just be passed by value.
            setattr(self, key, value.copy() if isinstance(value, (list, dict)) else value)

    def dump(self, gaz_path):
        """Persists the gazetteer to disk.

        Args:
            gaz_path (str): The location on disk where the gazetteer should be stored

        """
        # make directory if necessary
        folder = os.path.dirname(gaz_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)

        joblib.dump(self.to_dict(), gaz_path)

    def load(self, gaz_path):
        """Loads the gazetteer from disk

        Args:
            gaz_path (str): The location on disk where the gazetteer is stored

        """
        gaz_data = joblib.load(gaz_path)
        self.name = gaz_data['name']
        self.entity_count = gaz_data['total_entities']
        self.pop_dict = gaz_data['pop_dict']
        self.index = gaz_data['index']
        self.entities = gaz_data['entities']
        self.sys_types = gaz_data['sys_types']

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
        if self.pop_dict[entity] == 0:
            self.entities.append(entity)
            if not self.exclude_ngrams:
                for ngram in iterate_ngrams(entity.split(), max_length=self.max_ngram):
                    self.index[ngram].add(self.entity_count)
            self.entity_count += 1

        if keep_max:
            old_value = self.pop_dict[entity]
            self.pop_dict[entity] = max(self.pop_dict[entity], popularity)
            if self.pop_dict[entity] != old_value:
                logger.debug('Updating gazetteer value of entity {} from {} to {}'.format(
                    entity, old_value, self.pop_dict[entity]))
        else:
            self.pop_dict[entity] = popularity

    def update_with_entity_data_file(self, filename, popularity_cutoff, normalizer):
        """
        Updates this gazetteer with data from an entity data file.

        Args:
            filename (str): The filename of the entity data file.
            popularity_cutoff (float): A threshold at which entities with
                popularity below this value are ignored.
            normalizer (function): A function that normalizes text.
        """
        logger.info("Loading entity data from '{}'".format(filename))
        line_count = 0
        entities_added = 0
        num_cols = None

        if not os.path.isfile(filename):
            logger.warn("Entity data file was not found at {!r}".format(filename))
        else:
            with codecs.open(filename, encoding='utf8') as data_file:
                for i, row in enumerate(data_file.readlines()):
                    if not row:
                        continue
                    split_row = row.strip('\n').split('\t')
                    if num_cols is None:
                        num_cols = len(split_row)

                    if len(split_row) != num_cols:
                        msg = "Row {} of .tsv file '{}' malformed, expected {} columns"
                        raise ValueError(msg.format(i+1, filename, num_cols))

                    if num_cols == 2:
                        pop, entity = split_row
                    else:
                        pop = 1.0
                        entity = split_row[0]

                    pop = 0 if pop == 'null' else float(pop)
                    line_count += 1
                    entity = normalizer(entity)
                    if pop > popularity_cutoff:
                        self._update_entity(entity, float(pop))
                        entities_added += 1

            logger.info('{}/{} entities in entity data file exceeded popularity '
                        "cutoff and were added to the gazetteer".format(entities_added, line_count))

    def update_with_sys_types(self, numeric_types):
        self.sys_types.update(numeric_types)

    def update_with_entity_map(self, mapping, normalizer, update_if_missing_canonical=True):
        logger.info('Loading synonyms from entity mapping')
        line_count = 0
        synonyms_added = 0
        missing_canonicals = 0
        min_popularity = 0
        if len(self.pop_dict) > 0:
            min_popularity = min(self.pop_dict.values())
        for item in mapping:
            canonical = normalizer(item['cname'])

            for syn in item['whitelist']:
                line_count += 1
                synonym = normalizer(syn)

                if update_if_missing_canonical or canonical in self.pop_dict:
                    self._update_entity(synonym, self.pop_dict.get(canonical, min_popularity))
                    synonyms_added += 1
                if canonical not in self.pop_dict:
                    missing_canonicals += 1
                    logger.debug("Synonym '{}' for entity '{}' not in "
                                 'gazetteer'.format(synonym, canonical))
        logger.info('Added {}/{} synonyms from file into '
                    'gazetteer'.format(synonyms_added, line_count))
        if update_if_missing_canonical and missing_canonicals:
            logger.info('Loaded {} synonyms where the canonical name is not '
                        'in the gazetteer'.format(missing_canonicals))


def iterate_ngrams(tokens, min_length=1, max_length=1):
    """Iterates over all n-grams in a list of tokens.

    Args:
        tokens (list of str): A list of word tokens.
        min_length (int): The minimum length of n-gram to yield.
        max_length (int): The maximum length of n-gram to yield.

    Yields:
        (str) An n-gram from the input tokens list.
    """
    max_length = min(len(tokens), max_length)
    unrolled_tokens = [tokens[i:] for i in range(max_length)]
    for length in range(min_length, max_length+1):
        for ngram in zip(*unrolled_tokens[:length]):
            yield ' '.join(ngram)
