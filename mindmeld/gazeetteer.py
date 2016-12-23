from collections import defaultdict
import util
import logging


class Gazetteer:
    def __init__(self, facet_name, exclude_ngrams=False):
        """
        This class holds the gazetteer data which is exported to a picked file

        entity_count: Total entities in the file

        edict: This is a dictionary with the entity name as a key and the
        popularity score as the value. If there are more than one entity with
        the same name, the popularity is the maximum value across all duplicate
        entities.

        index: this contains the inverted index, which maps terms and
        n-grams to the set of documents which contain them

        entities: this is simply a list of all entities

        numtypes: the set of nested numeric types for this facet

        exclude_ngrams: boolean flag indicating whether to exclude ngrams

        :type entity_count: int
        :type edict: defaultdict|None
        :type index: defaultdict|None
        :type entities: list|None
        :type numtypes: set|None
        :rtype:
        """
        self.facet_name = facet_name
        self.entity_count = 0
        self.edict = defaultdict(int)
        self.index = defaultdict(set)
        self.entities = []
        self.numtypes = set()
        self.exclude_ngrams = exclude_ngrams
        self.max_ngram = 1

    def to_dict(self):
        """
        :rtype: dict
        """
        return {
            "facet_name": self.facet_name,
            "total_entities": self.entity_count,
            "edict": self.edict,
            "index": self.index,
            "entities": self.entities,
            "numtypes": self.numtypes
        }

    def _update_entity(self, entity, popularity, keep_max=True):
        """Updates all gazetteer data with an entity and its popularity.

        Args:
            entity (str): A normalized entity name.
            popularity (float): The entity's popularity value.
            keep_max (bool): If True, if the entity is already in the edict,
                then set the popularity to the max of popularity and the value in
                the edict. Otherwise, overwrite it.
        """
        # Only update the relevant data structures when the entity isn't
        # already in the gazetteer. Update the popularity either way.
        if self.edict[entity] == 0:
            self.entities.append(entity)
            if not self.exclude_ngrams:
                for ngram in util.iter_ngrams(entity.split(), max_length=self.max_ngram):
                    self.index[ngram].add(self.entity_count)
            self.entity_count += 1
        if keep_max:
            self.edict[entity] = max(self.edict[entity], popularity)
        else:
            self.edict[entity] = popularity

    def update_with_entity_data_file(self, filename, popularity_cutoff,
                                     normalizer):
        """Updates this gazetteer with data from an entity data file.

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

    def update_with_entity_map(self, mappings, numeric_types, normalizer):
        for mapping_key, mapping_value in mappings.items():
            if mapping_key == '*':
                continue
            mapping_key = normalizer(mapping_key)
            self._update_entity(mapping_key, 1)
            self.numtypes.update(util.get_nested_numeric_types(mapping_key))
        self.numtypes.update(numeric_types)
        logging.info("Nested numeric types for '{}' facet: {}".format(
            self.facet_name, map(str, self.numtypes)))

    def update_with_synonyms_file(self, filename, normalizer,
                                  update_if_missing_canonical=True):
        """Updates this gazetteer with data from a synonyms file.

        Args:
            filename (str): The filename of the entity data file.
            popularity_cutoff (float): A threshold at which entities with
                popularity below this value are ignored.
            normalizer (function): A function that normalizes text.
        """
        logging.info('Loading synonyms from {}'.format(filename))
        line_count = 0
        synonyms_added = 0
        missing_canonicals = 0
        min_popularity = 0
        if len(self.edict) > 0:
            min_popularity = min(self.edict.values())
        for canonical, synonym in util.read_tsv_lines(filename, 2):
            line_count += 1
            canonical = normalizer(canonical)
            synonym = normalizer(synonym)

            if update_if_missing_canonical or canonical in self.edict:
                self._update_entity(
                    synonym, self.edict.get(canonical, min_popularity))
                synonyms_added += 1
            if canonical not in self.edict:
                missing_canonicals += 1
                logging.debug(u"Synonym '{}' for entity '{}' not in gazetteer"
                              .format(synonym, canonical))
        logging.info('Added {}/{} synonyms from file into gazetteer'
                     .format(synonyms_added, line_count))
        if update_if_missing_canonical and missing_canonicals:
            logging.info('Loaded {} synonyms where the canonical name is not '
                         'in the gazetteer'.format(missing_canonicals))
