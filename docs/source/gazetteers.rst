Gazetteers
==========

Gazetteers are data structures that encapsulate information about entities relevant to a domain. This includes lists and dictionaries containing names of entities such as cities, organisations, days of the week etc. With a large vocabulary content catalog, it is hugely valuable to build a Gazetteer by extracting relevant entities from the Knowledge Base. The Gazetteer serves as a fast and efficient lookup when determining if phrases of text match with a known Knowledge Base entity. This provides tremendous power to the Machine Learning models in training on a large set of queries with known entities.

.. note:: Etymology of the word "Gazetteer"

  The Oxford English Dictionary defines "gazetteer" as a "geographical index or dictionary". It includes as an example a work by the British historian Laurence Echard (d. 1730) in 1693 that bore the title "The Gazetteer's: or Newsman's Interpreter: Being a Geographical Index".


Extracting Entity Data
----------------------

In a Deep Domain setting, the Knowledge Base may contain thousands, millions or even 100s of millions of entities. To eventually feed in the popular entities into a Gazetteer, we first extract the entities from the Knowledge Base and dump them into flat files (known as **Entity Data** files).

The entity data files are simply tsv text files which contain lists of important entity values and their popularity. 

.. code-block:: text

  0.00009602	Magda Horváth
  0.00475838	Alex Zahara
  0.00897593	Spencer Vrooman
  0.00119817	Hikmet Gül
  0.01335798	Lisa LoCicero
  0.00009390	Mary Ann Norment
  0.00095804	Richie Lawrence
  0.00422845	Barry Sigismondi
  ...           ...

The first column shows the relative popularity of the entities, and the second column contains the entities. If there are more than 10s of millions of documents in your Knowledge Base, it might be better to cut short the document list to scan based on the "popularity" field specified in the Knowledge Base schema.

.. code-block:: python

  from mindmeld.entity_extraction import EntityExtractor

  # Load KB Config
  with open('kb_conf.json') as conf_file:
    kb_conf = json.loads(conf_file)

  # Load KB schema
  with open('schema.json') as schema_file:
    kb_schema = json.loads(schema_file)

  # Extract Entities
  extractor = EntityExtractor(kb_conf, kb_schema)
  extractor.extract(num_docs=200000)


This generates entity data files for each schema entry in the KB schema.


Building The Gazetteer
----------------------

Gazetteers are built using 2 data sources - Entity Map and Entity Data files.

.. code-block:: python

  import mindmeld as mm

  # Load the Entity Map
  entity_map = mm.load_entity_map('/path/to/app/entity_map')

  entity_data_path = '/path/to/entity_data_files'

  mm.build_gazetteers(entity_map, entity_data_path)


Running the build_gazetters() function will generate pickle files with the gazetteers. The gazetteers can then be loaded in the various classifiers to power the **in-gaz** and **gaz-freq** features. Those will be discussed in more detail in the later chapters.