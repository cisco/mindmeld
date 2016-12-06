Knowledge Base
===========================

In order to build Deep-Domain Question Answering systems for large content catalogs, a Knowledge Base is required. For narrow-vocabulary domains, it is likely possible to create short lists which capture every possible entity value. But for medium- and large-vocabulary domains, a Knowledge Base can be helpful for storing and exporting all the potential entity values (names of artists, actors, titles, products, etc).

A Knowledge Base (KB) can be in the form of a simple Key-Value store, a relational database, or graph database.  In Workbench, we support both a lightweight relational database (Sqlite) with a fast full-text index (FTS-5) and a more full-featured Elasticsearch database.

Knowledge Base Configuration
****************************

Each domain defines its own configuration for knowledge base.

.. code-block:: javascript

  {
    "knowledgebase-type": "elasticsearch",
    "elasticsearch-host": "search.prod",
    "elasticsearch-build-host": "search-prod.mindmeld.com",
    "elasticsearch-port": 9200,
    "elasticsearch-index-name": "my-app-index"
  }

If the KB type is "elasticsearch", the following parameters need to be defined.

  +--------------------------+-----------------------------------------------+
  | Parameter                | Description                                   |
  +==========================+===============================================+
  | knowledgebase-type       | Valid values are “sqlite” and “elasticsearch” |
  +--------------------------+-----------------------------------------------+
  | elasticsearch-host       | ES cluster host name (used for AWS hosted env)|
  +--------------------------+-----------------------------------------------+
  | elasticsearch-build-host | ES cluster host name                          |
  +--------------------------+-----------------------------------------------+
  | elasticsearch-port       | ES cluster port                               |
  +--------------------------+-----------------------------------------------+
  | elasticsearch-index-name | name of ES index                              |
  +--------------------------+-----------------------------------------------+


Knowledge Base Schema
*********************

As part of building any database, a schema needs to be defined. This can be defined in the **"schema.json"** file for the Music-Assistant example as follows.

.. code-block:: javascript

  {
    "object-type": "music",
    "popularity-field": "reviewcount",
    "type-field": "type",
    "fields": [
      {
        "old-name": "oldid",
        "new-name": "id",
        "type": "ID"
      },
      {
        "old-name": "title",
        "new-name": "title",
        "type": "TEXT",
        "detect-entities": true
      },
      {
        "old-name": "type",
        "new-name": "type",
        "type": "TEXT"
      },
      {
        "old-name": "genres",
        "new-name": "genres",
        "type": "LIST",
        "detect-entities": true
      },
      {
        "old-name": "number-of-reviews",
        "new-name": "reviewcount",
        "type": "INTEGER",
        "do-not-index": true
      },
      ...
    ]
  }

+------------------+---------------------------------------------------------------------------------------------------------+
| Field            | Description                                                                                             |
+==================+=========================================================================================================+
| object-type      | the name for the table of documents.                                                                    |
+------------------+---------------------------------------------------------------------------------------------------------+
| popularity-field | the name of the field (of type INTEGER or REAL) which should be used for the default popularity ranking |
+------------------+---------------------------------------------------------------------------------------------------------+
| type-field       | the name of the field which defines the object type (if any)                                            |
+------------------+---------------------------------------------------------------------------------------------------------+
| old-name         | the name of the field in the json file to be imported                                                   |
+------------------+---------------------------------------------------------------------------------------------------------+
| new-name         | the name of the field in the knowledge base                                                             |
+------------------+---------------------------------------------------------------------------------------------------------+
| type             | the data type for this field (ID, TEXT, LIST, INTEGER, REAL, DATE, JSON)                                |
+------------------+---------------------------------------------------------------------------------------------------------+
| detect-entities  | a flag which indicates if this field should be used for extracting entity data files                    |
+------------------+---------------------------------------------------------------------------------------------------------+
| do-not-index     | a flag to indicate if this field should be stored in the full-text index                                |
+------------------+---------------------------------------------------------------------------------------------------------+

Create The Knowledge Base
*************************

Once the above dependencies are fulfilled, we are ready to import the data into the Knowledge Base.

.. code-block:: python

  from mindmeld.knowledge_base import KnowledgeBase

  # Initialize the KB
  kb = KnowledgeBase('/path/to/kb_config')

  # Read the data - can be read from flat files, cloud storage, or data-stream API
  data = read_data()

  # Import Data to KB
  kb.import_data(data, format='json')

Running **import_data** will setup a new Elasticsearch index with the latest imported data.

Advanced Options
****************

.. _here: https://www.elastic.co/

Elasticsearch (ES) is a versatile search engine based on Lucene. It provides a distributed, multitenant-capable full-text search engine with an HTTP web interface and schema-free JSON documents. The full set of documentation on tuning and adapting Elasticsearch to your needs is available here_.

Specifically, you might need to tweak the **"es_mapping.json"** file in your domain. The **es_mapping** file defines how a document, and the fields it contains, are stored and indexed. This mapping definition is used to create an ES index. ES is flexible on text analysis and indexing. It can be configured to -

* Support defining text analysis behavior per field
* Support defining sub-fields to process and index text differently for the same field.

An **Analyzer** can be defined for each document field to specify the desirable behavior for tokenizing and filtering the field values. A built-in default analyzer available in ES. Custom analyzers can be defined based on available tokenizers and filters. It is also possible to customize tokenizer and filter in ES. More details available in the Elasticsearch documentation.
