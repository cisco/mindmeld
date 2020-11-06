Working with the Auto Annotator
===============================

The Auto Annotator

  - is a tool to automatically annotate or unannotate select entities across all labelled data in an application.
  - supports the development of custom Annotators.

.. note::

   The examples in this section require the :doc:`HR Assistant <../blueprints/hr_assistant>` blueprint application. To get the app, open a terminal and run ``mindmeld blueprint hr_assistant``.

.. warning::

   Changes by an Auto Annotator cannot be undone and Mindmeld does not backup query data. We recommend using version control software such as Github.

Quick Start
-----------
This section briefly explains the use of the ``annotate`` and ``unannotate`` commands. For more details, please read the next section.

Before annotating the data, we will first remove all existing annotations using ``unannotate``. Be sure to include the :attr:`--unannotate_all` flag when running the following command in the command-line.

Command-line:

.. code-block:: console

	mindmeld unannotate --app-path "hr_assistant" --unannotate_all

We can now proceed to ``annotate`` our data using the command below.

Command-line:

.. code-block:: console

	mindmeld annotate --app-path "hr_assistant"

The following section explains this same process in more detail.

Using the Auto Annotator
------------------------

The Auto Annotator can be used by importing a class that implements the :class:`Annotator` abstract class in the :mod:`auto_annotator` module or through the command-line.
We will demonstrate both approaches for annotation and unannotation using the :class:`SpacyAnnotator` class.

Annotate
^^^^^^^^

By default, all entity types supported by an Annotator will by annotated if they do not overlap with existing annotations.

You can :attr:`annotate` using the command-line.
To overwrite existing annotations that overlap with new annotations, pass in the optional param :attr:`--overwrite`.

.. code-block:: console

	mindmeld annotate --app-path "hr_assistant" --overwrite

Alternatively, you can annotate by creating an instance of the :class:`Annotator` class and running the Python code below.
An optional param :attr:`overwrite` can be passed in here as well.

.. code-block:: python

	from mindmeld.auto_annotator import SpacyAnnotator 
	sa = SpacyAnnotator(app_path="hr_assistant")

	sa.annotate(overwrite=True)

If you do not want to annotate all supported entities, you can specify annotation rules instead.

For example, let's annotate :attr:`sys_person` entities from the :attr:`get_hierarchy_up` intent in the :attr:`hierarchy` domain.
To do this, we can add the following :attr:`AUTO_ANNOTATOR_CONFIG` dictionary to :attr:`config.py`.
Notice that we are setting :attr:`overwrite` to True since we want to replace the existing custom entity label, :attr:`name`.

.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = { 

		"annotator_class": "SpacyAnnotator",
		"overwrite": True, 
		"annotate": [
			{ 
				"domains": "hierarchy", 
				"intents": "get_hierarchy_up", 
				"files": "train.txt",
				"entities": "sys_person", 
			}
		],
		"unannotate_supported_entities_only": True, 
		"unannotate": None
	}

Before running the annotation, let's take a look at the first four queries in the train.txt file for the :attr:`get_hierarchy_up` intent: 

.. code-block:: none

	I wanna get a list of all of the employees that are currently {manage|manager} {caroline|name}
	I wanna know {Tayana Jeannite|name}'s person in {leadership|manager} of her?
	is it correct to say that {Angela|name} is a {boss|manager}?
	who all is {management|manager} of {tayana|name}

After running :attr:`annotate` we find that instances of :attr:`sys_person` have been labelled and have overwritten previous instances of the custom entity, :attr:`name`.

.. code-block:: none

	I wanna get a list of all of the employees that are currently {manage|manager} {caroline|sys_person}
	I wanna know {Tayana Jeannite|sys_person}'s person in {leadership|manager} of her?
	is it correct to say that {Angela|sys_person} is a {boss|manager}?
	who all is {management|manager} of {tayana|sys_person}

You can annotate with multiple annotation rules. For more details on annotation rules please read the "Auto Annotator Configuration" section below.

Unannotate
^^^^^^^^^^
By default, only the entities that are supported by an Annotator will be unannotated.

You can :attr:`unannotate` using the command-line. To unannotate all entities, pass in the optional param :attr:`--unannotate_all`.

.. code-block:: console

	mindmeld unannotate --app-path "hr_assistant" --unannotate_all

To unannotate by creating an instance of the :class:`Annotator` class, run the Python code below.
To unannotate all annotations, pass in the optional param :attr:`unannotate_all`.

.. code-block:: python

	from mindmeld.auto_annotator import SpacyAnnotator 
	sa = SpacyAnnotator(app_path="hr_assistant")

	sa.unannotate(unannotate_all=True)

If :attr:`unannotate_all` is not set to True and you see the following message, you need to update the unannotate parameter in your custom :attr:`AUTO_ANNOTATOR_CONFIG` dictionary in :attr:`config.py`.
You can refer to the config specifications in the "Auto Annotator Configuration" section below.

.. code-block:: console

	'unannotate' field is not configured or misconfigured in the `config.py`. We can't find any file to unannotate.

If you do not want to unannotate all entities, you can can specify annotation rules to be used for unannotation in the :attr:`unannotate` param of your config.
For example, let's unannotate :attr:`sys_time` entities from the :attr:`get_date_range_aggregate` intent in the :attr:`date` domain.
To do this, we can add the following :attr:`AUTO_ANNOTATOR_CONFIG` dictionary to :attr:`config.py`.


.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = { 

		"annotator_class": "SpacyAnnotator",
		"overwrite": False, 
		"annotate": [{"domains": ".*", "intents": ".*", "files": ".*", "entities": ".*"}],
		"unannotate_supported_entities_only": True, 
		"unannotate": [
			{ 
				"domains": "date", 
				"intents": "get_date_range_aggregate", 
				"files": "train.txt",
				"entities": "sys_time", 
			}
		], 
	}

.. note::

	The content of :attr:`annotate` in the config has no effect on unannotation. Similarly, :attr:`unannotate` in the config has no affect on annotation. These processes are independent and are only affected by the corresponding parameter in the config.

Before running the unannotation, let's take a look at the first four queries in the train.txt file for the :attr:`get_date_range_aggregate` intent: 

.. code-block:: none

	{sum|function} of {non-citizen|citizendesc} people {hired|employment_action} {after|date_compare} {2005|sys_time}
	What {percentage|function} of employees were {born|dob} {before|date_compare} {1992|sys_time}?
	{us citizen|citizendesc} people with {birthday|dob} {before|date_compare} {1996|sys_time} {count|function}
	{count|function} of {eligible non citizen|citizendesc} workers {born|dob} {before|date_compare} {1994|sys_time}

After running :attr:`unannotate` we find that instances of :attr:`sys_time` have been unannotated as expected.

.. code-block:: none

	{sum|function} of {non-citizen|citizendesc} people {hired|employment_action} {after|date_compare} 2005
	What {percentage|function} of employees were {born|dob} {before|date_compare} 1992?
	{us citizen|citizendesc} people with {birthday|dob} {before|date_compare} 1996 {count|function}
	{count|function} of {eligible non citizen|citizendesc} workers {born|dob} {before|date_compare} 1994


Default Auto Annotator: Spacy Annotator
---------------------------------------
The :mod:`mindmeld.auto_annotator` module contains an abstract :class:`Annotator` class.
This class serves as a base class for any Mindmeld Annotator including the :class:`SpacyAnnotator` class.
The :class:`SpacyAnnotator` leverages `Spacy's Named Entity Recognition <https://spacy.io/usage/linguistic-features#named-entities>`_ system to detect 21 different entities.
Some of these entities are resolvable by Duckling. 


+------------------------+-------------------------+-----------------------------------------------------------------------------+
| Supported Entities     | Resolvable by Duckling  | Examples or Definition                                                      |
+========================+=========================+=============================================================================+
| "sys_time"             | Yes                     | "today", "Tuesday, Feb 18" , "last week"                                    |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_interval"         | Yes                     | "from 9:30 to 11:00am", "Monday to Friday", "Tuesday 3pm to Wednesday 7pm"  |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_duration"         | Yes                     | "2 hours", "15 minutes", "3 days"                                           |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_number"           | Yes                     | "58", "two hundred", "1,394,345.45"                                         |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_amount-of-money"  | Yes                     | "ten dollars", "seventy-eight euros", "$58.67"                              |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_distance"         | Yes                     | "500 meters", "498 miles", "47.5 inches"                                    |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_weight"           | Yes                     | "400 pound", "3 grams", "47.5 mg"                                           |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_ordinal"          | Yes                     | "3rd place" ("3rd"), "fourth street" ("fourth"),  "5th"                     |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_percent"          | Yes                     | "four percent", "12%", "5 percent"                                          |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_org"              | No                      | "Cisco", "IBM", "Google"                                                    |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_loc"              | No                      | "Europe", "Asia", "the Alps", "Pacific ocean"                               |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_person"           | No                      | "Blake Smith", "Julia", "Andy Neff"                                         |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_gpe"              | No                      | "California", "FL", "New York City", "USA"                                  |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_norp"             | No                      | Nationalities or religious or political groups.                             |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_fac"              | No                      | Buildings, airports, highways, bridges, etc.                                |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_product"          | No                      | Objects, vehicles, foods, etc. (Not services.)                              |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_event"            | No                      | Named hurricanes, battles, wars, sports events, etc.                        |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_law"              | No                      | Named documents made into laws.                                             |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_language"         | No                      | Any named language.                                                         |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_work-of-art"      | No                      | Titles of books, songs, etc.                                                |
+------------------------+-------------------------+-----------------------------------------------------------------------------+
| "sys_other-quantity"   | No                      | "10 joules", "30 liters", "15 tons"                                         |
+------------------------+-------------------------+-----------------------------------------------------------------------------+


To detect entities in a single sentence first create an instance of the :class:`SpacyAnnotator` class.

.. code-block:: python

	from mindmeld.auto_annotator import SpacyAnnotator 
	sa = SpacyAnnotator(app_path="hr_assistant")

Then use the :meth:`parse` function.

.. code-block:: python
	
	sa.parse("Apple stock went up $10 last monday.") 

Three entities are automatically recognized and a list of dictionaries is returned. Each dictionary represents a detected entity.:

.. code-block:: python
	
	[
		{
			'body': 'Apple',
			'start': 0,
			'end': 5,
			'value': {'value': 'Apple'},
			'dim': 'sys_org'
		},
		{
			'body': '$10',
			'start': 20,
			'end': 23,
			'value': {'value': 10, 'type': 'value', 'unit': '$'},
			'dim': 'sys_amount-of-money'
		},
		{
			'body': 'last monday',
			'start': 24,
			'end': 35,
			'value': {'value': '2020-09-21T00:00:00.000-07:00',
			'grain': 'day',
			'type': 'value'},
			'dim': 'sys_time'
		}
	]

The Auto Annotator detected "Apple" as :attr:`sys_org`. Moreover, it recognized "$10" as :attr:`sys_amount-of-money` and resolved its :attr:`value` as 10 and :attr:`unit` as "$".
Lastly, it recognized "last monday" as :attr:`sys_time` and resolved its :attr:`value` to be a timestamp representing the last monday from the current date.

In general, detected entities will be represented in the following format:

.. code-block:: python

	entity = {

		"body": (substring of sentence), 
		"start": (start index), 
		"end": (end index + 1), 
		"dim": (entity type), 
		"value": (resolved value, if it exists), 

	}

To restrict the types of entities returned from the :attr:`parse()` method use the :attr:`entity_types` parameter and pass in a list of entities to restrict parsing to. By default, all entities are allowed.
For example, we can restrict the output of the previous example by doing the following:


.. code-block:: python
	
	allowed_entites = ["sys_org", "sys_amount-of-money", "sys_time"]
	sentence = "Apple stock went up $10 last monday."
	sa.parse(sentence=sentence, entity_types=allowed_entities) 

Auto Annotator Configuration
----------------------------

The :attr:`DEFAULT_AUTO_ANNOTATOR_CONFIG` shown below is the default config for an Annotator.
A custom config can be included in :attr:`config.py` by duplicating the default config and renaming it to :attr:`AUTO_ANNOTATOR_CONFIG`.
Alternatively, a custom config dictionary can be passed in directly to :class:`SpacyAnnotator` or any Annotator class upon instantiation.


.. code-block:: python

	DEFAULT_AUTO_ANNOTATOR_CONFIG = { 

		"annotator_class": "SpacyAnnotator",
		"overwrite": False, 
		"annotate": [ 
			{ 
				"domains": ".*", 
				"intents": ".*", 
				"files": ".*", 
				"entities": ".*", 
			} 
		], 
		"unannotate_supported_entities_only": True, 
		"unannotate": None, 
	}

Let's take a look at the allowed values for each setting in an Auto Annotator configuration.


``'annotator_class'`` (:class:`str`): The class in auto_annotator.py to use for annotation when invoked from the command line. By default, :class:`SpacyAnnotator` is used. 

``'overwrite'`` (:class:`bool`): Whether new annotations should overwrite existing annotations in the case of a span conflict. False by default. 

``'annotate'`` (:class:`list`): A list of annotation rules where each rule is represented as a dictionary. Each rule must have four keys: :attr:`domains`, :attr:`intents`, :attr:`files`, and :attr:`entities`.
Annotation rules are combined internally to create Regex patterns to match selected files. The character :attr:`'.*'` can be used if all possibilities in a section are to be selected, while possibilities within
a section are expressed with the usual Regex special characters, such as :attr:`'.'` for any single character and :attr:`'|'` to represent "or". 

.. code-block:: python

	{
		"domains": "(faq|salary)", 
		"intents": ".*", 
		"files": "(train.txt|test.txt)", 
		"entities": "(sys_amount-of-money|sys_time)", 
	}

The rule above would annotate all text files named "train" or "test" in the "faq" and "salary" domains. Only sys_amount-of-money and sys_time entities would be annotated.
Internally, the above rule is combined to a single pattern: "(faq|salary)/.*/(train.txt|test.txt)" and this pattern is matched against all file paths in the domain folder of your Mindmeld application. 

.. warning::

	The order of the annotation rules matters. Each rule overwrites the list of entities to annotate for a file if the two rules include the same file. It is good practice to start with more generic rules first and then have more specific rules.
	Be sure to use the regex "or" (:attr:`|`) if applying rules at the same level of specificity. Otherwise, if written as separate rules, the latter will overwrite the former.

.. warning::
	By default, all files in all intents across all domains will be annotated with all supported entities. Before annotating consider including custom annotation rules in :attr:`config.py`. 

``'unannotate_supported_entities_only'`` (:class:`boolean`): By default, when the unannotate command is used only entities that the Annotator can annotate will be eligible for removal. 

``'unannotate'`` (:class:`list`): List of annotation rules in the same format as those used for annotation. These rules specify which entities should have their annotations removed. By default, :attr:`files` is None.

``'spacy_model'`` (:class:`str`): :attr:`en_core_web_lg` is used by default for the best performance. Alternative options are :attr:`en_core_web_sm` and :attr:`en_core_web_md`. This parameter is optional and is specific to the use of the :class:`SpacyAnnotator`.
If the selected model is not in the current environment it will automatically be downloaded. Refer to Spacy's documentation to learn more about their `English models <https://spacy.io/models/en>`_. The Spacy Annotator is currently not designed to support other language but they may be used.

Using the Bootstrap Annotator
----------------------------
The :class:`BootstrapAnnotator` speeds up the data annotation process of new queries. When a :class:`BootstrapAnnotator` is instantiated a :class:`NaturalLanguageProcessor` is built for your app. For each intent, an entity recognizer is trained on the existing labeled data.
The :class:`BootstrapAnnotator` uses these entity recognizers to predict and label the entities for your app if you have existing labeled queries. The :class:`BootstrapAnnotator` labels the entities for new queries using the trained entity recognizer for the given.

First, ensure that files that you would like to label have the same name or pattern. For example, you may label your files :attr:`bootstrap.txt` across all intents.

Update the :attr:`annotator_class` field in your :attr:`AUTO_ANNOTATOR_CONFIG` to be :class:`BootstrapAnnotator` and set your annotation rules to include your desired patterns.
You can optionally set the :attr:`confidence_threshold` for labeling in the config as shown below. For this example, we will set it to 0.95. This means that entities will only be labeled if the entity recognizer assigns a confidence score over 95% to the entity.

.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = {
		"annotator_class": "BootstrapAnnotator",
		"confidence_threshold": 0.95,
		...
		"annotate": [
			{
				"domains": ".*",
				"intents": ".*",
				"files": ".*bootstrap.*\.txt",
				"entities": ".*",
			}
		],
	}

Check your :attr:`ENTITY_RECOGNIZER_CONFIG` in :attr:`config.py`. Make sure that you explicitly specify the regex pattern for training and testing and that this pattern does not overlap with the pattern for your unlabeled data (E.g. :attr:`bootstrap.txt`).

.. code-block:: python

	ENTITY_RECOGNIZER_CONFIG = {
		...
		'train_label_set': 'train.*\.txt',
		'test_label_set': 'test.*\.txt'
	}

To run from the command line:

.. code-block:: console

	mindmeld annotate --app-path "hr_assistant"

Alternatively, you can annotate by creating an instance of the :class:`BootstrapAnnotator` class and running the Python code below.
An optional param :attr:`overwrite` can be passed in here as well.

.. code-block:: python

	from mindmeld.auto_annotator import BootstrapAnnotator 
	ba = BootstrapAnnotator(app_path="hr_assistant")

	ba.annotate(overwrite=True)


Creating a Custom Annotator
---------------------------
The :class:`SpacyAnnotator` is a subclass of the abstract base class :class:`Annotator`.
The functionality for annotating and unannotating files is contained in :class:`Annotator` itself.
A developer simply needs to implement two methods to create a custom annotator.


Custom Annotator Boilerplate Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This section includes boilerplate code to build a :class:`CustomAnnotator` class to which you can add to your own python file, let's call it :attr:`custom_annotator.py`
There are two "TODO"s. To implement a :class:`CustomAnnotator` class a developer has to implement the :meth:`parse` and :meth:`supported_entity_types` methods.

.. code-block:: python

	class CustomAnnotator(Annotator):
		""" Custom Annotator class used to generate annotations.
		"""

		def __init__(self, app_path, config=None):
			super().__init__(app_path=app_path, config=config)
			
			# Add additional attributes if needed

		def parse(self, sentence, entity_types=None):
			""" 
			Args:
				sentence (str): Sentence to detect entities.
				entity_types (list): List of entity types to parse. If None, all
					possible entity types will be parsed.
			Returns: entities (list): List of entity dictionaries.
			"""

			# TODO: Add custom parse logic

			return entities

		@property
		def supported_entity_types(self):
			"""
			Returns:
				supported_entity_types (list): List of supported entity types.
			"""

			# TODO: Add the entities supported by CustomAnnotator to supported_entities (list)

			supported_entities = []
			return supported_entities
	
	if __name__ == "__main__":
		custom_annotator = CustomAnnotator(app_path="hr_assistant")
		custom_annotator.annotate()

Entities returned by :attr:`parse()` must have the following format:

.. code-block:: python

	entity = { 
		"body": (substring of sentence), 
		"start": (start index), 
		"end": (end index + 1), 
		"dim": (entity type), 
		"value": (resolved value, if it exists), 
	}

To run your custom Annotator, simply run in the command line: :attr:`python custom_annotator.py`.
To run unannotation with your custom Annotator, change the last line in your script to :attr:`custom_annotator.unannotate()`.

Getting Custom Parameters from the Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`spacy_model` is an example of an optional parameter in the config that is relevant only for a specific :class:`Annotator` class.

.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = { 
		... 
		"spacy_model": "en_core_web_md",
		... 
	}

:class:`SpacyAnnotator` checks if :attr:`spacy_model` exists in the config, and if it doesn't, it will use the default value of "en_core_web_lg".

Custom parameters for custom annotators can be implemented in a similar fashion.
