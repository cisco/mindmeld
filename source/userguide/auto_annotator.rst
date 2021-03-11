Working with the Auto Annotator
===============================

The Auto Annotator

  - is a tool to automatically annotate or unannotate select entities across all labelled data in an application.
  - supports the development of custom Annotators.

.. note::

   The examples in this section require the :doc:`HR Assistant <../blueprints/hr_assistant>` blueprint application. To get the app, open a terminal and run ``mindmeld blueprint hr_assistant``.
   Examples related to the MultiLingualAnnotator requires the :doc:`Health Screening <../blueprints/screening_app>` blueprint application. To get the app, open a terminal and run ``mindmeld blueprint hr_assistant``

.. warning::

   Changes by an Auto Annotator cannot be undone and MindMeld does not backup query data. We recommend using version control software such as Github.

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
We will demonstrate both approaches for annotation and unannotation using the :class:`MultiLingualAnnotator` class.

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

	from mindmeld.auto_annotator import MultiLingualAnnotator
	annotation_rules = [
		{
			"domains": ".*",
			"intents": ".*",
			"files": ".*",
			"entities": ".*",
		}
	]
	mla = MultiLingualAnnotator(
		app_path="hr_assistant",
		annotation_rules=annotation_rules,
		overwrite=True
	)
	mla.annotate()

If you do not want to annotate all supported entities, you can specify annotation rules instead.

For example, let's annotate :attr:`sys_person` entities from the :attr:`get_hierarchy_up` intent in the :attr:`hierarchy` domain.
To do this, we can add the following :attr:`AUTO_ANNOTATOR_CONFIG` dictionary to :attr:`config.py`.
Notice that we are setting :attr:`overwrite` to True since we want to replace the existing custom entity label, :attr:`name`.

.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = { 

		"annotator_class": "MultiLingualAnnotator",
		"overwrite": True, 
		"annotation_rules": [
			{ 
				"domains": "hierarchy", 
				"intents": "get_hierarchy_up", 
				"files": "train.txt",
				"entities": "sys_person", 
			}
		],
		"unannotate_supported_entities_only": True, 
		"unannotation_rules": None
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
To unannotate all annotations, use the the :attr:`unannotation_rules` shown below and set :attr:`unannotate_supported_entities_only` to False.

.. code-block:: python

	from mindmeld.auto_annotator import MultiLingualAnnotator
	unannotation_rules = [
		{
			"domains": ".*",
			"intents": ".*",
			"files": ".*",
			"entities": ".*",
		}
	]
	mla = MultiLingualAnnotator(
		app_path="hr_assistant",
		unannotation_rules=unannotation_rules,
		unannotate_supported_entities_only=False
	)
	mla.unannotate()

If you see the following message, you need to update the unannotate parameter in your custom :attr:`AUTO_ANNOTATOR_CONFIG` dictionary in :attr:`config.py`.
You can refer to the config specifications in the "Auto Annotator Configuration" section below.

.. code-block:: console

	'unannotate' field is not configured or misconfigured in the `config.py`. We can't find any file to unannotate.

If you do not want to unannotate all entities, you can can specify annotation rules to be used for unannotation in the :attr:`unannotate` param of your config.
For example, let's unannotate :attr:`sys_time` entities from the :attr:`get_date_range_aggregate` intent in the :attr:`date` domain.
To do this, we can add the following :attr:`AUTO_ANNOTATOR_CONFIG` dictionary to :attr:`config.py`.


.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = { 

		"annotator_class": "MultiLingualAnnotator",
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


Default Auto Annotator: MultiLingual Annotator
----------------------------------------------
The :mod:`mindmeld.auto_annotator` module contains an abstract :class:`Annotator` class.
This class serves as a base class for any MindMeld Annotator including the :class:`MultiLingualAnnotator` class.
The :class:`MultiLingualAnnotator` leverages `Spacy's Named Entity Recognition <https://spacy.io/usage/linguistic-features#named-entities>`_ system and duckling to detect entities.


Supported Entities and Languages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Up to 21 entities are supported across 15 languages. The table below defines these entities and whether they are resolvable by duckling.

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

Supported languages include English (en), Spanish (es), French (fr), German (de), Danish (da), Greek (el), Portuguese (pt), Lithuanian (lt), Norwegian Bokmal (nb), Romanian (ro), Polish (pl), Italian (it), Japanese (ja), Chinese (zh), Dutch (nl).
The table below identifies the supported entities for each language.

+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
|                     | EN | ES | FR | DE | DA | EL | PT | LT | NB | RO | PL | IT | JA | ZH | NL |
+=====================+====+====+====+====+====+====+====+====+====+====+====+====+====+====+====+
| sys_amount-of-money | y  | y  | y  | n  | n  | n  | y  | n  | y  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_distance        | y  | y  | y  | y  | n  | n  | y  | n  | n  | y  | n  | y  | n  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_duration        | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_event           | y  | n  | n  | n  | n  | y  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_fac             | y  | n  | n  | n  | n  | n  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_gpe             | y  | n  | n  | n  | n  | y  | n  | y  | n  | y  | y  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_interval        | y  | y  | y  | y  | y  | y  | y  | n  | y  | y  | y  | y  | n  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_language        | y  | n  | n  | n  | n  | n  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_law             | y  | n  | n  | n  | n  | n  | n  | n  | n  | n  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_loc             | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | n  | y  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_norp            | y  | n  | n  | n  | n  | n  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_number          | y  | y  | y  | y  | y  | y  | y  | n  | y  | y  | y  | y  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_ordinal         | y  | y  | y  | y  | y  | y  | y  | n  | y  | y  | y  | y  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_org             | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_other-quantity  | y  | n  | n  | n  | n  | n  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_percent         | y  | n  | n  | n  | n  | n  | n  | n  | n  | n  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_person          | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_product         | y  | n  | n  | n  | n  | y  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_time            | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | y  | n  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_weight          | y  | n  | n  | n  | n  | n  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+
| sys_work_of_art     | y  | n  | n  | n  | n  | n  | n  | n  | n  | y  | n  | n  | y  | y  | y  |
+---------------------+----+----+----+----+----+----+----+----+----+----+----+----+----+----+----+


Working with English Sentences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To detect entities in a single sentence first create an instance of the :class:`MultiLingualAnnotator` class.
If a language is not specified in :attr:`LANGUAGE_CONFIG` (:attr:`config.py`) then by default English will be used.

.. code-block:: python

	from mindmeld.auto_annotator import MultiLingualAnnotator 
	mla = MultiLingualAnnotator(app_path="hr_assistant")

Then use the :meth:`parse` function.

.. code-block:: python
	
	mla.parse("Apple stock went up $10 last monday.") 

Three entities are automatically recognized and a list of QueryEntity objects is returned. Each QueryEntity represents a detected entity.:

.. code-block:: python
	
	[
		<QueryEntity 'Apple' ('sys_org') char: [0-4], tok: [0-0]>,
		<QueryEntity '$10' ('sys_amount-of-money') char: [20-22], tok: [4-4]>,
		<QueryEntity 'last monday' ('sys_time') char: [24-34], tok: [5-6]>
	]

The Auto Annotator detected "Apple" as :attr:`sys_org`. Moreover, it recognized "$10" as :attr:`sys_amount-of-money` and resolved its :attr:`value` as 10 and :attr:`unit` as "$".
Lastly, it recognized "last monday" as :attr:`sys_time` and resolved its :attr:`value` to be a timestamp representing the last monday from the current date.

To restrict the types of entities returned from the :attr:`parse()` method use the :attr:`entity_types` parameter and pass in a list of entities to restrict parsing to. By default, all entities are allowed.
For example, we can restrict the output of the previous example by doing the following:


.. code-block:: python
	
	allowed_entites = ["sys_org", "sys_amount-of-money", "sys_time"]
	sentence = "Apple stock went up $10 last monday."
	mla.parse(sentence=sentence, entity_types=allowed_entities)

Working with Non-English Sentences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`MultiLingualAnnotator` will use the language and locale specified in the :attr:`LANGUAGE_CONFIG` (:attr:`config.py`) if it used through the command-line.

.. code-block:: python
	
	LANGUAGE_CONFIG = {'language': 'es'}

Many Spacy non-English NER models have limited entity support. To overcome this, in addition to the entities detected by non-English NER models, the :class:`MultiLingualAnnotator` translates the sentence to English and detects entities
using the English NER model. The English detected entities are compared against duckling candidates for the non-English sentence. Duckling candidates with a match between the type and value of the entity or the translated body text
are selected. If a translation service is not available, the :class:`MultiLingualAnnotator` selects the duckling candidates with the largest non-overlapping spans. The sections below describe the steps to setup the annotator depending on whether a translation service is being used.

Annotating with a Translation Service (Google)
''''''''''''''''''''''''''''''''''''''''''''''
The :class:`MultiLingualAnnotator` can leverage the Google Translation API to better detect entities in non-English sentences. To use this feature, export your Google application credentials.

.. code-block:: console

	export GOOGLE_APPLICATION_CREDENTIALS="/<YOUR_PATH>/google_application_credentials.json"

Install the extras requirements for annotators.

.. code-block:: console

	pip install mindmeld[language_annotator]

Finally, specify the translator in :attr:`AUTO_ANNOTATOR_CONFIG`. Set :attr:`translator` to :attr:`GoogleTranslator`.

Annotating without a Translation Service
''''''''''''''''''''''''''''''''''''''''
We can still use the :class:`MultiLingualAnnotator` without a translation service. To do so, set :attr:`translator` to :attr:`NoOpTranslator` in :attr:`AUTO_ANNOTATOR_CONFIG`.

Spanish Sentence Example
''''''''''''''''''''''''
Let's take a look at an example of the :class:`MultiLingualAnnotator` detecting entities in Spanish sentences.  
To use a Spanish MindMeld application we can download the :attr:`Screening App` blueprint with the following command:

.. code-block:: console

	mindmeld blueprint screening_app

We can now create our :class:`MultiLingualAnnotator` object and pass in the app_path. If a spanish Spacy model is not found in the environment, it will automatically be downloaded.

.. code-block:: python

	from mindmeld.auto_annotator import MultiLingualAnnotator 
	mla = MultiLingualAnnotator(
		app_path="screening_app",
		language="es",
		locale=None,
	)

Then use the :meth:`parse` function.

.. code-block:: python
	
	mla.parse("Las acciones de Apple subieron $10 el lunes pasado.") 

Three entities are automatically recognized.

.. code-block:: python
	
	[
		<QueryEntity 'Apple' ('sys_org') char: [16-20], tok: [3-3]>,
		<QueryEntity 'el lunes pasado' ('sys_time') char: [35-49], tok: [6-8]>,
		<QueryEntity '$10' ('sys_amount-of-money') char: [31-33], tok: [5-5]>
	]


Auto Annotator Configuration
----------------------------

The :attr:`DEFAULT_AUTO_ANNOTATOR_CONFIG` shown below is the default config for an Annotator.
A custom config can be included in :attr:`config.py` by duplicating the default config and renaming it to :attr:`AUTO_ANNOTATOR_CONFIG`.
Alternatively, a custom config dictionary can be passed in directly to :class:`MultiLingualAnnotator` or any Annotator class upon instantiation.


.. code-block:: python

	DEFAULT_AUTO_ANNOTATOR_CONFIG = { 

		"annotator_class": "MultiLingualAnnotator",
		"overwrite": False, 
		"annotation_rules": [ 
			{ 
				"domains": ".*", 
				"intents": ".*", 
				"files": ".*", 
				"entities": ".*", 
			} 
		], 
		"unannotate_supported_entities_only": True, 
		"unannotation_rules": None, 
	}

Let's take a look at the allowed values for each setting in an Auto Annotator configuration.


``'annotator_class'`` (:class:`str`): The class in auto_annotator.py to use for annotation when invoked from the command line. By default, :class:`MultiLingualAnnotator` is used. 

``'overwrite'`` (:class:`bool`): Whether new annotations should overwrite existing annotations in the case of a span conflict. False by default. 

``'annotation_rules'`` (:class:`list`): A list of annotation rules where each rule is represented as a dictionary. Each rule must have four keys: :attr:`domains`, :attr:`intents`, :attr:`files`, and :attr:`entities`.
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
Internally, the above rule is combined to a single pattern: "(faq|salary)/.*/(train.txt|test.txt)" and this pattern is matched against all file paths in the domain folder of your MindMeld application. 

.. warning::

	The order of the annotation rules matters. Each rule overwrites the list of entities to annotate for a file if the two rules include the same file. It is good practice to start with more generic rules first and then have more specific rules.
	Be sure to use the regex "or" (:attr:`|`) if applying rules at the same level of specificity. Otherwise, if written as separate rules, the latter will overwrite the former.

.. warning::
	By default, all files in all intents across all domains will be annotated with all supported entities. Before annotating consider including custom annotation rules in :attr:`config.py`. 

``'language'`` (:class:`str`): Language as specified using a 639-1/2 code.

``'locale'`` (:class:`str`): The locale representing the ISO 639-1 language code and ISO3166 alpha 2 country code separated by an underscore character.

``'unannotate_supported_entities_only'`` (:class:`boolean`): By default, when the unannotate command is used only entities that the Annotator can annotate will be eligible for removal. 

``'unannotation_rules'`` (:class:`list`): List of annotation rules in the same format as those used for annotation. These rules specify which entities should have their annotations removed. By default, :attr:`files` is None.

``'spacy_model_size'`` (:class:`str`): :attr:`lg` is used by default for the best performance. Alternative options are :attr:`sm` and :attr:`md`. This parameter is optional and is specific to the use of the :class:`SpacyAnnotator` and :class:`MultiLingualAnnotator`.
If the selected model is not in the current environment it will automatically be downloaded. Refer to Spacy's documentation to learn more about their `NER models <https://spacy.io/models/>`_.

``'translator'`` (:class:`str`): This parameter is used by the :class:`MultiLingualAnnotator`. If Google application credentials are available and have been exported, set this parameter to :attr:`GoogleTranslator`. Otherwise, set this paramter to :attr:`NoOpTranslator`.

Using the Bootstrap Annotator
-----------------------------
The :class:`BootstrapAnnotator` speeds up the data annotation process of new queries. When a :class:`BootstrapAnnotator` is instantiated a :class:`NaturalLanguageProcessor` is built for your app. For each intent, an entity recognizer is trained on the existing labeled data.
The :class:`BootstrapAnnotator` uses these entity recognizers to predict and label the entities for your app if you have existing labeled queries. The :class:`BootstrapAnnotator` labels the entities for new queries using the trained entity recognizer for each given intent.

First, ensure that files that you would like to label have the same name or pattern. For example, you may label your files :attr:`train_bootstrap.txt` across all intents.

Update the :attr:`annotator_class` field in your :attr:`AUTO_ANNOTATOR_CONFIG` to be :class:`BootstrapAnnotator` and set your annotation rules to include your desired patterns.
You can optionally set the :attr:`confidence_threshold` for labeling in the config as shown below. For this example, we will set it to 0.95. This means that entities will only be labeled if the entity recognizer assigns a confidence score over 95% to the entity.

.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = {
		"annotator_class": "BootstrapAnnotator",
		"confidence_threshold": 0.95,
		...
		"annotation_rules": [
			{
				"domains": ".*",
				"intents": ".*",
				"files": ".*bootstrap.*\.txt",
				"entities": ".*",
			}
		],
	}

Check your :attr:`ENTITY_RECOGNIZER_CONFIG` in :attr:`config.py`. Make sure that you explicitly specify the regex pattern for training and testing and that this pattern does not overlap with the pattern for your unlabeled data (E.g. :attr:`train_bootstrap.txt`).

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
	annotation_rules: [
		{
			"domains": ".*",
			"intents": ".*",
			"files": ".*bootstrap.*\.txt",
			"entities": ".*",
		}
	]
	ba = BootstrapAnnotator(
		app_path="hr_assistant",
        annotation_rules=annotation_rules,
        confidence_threshold=0.95,
	)
	ba.annotate()

.. note::

   The Bootstrap Annotator is different from the :attr:`predict` command-line function. Running ``python -m hr_assistant predict train_bootstrap.txt -o labeled.tsv`` will output a tsv with annotated queries.
   Unlike the Bootstrap Annotator, the :attr:`predict` only annotates a single file and does not use the entity recognizer of a specific intent. Instead, it uses the intent classified by :attr:`nlp.process(query_text)`.

Creating a Custom Annotator
---------------------------
The :class:`MultiLingualAnnotator` is a subclass of the abstract base class :class:`Annotator`.
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

		def __init__(
			self,
			app_path,
			annotation_rules=None,
			language=None,
			locale=None,
			overwrite=False,
			unannotate_supported_entities_only=True,
			unannotation_rules=None,
			custom_param=None,
		):
			super().__init__(
				app_path,
				annotation_rules=annotation_rules,
				language=language,
				locale=locale,
				overwrite=overwrite,
				unannotate_supported_entities_only=unannotate_supported_entities_only,
				unannotation_rules=unannotation_rules,
			)
			self.custom_param = custom_param
			# Add additional params to init if needed

		def parse(self, sentence, entity_types=None, **kwargs):
			""" 
			Args:
				sentence (str): Sentence to detect entities.
				entity_types (list): List of entity types to parse. If None, all
					possible entity types will be parsed.
			Returns:
				query_entities (list[QueryEntity]): List of QueryEntity objects.
			"""

			# TODO: Add custom parse logic

			return query_entities

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
		annotation_rules: [
			{
				"domains": ".*",
				"intents": ".*",
				"files": ".*",
				"entities": ".*",
			}
		]
		custom_annotator = CustomAnnotator(
			app_path="hr_assistant",
			annotation_rules=annotation_rules,
		)
		custom_annotator.annotate()


To run your custom Annotator, simply run in the command line: :attr:`python custom_annotator.py`.
To run unannotation with your custom Annotator, change the last line in your script to :attr:`custom_annotator.unannotate()`.

Getting Custom Parameters from the Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:attr:`spacy_model_size` is an example of an optional parameter in the config that is relevant only for a specific :class:`Annotator` class.

.. code-block:: python

	AUTO_ANNOTATOR_CONFIG = { 
		... 
		"spacy_model": "en_core_web_md",
		... 
	}

If a :class:`SpacyAnnotator` is created using the command-line, it will use the value for :attr:`spacy_model_size` that exists in the config during instantiation.

A similar approach can be taken for custom Annotators.
