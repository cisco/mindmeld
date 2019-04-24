MindMeld Blueprints
====================

MindMeld provides example applications, called *MindMeld blueprints*, that cover common conversational use cases. Each blueprint comes with a pre-configured application structure and a pre-built set of code samples and data sets.

The blueprint allows you to quickly build and test a fully working conversational app without writing code or collecting training data. If desired, you can then treat the blueprint app as a baseline for improvement and customization by adding data and logic specific to your business or application needs.

Blueprints are also ideal when you want to:

  - work through a tutorial on building practical applications using the MindMeld toolkit, or
  - bootstrap your app on MindMeld (assuming your app fits one of the MindMeld blueprint use cases)

MindMeld provides four blueprints:

+-------------------------------------------------+-----------------------------------------+------------------------------+
| | Blueprint                                     | |  Use Case                             | | Blueprint Name             |
| |                                               | |                                       | | in Python                  |
+=================================================+=========================================+==============================+
| |  :doc:`Food Ordering <food_ordering>`         | | Users order food for delivery         | | ``food_ordering``          |
| |                                               | | from nearby restaurants               | |                            |
+-------------------------------------------------+-----------------------------------------+------------------------------+
| |  :doc:`Home Assistant <home_assistant>`       | | Users control devices                 | | ``home_assistant``         |
| |                                               | | and appliances in a smart home        | |                            |
+-------------------------------------------------+-----------------------------------------+------------------------------+
| |  :doc:`Video Discovery <video_discovery>`     | | Users search for, ask questions about,| | ``video_discovery``        |
| |                                               | | and ask to view movies and TV shows   | |                            |
+-------------------------------------------------+-----------------------------------------+------------------------------+
| | :doc:`Kwik-E-Mart <../quickstart/00_overview>`| | The Kwik-E-Mart app described         | | ``kwik_e_mart``            |
| |                                               | | in the Step-by-Step Guide             | |                            |
+-------------------------------------------------+-----------------------------------------+------------------------------+

You can use the Quick Start below to get any blueprint up and running.

Each of the sections beyond the Quick Start covers one blueprint in depth, with exercises that explain what you need to know to convert the blueprint into a production-quality app.

You'll be guided through the following steps:

  - define the app scope clearly in terms of domains, intents, entities and roles
  - set up a MindMeld project by creating the app directory structure
  - create the knowledge base for the app
  - acquire and annotate training data for the statistical NLP models
  - train, tune and test the NLP models
  - configure the language parser
  - design and implement the dialogue states
  - deploy the app to a conversational platform or device

Before you Continue
-------------------

All blueprints require you to install MindMeld and its required dependencies before you begin.

See the :doc:`Getting Started <../userguide/getting_started>` page for instructions on acquiring and installing the MindMeld toolkit on your system.

The Home Assistant blueprint requires you to register for an `Open Weather Map <https://openweathermap.org/appid>`_ API key, and then set an environment variable with the command ``export OPEN_WEATHER_KEY=[YOUR-KEY]``. If you skip this step, the app can run but cannot retrieve weather forecasts.

Quick Start
-----------

Depending on which blueprint you choose to run, this Quick Start should take between five and fifteen minutes to complete.

1. Download
^^^^^^^^^^^

Open a Python shell and type the following commands to download and set up the blueprint application of your choice.

  - Use the appropriate Python blueprint name as the value of the ``bp_name`` variable (either ``food_ordering``, ``home_assistant``, ``video_assistant``, or ``kwik_e_mart``). In the example, we specify the Food Ordering blueprint.

.. code:: python

   import mindmeld as mm
   mm.configure_logs()
   bp_name = 'food_ordering'
   mm.blueprint(bp_name)

2. Build
^^^^^^^^

Build the Natural Language Processing models that power the app.

.. code:: python

   from mindmeld.components import NaturalLanguageProcessor
   nlp = NaturalLanguageProcessor(bp_name)
   nlp.build()


3. Run
^^^^^^

Interact with the app in the Python shell using the commands below. Try the queries shown in the examples, then try some queries of your own invention.

*Food Ordering example*

.. code:: python

   from mindmeld.components.dialogue import Conversation
   conv = Conversation(nlp=nlp, app_path=bp_name)
   conv.say('Hello!')

.. code-block:: console

   ["Hello. Some nearby popular restaurants you can order delivery from are Firetrail Pizza, Grandma's Deli & Cafe, The Salad Place"]

.. code-block:: python

   conv.say("Get me a pepperoni pizza from firetrail pizza")

.. code-block:: console

   ['Sure, I have 1 order of Pepperoni Pizza from Firetrail Pizza for a total price of $11.00. Would you like to place the order?', 'Listening...']

.. code-block:: python

   conv.say("Bye")

.. code-block:: console

   ['Goodbye!']

*Home Assistant example*

.. code:: python

    >>> from mindmeld.components.dialogue import Conversation
    >>> conv = Conversation(nlp=nlp, app_path=bp_name)
    >>> conv.say('Hi')
    ['Hi, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances.']
    >>> conv.say('What is the weather today?')
    ['The weather forecast in San Francisco is haze with a min of 66.2 F and a max of 89.6 F']
    >>> conv.say('Set the temperature to 72')
    ['The thermostat temperature in the home is now 72 degrees F.']

*Video Discovery example*

.. code:: python

    >>> from mindmeld.components.dialogue import Conversation
    >>> conv = Conversation(nlp=nlp, app_path='video_discovery')
    >>> conv.say('Hi')
    ['Hello.', 'I can help you find movies and TV shows. What do you feel like watching today?', "Unsupported response: {'videos': [{'type': 'movie', 'title': 'Wonder Woman', 'release_year': 2017}, {'type': 'movie', 'title': 'Beauty and the Beast', 'release_year': 2017}, {'type': 'movie', 'title': 'Transformers: The Last Knight', 'release_year': 2017}, {'type': 'movie', 'title': 'Logan', 'release_year': 2017}, {'type': 'movie', 'title': 'The Mummy', 'release_year': 2017}, {'type': 'movie', 'title': 'Kong: Skull Island', 'release_year': 2017}, {'type': 'tv-show', 'title': 'Doctor Who', 'release_year': 2005}, {'type': 'tv-show', 'title': 'Game of Thrones', 'release_year': 2011}, {'type': 'tv-show', 'title': 'The Walking Dead', 'release_year': 2010}, {'type': 'movie', 'title': 'Pirates of the Caribbean: Dead Men Tell No Tales', 'release_year': 2017}]}", "Suggestions: 'Most popular', 'Most recent', 'Movies', 'TV Shows', 'Action', 'Dramas', 'Sci-Fi'"]
    >>> conv.say('Show me movies with Tom Hanks')
    ['Perfect. Here are some movies with Tom Hanks:', "Unsupported response: {'videos': [{'type': 'movie', 'title': 'Forrest Gump', 'release_year': 1994}, {'type': 'movie', 'title': 'Toy Story', 'release_year': 1995}, {'type': 'movie', 'title': 'Inferno', 'release_year': 2016}, {'type': 'movie', 'title': 'Cars', 'release_year': 2006}, {'type': 'movie', 'title': 'Toy Story 3', 'release_year': 2010}, {'type': 'movie', 'title': 'Toy Story 2', 'release_year': 1999}, {'type': 'movie', 'title': 'Sully', 'release_year': 2016}, {'type': 'movie', 'title': 'Saving Private Ryan', 'release_year': 1998}, {'type': 'movie', 'title': 'Catch Me If You Can', 'release_year': 2002}, {'type': 'movie', 'title': 'The Green Mile', 'release_year': 1999}]}"]
    >>> conv.say('romantic')
    ['Perfect. Here are some romance movies with Tom Hanks:', "Unsupported response: {'videos': [{'type': 'movie', 'title': 'Forrest Gump', 'release_year': 1994}, {'type': 'movie', 'title': 'Big', 'release_year': 1988}, {'type': 'movie', 'title': 'Larry Crowne', 'release_year': 2011}, {'type': 'movie', 'title': 'Joe Versus the Volcano', 'release_year': 1990}, {'type': 'movie', 'title': 'Splash', 'release_year': 1984}, {'type': 'movie', 'title': 'Sleepless in Seattle', 'release_year': 1993}, {'type': 'movie', 'title': 'The Money Pit', 'release_year': 1986}, {'type': 'movie', 'title': 'Toy Story 4', 'release_year': 2019}, {'type': 'movie', 'title': "You've Got Mail", 'release_year': 1998}, {'type': 'movie', 'title': 'Nothing in Common', 'release_year': 1986}]}"]
