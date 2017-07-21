Home Assistant
==============

This page documents the Workbench blueprint for a conversational application for a smart home that allows users to control different devices and appliances.

This blueprint is great for:

   - Learning how to handle a large number of domains and intents.
   - Learning how to use system entities such as dates and times.
   - Learning how to use roles in the entity hierarchy.

1. The Use Case
^^^^^^^^^^^^^^^

This application provides a conversational interface for home automation systems. It allows users to interact with various appliances and home-related functions using natural language. With this application, users will be able to check the weather, set alarms, set timers, and control the lights, the doors, the thermostat and different appliances in the house.

2. Example Dialogue Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversational user flows for a home assistant application can get complex depending on the envisioned functionality and the amount of user guidance required at each step. This design exercise usually requires multiple iterations to finalize and enumerate all the possible user interactions. Below are examples of scripted dialogue interactions for a couple of possible user flows.

.. code:: bash

   App: Hi, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances.

   You: I want to turn on the lights in the kitchen

   App: Ok. The kitchen lights have been turned on.

   You: Turn the kitchen lights off

   App: Ok. The kitchen lights have been turned off.

   You: Turn on the thermostat

   App: Ok. The thermostat in the home has been turned on.

   You: Turn up the thermostat

   App: The thermostat temperature in the home is now 73 degrees F.

   You: Set the thermostat to 70

   App: The thermostat temperature in the home is now 70 degrees F.

   You: Lock all the doors

   App: Ok. All doors have been locked.

   You: What's the weather today?

   App: The weather forecast in San Francisco is clouds with a min of 66.2 F and a max of 87.8 F.

   You: Set a timer for 30 minutes

   App: Ok. A timer for 30 minutes has been set.

   You: Set alarm for 9am

   App: Ok, I have set your alarm for 09:00:00.

In this blueprint, the application provides a conversational interface for users to check weather, set alarms and timers, and control the lights, doors, thermostat and different appliances in the house.

3. Domain-Intent-Entity Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The home assistant blueprint is organized into five domains: Greeting, Smart Home, Time & Dates, Weather and Unknown. In contrast with the E-mart example, the home assistant blueprint requires more domains and intents as the application supports more activities. For example, turning on and off the lights require two intents, one for turning on and one for turning off. Similar logic applies for turning on/off appliance, closing/opening doors, locking/unlocking doors, etc, ...

Below is the full list of intents for every domain:

   - Greeting
       - greet
       - exit
   - Smart Home
       - check_thermostat
       - close_door
       - lock_door
       - open_door
       - set_thermostat
       - specify_location
       - turn_appliance_on
       - turn_appliance_off
       - turn_down_thermostat
       - turn_lights_off
       - turn_lights_on
       - turn_off_thermostat
       - turn_on_thermostat
       - turn_up_thermostat
       - unlock_door
   - Time and dates
       - change_alarm
       - check_alarm
       - set_alarm
       - start_timer
       - stop_timer
   - Weather
       - check_weather
   - Unknown
       - unknown

There are two types of entities: :doc:`Named Entities <../userguide/entity_recognizer>` and :doc:`System Entities <../userguide/system_entities>`. Named entities are defined and used by application; the full list of values for each entity is defined in the file ``gazetteer.txt`` under each entity folder. System entities are defined by Workbench, and there is no need to define them. Some examples of system entities are ``sys_temperature``, ``sys_time``, ``sys_interval``, etc.

Home assistant defines and uses the following named entities:

    - ``all``: this entity is used to detect whether the user has asked for all entities, for example: ``turn on the lights in {all|all} room``.
    - ``appliance``: this entity is used to detect household appliances, for example: ``can you turn on the {tv|appliance}?``
    - ``city``: this entity is used to detect cities, for example: ``what is the weather in {shanghai|city}``
    - ``color``: this entity is used to detect color of the lights, for example: ``turn the lights to {soft white|color}``
    - ``interval``: this entity is used to detect time interval, for example: ``cancel {tomorrow night|interval} s alarms``
    - ``location``: this entity is used to detect household location, for example: ``lock {back|location} door``
    - ``unit``: this entity is used to detect weather unit, for example: ``what is the forecast for {london|city} in {celsius|unit}``
    - ``duration``: this entity is used to detect time duration, for example: ``{15 minute|duration} alarm``

Home assistant uses three system entities: ``sys_time`` (time), ``sys_interval`` (interval) and ``sys_temperature`` (temperature). Some examples for annotation with system entities: ``set my thermostat to turn on at {6 am|sys_time}`` and ``turn the heat off at {76 degrees|sys_temperature}``.

In many queries, there might be more than one entity of the same type. For example, ``change my alarm from 7 am to 6 am``, both ``7 am`` and ``6 am`` are both system entities. Therefore, in order to distinguish between the two entities, we can use roles to annotate ``old_time`` for ``7 am`` and ``new_time`` for ``6 am``. We annotate the example as ``change alarm from {7 am|sys_time|old_time} to {6 am|sys_time|new_time}`` with ``old_time`` and ``new_time`` as roles. This way, we can distinguish each entity based on their roles.

For more information on the usage of role, check :doc:`Role <../userguide/role_classifier>`.

4. Dialogue States
^^^^^^^^^^^^^^^^^^

Dialogue state logic can get arbitrarily complex. Simple handlers can just return a canned text response while sophisticated handlers can make 3rd party calls, calculate state transitions and return complex responses. For handling intents in the Dialogue Manager, Workbench provides a helpful programming construct for consolidating duplicated dialogue state logic. In E-mart example, we can define a dialogue state for every intent. Workbench3 also supports defining a single dialogue state for multiple intents. In this section we will explore both options in detail.

Let's take a closer look at these intents for controlling doors: ``close_door``, ``open_door``, ``lock_door``, and ``unlock_door``. Now we can define a dialogue state for each of these intents.

.. code:: python

  @app.handle(intent='close_door')
  def close_door(context, slots, responder):

      ...

  @app.handle(intent='open_door')
  def open_door(context, slots, responder):

      ...

  @app.handle(intent='lock_door')
  def lock_door(context, slots, responder):

      ...

  @app.handle(intent='unlock_door')
  def unlock_door(context, slots, responder):

      ...

However, since close/open/lock/unlock door are very similar to each other in the controller logic (for example, setting the state variable for the door), we can handle all of these intents in the one state ``handle_door``.

.. code:: python

  @app.handle(intent='close_door')
  @app.handle(intent='open_door')
  @app.handle(intent='lock_door')
  @app.handle(intent='unlock_door')
  def handle_door(context, slots, responder):

      ...

Which approach to take depends on the exact application and it takes some trial and error to figure this out. The home assistant blueprint uses both patterns - check it out!

Another conversational pattern that would be useful to the reader is the follow-up request pattern. Take a look at the following interaction:

.. code:: bash

  User: Turn on the lights.
  App: Sure. Which lights?
  User: In the kitchen

In this pattern, the first request does not specify the required information, in this case the location of the light. Therefore, the application has to prompt the user for the missing information in the second request. To implement this, we define the ``specify_location`` intent and define the ``specify_location`` state. Since a number of states (``close/open door``, ``lock/unlock door``, ``turn on/off lights``, ``turn on/off appliance``, ``check door/light``) can lead to the ``specify location`` state, we need to pass in the previous state/action information in the request context as ``context['frame']['desired_action']``.

We include a code snippet for ``specify_location`` for your reference.

.. code:: python

  @app.handle(intent='specify_location')
  def specify_location(context, slots, responder):
  selected_all = False
  selected_location = _get_location(context)

  if selected_location:
      try:
          if context['frame']['desired_action'] == 'Close Door':
              reply = self._handle_door_open_close_reply(
                  selected_all, selected_location, context, desired_state="closed")
          elif context['frame']['desired_action'] == 'Open Door':
              reply = self._handle_door_open_close_reply(
                  selected_all, selected_location, context, desired_state="opened")
          elif context['frame']['desired_action'] == 'Lock Door':
              reply = self._handle_door_lock_unlock_reply(
                  selected_all, selected_location, context, desired_state="locked")
          elif context['frame']['desired_action'] == 'Unlock Door':
              reply = self._handle_door_lock_unlock_reply(
                  selected_all, selected_location, context, desired_state="unlocked")
          elif context['frame']['desired_action'] == 'Check Door':
              reply = self._handle_check_door_reply(selected_location, context)
          elif context['frame']['desired_action'] == 'Turn On Lights':
              reply = self._handle_lights_reply(
                  selected_all, selected_location, context, desired_state="on")
          elif context['frame']['desired_action'] == 'Turn Off Lights':
              reply = self._handle_lights_reply(
                  selected_all, selected_location, context, desired_state="off")
          elif context['frame']['desired_action'] == 'Check Lights':
              reply = self._handle_check_lights_reply(selected_location, context)
          elif context['frame']['desired_action'] == 'Turn On Appliance':
              selected_appliance = context['frame']['appliance']
              reply = self._handle_appliance_reply(
                  selected_location, selected_appliance, desired_state="on")
          elif context['frame']['desired_action'] == 'Turn Off Appliance':
              selected_appliance = context['frame']['appliance']
              reply = self._handle_appliance_reply(
                  selected_location, selected_appliance, desired_state="off")

          del context['frame']['desired_action']

      except KeyError:
          reply = "Please specify an action to go along with that location."

      responder.reply(reply)
  else:
      prompt = "I'm sorry, I wasn't able to recognize that location, could you try again?"
      responder.prompt(prompt)


Here is the full list of states in the home assistant blueprint:

   - greet
   - exit
   - check_weather
   - specify_location
   - specify_time
   - check_door
   - close_door
   - open_door
   - lock_door
   - unlock_door
   - turn_appliance_on
   - turn_appliance_off
   - check_lights
   - turn_lights_on
   - turn_lights_off
   - check_thermostat
   - set_thermostat
   - change_thermostat
   - turn_thermostat
   - change_alarm
   - check_alarm
   - remove_alarm
   - set_alarm
   - start_timer
   - stop_timer
   - unknown


5. Knowledge Base
^^^^^^^^^^^^^^^^^

The home assistant is a straight forward command-and-control house application, and therefore it does not have a catalog of items and does not use a knowledge base. Workbench3 does need an Elasticsearch connection for validation, and therefore we still need a local instance of Elasticsearch running in the background. If you have homebrew, you can set one up quickly:

.. code:: bash

   >>> brew install elasticsearch
   >>> elasticsearch


6. Training Data
^^^^^^^^^^^^^^^^

The labeled data for training our NLP pipeline was created using a combination of in-house data generation and crowdsourcing techniques. This is a highly important multi-step process that is described in more detail in :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` of the Step-By-Step Guide. But briefly, it requires at least the following data generation tasks:

+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Purpose                                                      | Question posed to data annotators                                                                                       |
+==============================================================+=========================================================================================================================+
| Exploratory data generation for guiding the app design       | "How would you talk to a conversational app to control your smart home appliances?"                                     |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Divide your application use case into separate domains       | If your application has to control appliances in a smart home, check the weather and control a smart alarm, divide these|
|                                                              | use cases into separate domains: smart_home, times_and_dates, weather. One way to break an application into smaller     |
|                                                              | domains is by clustering the queries by similar use case and then naming each cluster as a domain                       |
+==============================================================+=========================================================================================================================+
| Targeted query generation for training Domain and Intent     | For domain ``times_and_dates``, the following intents are constructed:                                                  |
| Classifiers.                                                 | ``change_alarm``: "What would you say to the app to change your alarm time from a previous set time to a new set time?" |
|                                                              | ``set_alarm``: "What would you say to the app to set a new alarm time?"                                                 |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Targeted query annotation for training the Entity Recognizer | ``set_alarm``: "Annotate all occurrences of sys_time and sys_interval system entities in the given query."              |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Targeted query annotation for training the Role Classifier   | ``set_alarm``: "Annotate all entities with their corresponding roles, when needed. For eg: old_time, new_time"          |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+
| Targeted synonym generation for gazetteer generation to      | ``city`` entity: "Enumerate a list of names of cities"                                                                  |
| improve entity recognition accuracies                        |                                                                                                                         |
|                                                              | ``location`` entity: "What are some names of locations in your home"                                                    |
+--------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------+

The training data for intent classification and entity recognition can be found in the ``domains`` directory, whereas the data for entity resolution is in the ``entities`` directory, both located at the root level of the blueprint folder.

.. admonition:: Exercise

   - Read :doc:`Step 6 <../quickstart/06_generate_representative_training_data>` of the Step-By-Step Guide for best practices around training data generation and annotation for conversational apps. Following those principles, create additional labeled data for all the intents in this blueprint and use them as held-out validation data for evaluating your app. You can read more about :doc:`NLP model evaluation and error analysis <../userguide/nlp>` in the user guide.

   - To train NLP models for your own home assistant application, you can start by reusing the blueprint data for generic intents like ``greet`` and ``exit``. However, for core intents like ``check_weather`` in the ``weather`` domain, it's recommended that you collect new training data that is tailored towards the entities (``city``, ``sys_time``) that your application needs to support. Follow the same approach to gather new training data for the ``check_weather`` intent or any additional intents and entities needed for your application.


7. Training the NLP Classifiers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To put the training data to use and train a baseline NLP system for your applicatio using Workbench's default machine learning settings, use the :meth:``build()`` method of the :class:``NaturalLanguageProcessor`` class:

.. code:: python

   >>> from mmworkbench.components.nlp import NaturalLanguageProcessor
   >>> import mmworkbench as wb
   >>> wb.configure_logs()
   >>> nlp = NaturalLanguageProcessor(app_path='home_assistant')
   >>> nlp.build()
   Fitting domain classifier
   Loading queries from file weather/check_weather/train.txt
   Loading queries from file times_and_dates/remove_alarm/train.txt
   Loading queries from file times_and_dates/start_timer/train.txt
   Loading queries from file times_and_dates/change_alarm/train.txt
   .
   .
   .
   Fitting intent classifier: domain='greeting'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 99.31%, params: {'fit_intercept': False, 'C': 1, 'class_weight': {0: 1.5304182509505702, 1: 0.88306789606035196}}
   Fitting entity recognizer: domain='greeting', intent='exit'
   No entity model configuration set. Using default.
   Fitting entity recognizer: domain='greeting', intent='greet'
   No entity model configuration set. Using default.
   Fitting entity recognizer: domain='unknown', intent='unknown'
   No entity model configuration set. Using default.
   Fitting intent classifier: domain='smart_home'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 98.43%, params: {'fit_intercept': True, 'C': 100, 'class_weight': {0: 0.99365079365079367, 1: 1.5915662650602409, 2: 1.3434782608695652, 3: 1.5222222222222221, 4: 0.91637426900584784, 5: 0.74743589743589745, 6: 1.9758620689655173, 7: 1.4254901960784312, 8: 1.0794871794871794, 9: 1.0645320197044335, 10: 1.1043715846994535, 11: 1.2563909774436088, 12: 1.3016260162601625, 13: 1.0775510204081633, 14: 1.8384615384615384}}
.. tip::

  During active development, it is helpful to increase the :doc:`Workbench logging level <../userguide/getting_started>` to better understand what is happening behind the scenes. All code snippets here assume that logging level has been set to verbose (``wb.configure_logs()``).

You should see a cross validation accuracy of around 98% for the :doc:`Intent Classifier <../userguide/intent_classifier>` for the domain ``smart_home`` and about 99% for the :doc:`Entity Recognizer <../userguide/entity_recognizer>` for the domain ``smart_home`` and intent ``turn_on_thermostat``. To see how the trained NLP pipeline performs on a test query, use the :meth:``process()`` method.

.. code:: python

   >>> nlp.process("please set my alarm to 8am for tomorrow")
   {'domain': 'times_and_dates',
    'entities': [{'confidence': -0.0,
      'role': None,
      'span': {'end': 38, 'start': 31},
      'text': 'tomorrow',
      'type': 'sys_time',
      'value': [{'grain': 'day', 'value': '2017-07-08T00:00:00.000-07:00'}]}],
    'intent': 'set_alarm',
    'text': 'please set my alarm to 8am for tomorrow'
    }

For the data distributed with this blueprint, the baseline performance is already high. However, when extending the blueprint with your own custom home assistant data, you may find that the default settings may not be optimal and you can get better accuracy by individually optimizing each of the NLP components.

Home assistant application consists of five domains and more than twenty intents so we need to do a fair bit of fine tuning of the classifiers.

A good place to start is by inspecting the baseline configuration used by the different classifiers. The user guide lists and describes all of the available configuration options in detail. As an example, the code below shows how to access the model and feature extraction settings for the Intent Classifier.

.. code:: python

   >>> ic = nlp.domains['smart_home'].intent_classifier
   >>> ic.config.model_settings['classifier_type']
   'logreg'
   >>> ic.config.features
   {'bag-of-words': {'lengths': [1, 2]},
    'edge-ngrams': {'lengths': [1, 2]},
    'exact': {'scaling': 10},
    'freq': {'bins': 5},
    'gaz-freq': {},
    'in-gaz': {}
   }

You can experiment with different learning algorithms (model types), features, hyperparameters, and cross-validation settings, by passing the appropriate parameters to the classifier's :meth:``fit()`` method. Here are a couple of examples.

For example, we can hange the feature extraction settings to use bag of bigrams in addition to the default bag of words:

.. code:: python

   >>> features = {
   ...             'bag-of-words': {'lengths': [1, 2]},
   ...             'freq': {'bins': 5},
   ...             'in-gaz': {},
   ...             'length': {}
   ...            }
   >>> ic.fit(features=features)
   Fitting intent classifier: domain='smart_home'
   Selecting hyperparameters using k-fold cross validation with 5 splits
   Best accuracy: 98.46%, params: {'fit_intercept': False, 'C': 10, 'class_weight': {0: 0.98518518518518516, 1: 2.3803212851405622, 2: 1.801449275362319, 3: 2.2185185185185183, 4: 0.80487329434697852, 5: 0.41068376068376072, 6: 3.2770114942528741, 7: 1.9928104575163397, 8: 1.1854700854700853, 9: 1.1505747126436781, 10: 1.2435336976320581, 11: 1.5982456140350876, 12: 1.7037940379403793, 13: 1.180952380952381, 14: 2.9564102564102566}}

In another example, we can change the model for the intent classifier to Support Vector Machine (SVM) classifier, which can work well in some dataset:

.. code:: python

   >>> search_grid = {
   ...    'C': [0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000],
   ...    'kernel': ['linear', 'rbf', 'poly'],
   ... }
   ...
   >>> param_selection_settings = {
   ...      'grid': search_grid,
   ...      'type': 'k-fold',
   ...      'k': 10
   ... }
   ...
   >>> ic = nlp.domains['smart_home'].intent_classifier
   >>> ic.fit(model_settings={'classifier_type': 'svm'}, param_selection=param_selection_settings)
   Fitting intent classifier: domain='smart_home'
   Loading queries from file smart_home/check_lights/train.txt
   Loading queries from file smart_home/specify_location/train.txt
   Loading queries from file smart_home/turn_appliance_off/train.txt
   Loading queries from file smart_home/check_thermostat/train.txt
   Loading queries from file smart_home/set_thermostat/train.txt
   Loading queries from file smart_home/turn_up_thermostat/train.txt
   Loading queries from file smart_home/turn_lights_on/train.txt
   Loading queries from file smart_home/unlock_door/train.txt
   Loading queries from file smart_home/turn_on_thermostat/train.txt
   Loading queries from file smart_home/lock_door/train.txt
   Loading queries from file smart_home/turn_down_thermostat/train.txt
   Unable to load query: Unable to resolve system entity of type 'sys_time' for '12pm'.
   Loading queries from file smart_home/close_door/train.txt
   Loading queries from file smart_home/turn_lights_off/train.txt
   Loading queries from file smart_home/open_door/train.txt
   Loading queries from file smart_home/turn_off_thermostat/train.txt
   Loading queries from file smart_home/turn_appliance_on/train.txt
   Selecting hyperparameters using k-fold cross-validation with 10 splits
   Best accuracy: 98.27%, params: {'C': 5000, 'kernel': 'rbf'}

Similar options are available for inspecting and experimenting with the Entity Recognizer and other NLP classifiers as well. Finding the optimal machine learning settings is an iterative process involving several rounds of parameter tuning, testing, and error analysis. Refer to the :doc:`Intent Classifier <../userguide/intent_classifier>` in the user guide for a detailed discussion on training, tuning, and evaluating the various Workbench classifiers.

The home assistant application also has role classifiers to distinguish between different role labels. For example, the annotated data in the ``times_and_dates`` domain and ``check_alarm`` intent have two types of roles: ``old_time`` and ``new_time``. We use the role classifier to correctly classify these roles for the ``sys_time`` entity:

.. code:: python

   >>> nlp.domains["times_and_dates"].intents["change_alarm"].load()
   >>> nlp.domains["times_and_dates"].intents["change_alarm"].entities["sys_time"].role_classifier.fit()
   >>> nlp.domains["times_and_dates"].intents["change_alarm"].entities["sys_time"].role_classifier.evaluate()
   <StandardModelEvaluation score: 100.00%, 15 of 15 examples correct>

In the above case, the role classifier was able to correctly distinguish between ``new_time`` and ``old_time`` for all test cases.

The application configuration file, ``config.py``, at the top level of home assistant folder contains custom intent and domain classifier model configurations that are namespaced by ``DOMAIN_MODEL_CONFIG and INTENT_MODEL_CONFIG`` respectively; other namespaces include ``ENTITY_MODEL_CONFIG and ROLE_MODEL_CONFIG``. If no custom model configuration is added to ``config.py`` file, Workbench will use its default classifier configurations for training and evaluation. Here is an example of an intent configuration:

.. code:: python

   INTENT_MODEL_CONFIG = {
       'model_type': 'text',
       'model_settings': {
           'classifier_type': 'logreg'
       },
       'param_selection': {
           'type': 'k-fold',
           'k': 5,
           'grid': {
               'fit_intercept': [True, False],
               'C': [0.01, 1, 10, 100],
               'class_bias': [0.7, 0.3, 0]
           }
       },
       'features': {
           "bag-of-words": {
               "lengths": [1, 2]
           },
           "edge-ngrams": {"lengths": [1, 2]},
           "in-gaz": {},
           "exact": {"scaling": 10},
           "gaz-freq": {},
           "freq": {"bins": 5}
       }
   }

.. admonition:: Exercise

   Experiment with different models, features, and hyperparameter selection settings to see how they affect the classifier performance. It is helpful to have a held-out validation set to evaluate your trained NLP models and analyze the misclassified test instances. You could then use observations from the error analysis to inform your machine learning experimentation. For more examples and discussion on this topic, refer to the :doc:`user guide <../userguide/nlp>`.


8. Parser Configuration
^^^^^^^^^^^^^^^^^^^^^^^

The queries in the home assistant do not have complex relationships between entities. For example, for the annotated query ``is the {back|location} door closed or open``, there is no entity that describes the ``location`` entity. As queries become more complex, for example, ``is the {green|color} {back|location} door closed or open``, we would need to relate the ``color`` entity with the ``location`` entity. When this happens, we call these two related entities ``entity groups``.
Since we do not have entity groups in the home assistant application, we therefore do not need a parser configuration, which is a component that helps group entities together. As the applications evolves, such entity relationships will form. Please refer to :doc:`Entity Groups <../userguide/language_parsing.html?highlight=entity%20groups>` and :doc:`Language Parser <../userguide/parser>` to read more about entity groups and parser configurations.


9. Using the Question Answerer
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :doc:`Question Answerer <../userguide/kb>` component in Workbench is mainly used within dialogue state handlers for retrieving information from the knowledge base. Since the home assistant application does not use a knowledge base, a question answerer component is not needed.


10. Testing and Deployment
^^^^^^^^^^^^^^^^^^^^^^^^^^

Once all the individual pieces (NLP, Dialogue State Handlers) have been trained, configured or implemented, you can do an end-to-end test of your conversational application using the :class:``Conversation`` class in Workbench.

.. code:: python

   >>> from mmworkbench.components.dialogue import Conversation
   >>> conv = Conversation(nlp=nlp, app_path='home_assistant')
   >>> conv.say('set alarm for 6am')
   ['Ok, I have set your alarm for 06:00:00.']

The :meth:``say()`` method packages the input text in a :doc:`user request <../userguide/interface>` object and passes it to the Workbench :doc:`Application Manager <../userguide/application_manager>` to a simulate an external user interaction with the application. It then outputs the textual part of the response sent by the application's dialogue manager. In the above example, we requested to set an alarm for 6am and the app responded, as expected, with a confirmation prompt of setting the alarm.

You can also try out multi-turn dialogues:

.. code:: python

   >>> conv.say('Hi there!')
   ['Hi, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances.]
   >>> conv.say("close the front door")
   ['Ok. The front door has been closed.']
   >>> conv.say("set the thermostat for 60 degrees")
   ['The thermostat temperature in the home is now 60 degrees F.']
   >>> conv.say("decrease the thermostat by 5 degrees")
   ['The thermostat temperature in the home is now 55 degrees F.']
   >>> conv.say("open the front door")
   ['Ok. The front door has been opened.']
   >>> conv.say("Thank you!")
   ['Bye!']


We can also enter the conversation mode directly from the commandline.

.. code:: bash

   >>> python app.py converse

   App: Hi, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances.
   You: What's the weather today in San Francisco?
   App: The weather forecast in San Francisco is clouds with a min of 62.6 F and a max of 89.6 F

Exercise: test the app and play around with different language patterns to figure out the edge cases that our classifiers are not able to handle. The more language patterns we can collect in our training data, the better our classifiers can handle in live usage with real users. Good luck and have fun - now you have your very own Jarvis!
