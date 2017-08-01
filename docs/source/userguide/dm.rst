.. meta::
    :scope: private

Working with the Dialogue Manager
=================================

The Dialogue Manager is the component responsible for managing the conversational aspect of your application. It uses pattern based rules to determine the dialogue state for each incoming request and implements handlers which execute business logic and return a natural language response to the user. Developing a dialogue manager for all but the simplest conversational apps can be a daunting task. Workbench mitigates the challenge by providing a pattern matching system and helpers for generating responses.

.. note::

  **Recommended prior reading:**

  - :ref:`Step 4 Define the Dialogue State Handlers <define_dialogue_state_handlers>` (Step-By-Step Guide)

The dialogue state is determined based on the request context and on the output from the natural language processor. When developing your application you will set up a system of rules to match requests to dialogue states using the a flexible syntax. Each dialogue state has a handler which contains logic to fulfill a user's request or gather more information if necessary and to generate a natural language response.

The primary dialogue manager concepts are request context, dialogue state rules, dialogue state handlers, and the responder.

Dialogue States
~~~~~~~~~~~~~~~

Each dialogue state represents a task the conversational agent can complete. It is usually named according to this task. Note that the naming is from the perspective of the conversational agent, as opposed to the user's intent, which is from the perspective of the user. Dialogue state rules and handlers are implemented in the application container (also known as the ``app.py``).

Dialogue State Rules
~~~~~~~~~~~~~~~~~~~~

Dialogue State Rules match requests to dialogue states based on the request context including the output of the Natural Language Processor. Each rule can have a single domain, a single intent and a set of entity  can match based on a set of entity types.

To better understand dialogue state rules, let's jump into the skeleton of a dialogue manager for an app that provides store hours.

.. code:: python

  from mmworkbench import Application

  app = Application(__name__)

  @app.handle(domain='stores', intent='greet')
  def welcome(context, responder):
      pass

  @app.handle(domain='stores', intent='exit')
  def say_goodbye(context, responder):
      pass

  @app.handle(domain='stores', intent='get_store_hours')
  def prompt_for_store(context, responder):
      pass

  @app.handle(domain='stores', intent='get_store_hours', has_entity='store_name')
  def send_store_hours(context, responder):
      pass

  @app.handle(domain='stores', intent='unsupported')
  @app.handle(domain='unknown')
  def send_help(context, responder):
      pass

  @app.handle()
  def default(context, responder):
      pass

  if __name__ == '__main__':
      app.cli()


This dialogue manager has six dialogue states: ``welcome``, ``say_goodbye``, ``prompt_for_store``, ``send_store_hours``, ``send_help``, and ``default``. The handler for each state is defined by the function of the same name. More on dialogue state handlers later. The rules for each state are specified by decorating a dialogue state handler with :py:meth:`app.handle`. As you can see in the example above, we use ``domain``, ``intent``, and ``has_entity`` parameters to specify the details of a rule. You can also use ``has_entities`` to specify multiple entities. A request satisfies a rule when its NLP result has the same domain, and/or intent, if specified, and has entities of the specified types. A dialogue state can have multiple rules, and if any of them match, the dialogue.

Tie Breaking
^^^^^^^^^^^^

A request may satisfy multiple rules, but the dialogue manager will always resolve to exactly one dialogue state. When a single request satisfies multiple rules, the most specific rule is used. The *specificity* of a rule is based on it's parameters. A rule with no parameters, like that used for the ``default`` dialogue state in our example above is the least specific rule. A rule which has a domain has some specificity, a rule which specifies an intent is more specific, and rules which include entities are the still more specific. Using more entities will further increase a rule's specificity. If a request matches two requests with the same specificity, the rule which was specified earliest in an ``app.py`` will be used.

Dialogue State Handlers
~~~~~~~~~~~~~~~~~~~~~~~

Dialogue State Handlers are the functions which are invoked when a request matches one of the corresponding rules. Within a handler the developer can execute arbitrary code. Because different applications might

Dialogue State Handlers have two arguments: ``context`` and ``responder``.

``context``
^^^^^^^^^^^

The ``context`` object is a dictionary containing the contextual information needed to manage dialogues. You can use this information to fulfill user requests, or determine additional information needed from the user, as well as to fill slots in your natural language templates.

+----------------+-------------------------------------------------------------------------------+
| Key            | Value                                                                         |
+================+===============================================================================+
| ``'request'``  | a read-only dictionary containing the original user text and session details. |
+----------------+-------------------------------------------------------------------------------+
| ``'frame'``    | a dictionary which should be used to store information across dialogue turns, |
|                | not intended for use by front-end clients.                                    |
+----------------+-------------------------------------------------------------------------------+
| ``'domain'``   | the domain of the current message as classified by the natural                |
|                | language processor.                                                           |
+----------------+-------------------------------------------------------------------------------+
| ``'intent'``   | the intent of the current message as classified by the natural                |
|                | language processor.                                                           |
+----------------+-------------------------------------------------------------------------------+
| ``'entities'`` | the entities recognized in the current message by the natural                 |
|                | language processor.                                                           |
+----------------+-------------------------------------------------------------------------------+
| ``'history'``  | a list containing previous requests and responses in the                      |
|                | current conversation.                                                         |
+----------------+-------------------------------------------------------------------------------+

``responder``
^^^^^^^^^^^^^

The ``responder`` object is used to send responses to the user. ``responder`` allows you to use templated natural language responses, as well as additional metadata needed to fulfill the request on the client endpoint. The ``responder`` has methods which accept template strings and a ``slots`` attribute which can store values to fill in the templates.

+------------------------------+-----------------------------------------------------------------+
| Method                       | Description                                                     |
+------------------------------+-----------------------------------------------------------------+
| :py:meth:`responder.reply`   | Used to send a text or voice response and end the dialogue.     |
+------------------------------+-----------------------------------------------------------------+
| :py:meth:`responder.prompt`  | Used to send a text or voice response and wait for a            |
|                              | user response.                                                  |
+------------------------------+-----------------------------------------------------------------+
| :py:meth:`responder.respond` | Used to send an arbitrary client action object.                 |
+------------------------------+-----------------------------------------------------------------+

.. note::

   :py:meth:`responder.reply` and :py:meth:`responder.prompt` accept a single template, or a list of templates. If a list is provided, one item will be selected at random. This makes your conversational agent a little more varied and life-like.

Let's take a look at a basic example of a dialogue state handler for greeting a user.

.. code:: python

  @app.handle(intent='greet')
  def welcome(context, responder):
      try:
          responder.slots['name'] = context['request]['session']['name']
          templates = ['Hello {name}', 'Hey {name}!', '{name}, how are you?']
      except KeyError:
          # name was not included in request
          templates = ['Hello', 'Hey!', 'How are you?']
      responder.prompt(templates)

This handler attempts to use the user's name, retrieving it from the request session.

Examples
~~~~~~~~

Review the following documents for more examples of dialogue manager implementations.

 - :ref:`Step 4 <define_dialogue_state_handlers>` of the Step-By-Step Guide
 - :doc:`Food Ordering <../blueprints/food_ordering>` Blueprint
 - :doc:`Video Discovery <../blueprints/video_discovery>` Blueprint
 - :doc:`Home Assistant <../blueprints/home_assistant>` Blueprint





