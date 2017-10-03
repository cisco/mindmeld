Working with the Dialogue Manager
=================================

The Dialogue Manager

 - manages the conversational aspect of a Workbench application
 - uses pattern-based rules to determine the dialogue state for each incoming request
 - implements handlers which execute business logic and return a natural language response to the user

Developing a dialogue manager can be a daunting task for all but the simplest conversational apps. Workbench mitigates the challenge by providing a pattern-matching system and helpers for generating responses.

.. note::

    This is an in-depth tutorial to work through from start to finish. Before you begin, read the :ref:`Step-by-Step Guide <quickstart>`, paying special attention to the section about defining the :ref:`Dialogue State Handlers <define_dialogue_state_handlers>`.

Let's explore the main concepts you'll apply when you develop a dialogue manager in Workbench: *dialogue states*, *dialogue state rules*, and *dialogue state handlers*.

Dialogue States
---------------

For every incoming request, the dialogue manager (DM) determines which dialogue state applies. When developing your application, you set up a system of rules to match requests to dialogue states, using a flexible syntax that Workbench provides. Each dialogue state represents a task that the conversational agent can complete. Usually, you name the dialogue state for the task it represents, for example 'welcome' or 'send_store_hours.' Name the dialogue states from the perspective of the conversational agent. (By contrast, you name intents from the user's perspective).

Each dialogue state has a handler which contains logic to fulfill a user's request or gather more information if necessary, and to generate a natural language response. Dialogue state rules and handlers are implemented in ``app.py``, the :ref:`application container <app_container>`.

Dialogue State Rules
--------------------

Dialogue state rules match requests to dialogue states based on the request context, including the output from the natural language processor. Each rule specifies what to match, in the following form: exactly one domain, one intent, and a set of entity types. In single-domain apps the domain is omitted; in all apps, the entity types are optional. The exception to all this is the `default` state, which must match all requests, and therefore has neither domain, intent, nor entity types.

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


This dialogue manager has six dialogue states: ``welcome``, ``say_goodbye``, ``prompt_for_store``, ``send_store_hours``, ``send_help``, and ``default``.

In each state:

 - There is a function whose name is the name of the state. This function defines the handler. (In the example, ``pass`` substitutes for detailed definitions, since we explain handlers in the next section.)

 - Rules are specified by decorating the handler with the :py:meth:`app.handle` method, whose parameters can include ``domain``, ``intent``, and ``has_entity``. To specify multiple entities, we would use ``has_entities``.

When the NLP result of a request and a dialogue state rule have the same combination of domain, intent, and entity types, then the request satisfies (matches) the rule. A dialogue state can have multiple rules, and if any of them match the request, the dialogue handler responds.

Tie Breaking
^^^^^^^^^^^^

The DM always resolves to exactly one dialogue state.

Rules are considered more or less *specific* according to what parameters they have:

 - The least specific rule is one (like ``default`` in the example above) with no parameters
 - A rule with a domain has some specificity
 - A rule with an intent is more specific
 - A rule with entities is still more specific
 - A rule with *the most* entities is the most specific

When a single request satisfies multiple rules, the DM chooses the most specific rule. If a request matches two requests with the same specificity, the DM chooses the rule that appears earliest in ``app.py``.

Dialogue State Handlers
-----------------------

Dialogue state handlers are the functions invoked when a request matches a rule for the handler's corresponding dialogue state. Workbench places no restrictions on the code within a handler. This is important because requirements differ for different applications, and developers must have the flexibility to organize code as they wish.

Dialogue state handlers take two arguments: ``context`` and ``responder``.

``context``
^^^^^^^^^^^

The ``context`` object is a dictionary containing the contextual information needed to manage dialogues. You can use this information to fulfill user requests, determine additional information needed from the user, or to fill slots in your natural language templates.

+----------------+-------------------------------------------------------------------------------+
| Key            | Value                                                                         |
+================+===============================================================================+
| ``'request'``  | Dictionary containing the original user text and session details (read-only)  |
+----------------+-------------------------------------------------------------------------------+
| ``'frame'``    | Dictionary for storing information across dialogue turns                      |
|                | (not for use by front-end clients)                                            |
+----------------+-------------------------------------------------------------------------------+
| ``'domain'``   | Domain of the current message as classified by the natural                    |
|                | language processor                                                            |
+----------------+-------------------------------------------------------------------------------+
| ``'intent'``   | Intent of the current message as classified by the natural                    |
|                | language processor                                                            |
+----------------+-------------------------------------------------------------------------------+
| ``'entities'`` | Entities in the current message, as recognized by the natural                 |
|                | language processor                                                            |
+----------------+-------------------------------------------------------------------------------+
| ``'history'``  | List of previous requests and responses in the                                |
|                | current conversation                                                          |
+----------------+-------------------------------------------------------------------------------+

``responder``
^^^^^^^^^^^^^

Use the ``responder`` object to send responses to the user. You can use templated natural language responses, as well as metadata needed to fulfill the request on the client endpoint. The ``responder`` has methods which accept template strings, and a ``slots`` attribute to store values with which to fill in the templates.

+-------------------------------+----------------------------------------------------------------+
| Method                        | Description                                                    |
+===============================+================================================================+
| :py:meth:`responder.reply`    | Used to send a text view directive                             |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.speak`    | Used to send a voice action directive                          |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.suggest`  | Used to send a suggestions view directive                      |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.list`     | Used to send a list view directive                             |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.listen`   | Used to send a directive to listen for user voice response     |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.reset`    | Used to send a reset action directive, explicitly ending the   |
|                               | conversation                                                   |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.display`  | Used to send a custom view directive                           |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.act`      | Used to send a custom action directive                         |
+-------------------------------+----------------------------------------------------------------+
| :py:meth:`responder.direct`   | Used to send an arbitrary directive object                     |
+-------------------------------+----------------------------------------------------------------+

.. note::

   :py:meth:`responder.reply` and :py:meth:`responder.speak` accept a single template, or a list of templates. If a list is provided, the DM selects one item at random. This makes your conversational agent a little more varied and life-like.

Consider a basic dialogue state handler that greets a user by name, retrieving the user's name from the request session.

.. code:: python

  @app.handle(intent='greet')
  def welcome(context, responder):
      try:
          responder.slots['name'] = context['request']['session']['name']
          templates = ['Hello {name}', 'Hey {name}!', '{name}, how are you?']
      except KeyError:
          # name was not included in request
          templates = ['Hello', 'Hey!', 'How are you?']
      responder.reply(templates)
      responder.listen()


Next Steps
----------

The concepts and techniques described up to this point are exactly what you will use in coding the dialogue handlers you defined (as directed in :ref:`Step 4 <define_dialogue_state_handlers>` of the Step-By-Step Guide). Before you begin, you may want to study how the dialogue managers are implemented in the Workbench blueprint apps:

 - :doc:`Food Ordering <../blueprints/food_ordering>`
 - :doc:`Video Discovery <../blueprints/video_discovery>`
 - :doc:`Home Assistant <../blueprints/home_assistant>`
