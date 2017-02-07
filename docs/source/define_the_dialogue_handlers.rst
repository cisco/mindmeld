Step 4: Define the Dialogue State Handlers
==========================================

Today's commercial voice and chat assistants guide users through a conversational interaction in order to find information or accomplish a task. Conversational interactions consist of steps called *dialogue states*. For each dialogue state, a particular form of response is appropriate, and, particular logic may be invoked to determine certain parts of the content of the response. A set of dialogue state handlers define the logic and response required for every dialogue state that a given application supports.

A *dialogue manager* is at the core of every conversational application. The dialogue manager analyzes each incoming request and assigns it to a dialogue state handler which then executes the required logic and returns a response. The task of mapping incoming requests to appropriate dialogue states is called *dialogue state tracking*. Applying large-scale machine learning techniques for dialogue state tracking is an active area of research today. For now, however, nearly all commercial applications rely heavily on rule-based and pattern-matching approaches to accomplish dialogue state tracking.

For most use cases, the procedures described in this section suffice to configure the dialogue manager, which you need not deal with directly.

MindMeld Workbench provides advanced capabilities for dialogue state tracking, beginning with a flexible syntax for defining rules and patterns for mapping requests to dialogue states. And because Workbench is fully extensible, you can supplement MindMeld's built-in pattern matching capabilities with whatever custom logic you need.

Specify the Superset of Dialogue States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you can begin to implement dialogue state handlers, you must first define the dialogue states your application requires. For simple conversational interactions, the set of dialogue states can be straightforward, as illustrated in the flow diagram below.

.. image:: images/simple_dialogue_states.png
    :width: 700px
    :align: center

The power of conversational applications lies in providing minimal constraints on what a user can say during an interaction. Users can shortcut directly to the functionality they want. They are also free to change topics or otherwise throw a curve ball at your application at any point in the interaction without warning. All of this means that in practice, dialogue flows can become quite convoluted, as suggested below.

.. image:: images/complex_dialogue_states.png
    :width: 700px
    :align: center

For our example, we need to define the dialogue states that the scripted conversational interaction in :doc:`Step 2 <script_interactions>` requires. To capture the functionality we envision, we need four different dialogue states: ``welcome``, ``send_store_hours``, ``send_nearest_store``, and ``say_goodbye``, as shown in the diagram below.

.. image:: images/quickstart_dialogue_states.png
    :width: 700px
    :align: center

As the diagram illustrates, each dialogue state prescribes a natural language template that defines the form of the system response, and the template is populated on-the-fly using contextual state information gleaned from the conversation. The filled-in template represents an appropriate reply to, or a prompt for more information from, the user. The response may also include additional information to render client-side interactive elements such as image carousels or quick reply buttons.

.. note::

  By convention, dialogue state names are verbs that describe the action your application should take at particular points in the interaction.


Create the Application Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In MindMeld Workbench, the *application container* is a Python file which contains all of the logic and functionality for your application. This Python file is located in your project's root directory, and it enumerates all of the dialogue states and their associated handlers. If you based your application structure on a blueprint, you will see a file `my_app.py` in the root directory. If not, create a Python file called `my_app.py` with the following minimal implementation in your root directory.

.. code:: python

  from mmworkbench import Application

  app = Application(__name__)

  @app.handle(intent='greet')
  def welcome():
      response = {
          'replies': [
          ]
      }
      return response

Your directory structure should now resemble the following.

.. image:: images/directory2.png
    :width: 350px
    :align: center

The above code snippet illustrates the conventions for implementing dialogue state tracking and dialogue state handling logic in Workbench. The code is written to perform four steps:

   1. Import the Application class from the MindMeld Workbench package.
   2. Define an Application instance to serve as the parent container for the application.
   3. Using the :keyword:`@app.handle()` decorator, define a pattern which, when matched, invokes the associated handler function.
   4. Specify the handler function :keyword:`welcome()` to define the ``welcome`` dialogue state and return the desired response. We decided that ``welcome`` would be one of our dialogue states based on the scripting exercise in :doc:`Step 2 <script_interactions>`. For now, we are leaving the response empty.

The patterns and associated handlers which you enumerate using this straighforward application structure will comprise the core interaction logic for your application.

Implement the Dialogue State Handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We have already defined the dialogue handlers that the interaction in :doc:`Step 2 <script_interactions>` requires.

Now, to finish implementing the dialogue handlers, we need to add the desired response for each dialogue state. As we do so, we will learn about capabilities of Workbench which are explained further in the :ref:`User Manual <userguide>`.

First, consider the handler for the ``welcome`` dialogue state.

.. code:: python

  from mmworkbench import Application, context, slots

  app = Application(__name__)

  @app.handle(intent='greet')
  def welcome():
      slots['name'] = context.request.session.user_name
      response = {
          'replies': [
              'Hello, {name}. I can help you find store hours ' +
              'for your local Kwik-E-Mart. How can I help?'
          ]
      }
      return response

Following convention, we use the dialogue state name, ``welcome``, as the method name of the dialogue state handler, :keyword:`welcome()`.

The :keyword:`@app.handle()` decorator specifies the pattern which must be matched to invoke the handler method. Here, the pattern specified is simply :keyword:`intent='greet'`. In other words, if the natural language processor predicts that the intent of the incoming request is ``greet``, the :keyword:`welcome()` handler is invoked.

Every dialogue handler returns a :keyword:`response` object that specifies the natural language text and any other data to be returned in the response. Text strings contained in this response can use templated expressions in standard Python string formatting syntax, like :keyword:`'Hello, {name}.'` in our example. Templated expressions are populated with real values before the response is returned to the client. Workbench uses the :keyword:`slots` object to store the named string values which populate the templates.

The code snippet also introduces the :keyword:`context` object, which Workbench uses to keep track of all of the state information associated with the conversational interaction as it progresses. This state information can include output data from the natural language processing models, aggregated state from multiple previous interactions, and user and session information. The contents of the :keyword:`context` can be very useful for implementing custom dialogue state handling logic. See the :ref:`User Manual <userguide>` for details.

Let's follow this same approach to define handlers for the dialogue states ``send_store_hours``, ``send_nearest_store``, and ``say_goodbye``. The resulting `my_app.py` file looks like the following.

.. code:: python

  from mmworkbench import Application, QuestionAnswerer, context, slots

  qa = QuestionAnswerer()
  app = Application(__name__, qa)

  @app.handle(intent='greet')
  def welcome():
      slots['name'] = context.request.session.user_name
      response = {
          'replies': [
              'Hello, {name}. I can help you find store hours ' +
              'for your local Kwik-E-Mart. How can I help?'
          ]
      }
      return response

  @app.handle(intent='get_store_hours')
  def send_store_hours():
      set_target_store(context)
      if context.frame.target_store:
          slots['open_time'] = context.frame.target_store['open_time']
          slots['close_time'] = context.frame.target_store['close_time']
          slots['store_name'] = context.frame.target_store['name']
          dates = [e.value for e in context.entities if e.type == 'date']
          if dates: slots['date'] = dates[0]
          response = {
              'replies': [
                  'The {store_name} Kwik-E-Mart opens at {open_time} and closes at {close_time} {date}.'
              ]
          }
      else:
          response = {'replies': ['For which store?']}
      return response

  @app.handle(intent='get_nearest_store')
  def send_nearest_store():
      loc = context.request.session.location
      stores = qa.get(index='stores', sort='location', current_location=loc)
      slots['store_name'] = stores[0]['name']
      response = {
          'replies': [
              'Your nearest Kwik-E-Mart is located at {store_name}.'
          ]
      }
      return response

  @app.handle(intent='exit')
  def say_goodbye():
      return {'replies': ['Bye', 'Goodbye', 'Have a nice day.']}

  @app.handle()
  def default():
      return {
          'replies': [
              'I did not understand. Please you rephrase your request.'
          ]
      }

  def set_target_store(context):
      stores = [e.value for e in context.entities if e.type == 'store_name']
      if stores: context.frame.target_store = stores[0]

This code snippet introduces the `QuestionAnswerer` class. In Workbench, `QuestionAnswerer` is the module that creates and searches across a knowledge base of information relevant to your application. In this example, the ``send_nearest_store`` dialogue state relies on the `QuestionAnswerer` component to retrieve the closest retail store location from the knowledge base. The `QuestionAnswerer` is discussed further in the next section.

The snippet also demonstrates the use of a default handler. The :keyword:`@app.handle()` decorator serves as a 'catchall' pattern that returns a default response if no other specified patterns are matched.

Now that our initial set of dialogue handlers are in place, we can begin building a knowledge base and training machine learning models to understand natural language requests.

