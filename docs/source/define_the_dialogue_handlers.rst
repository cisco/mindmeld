Define the Dialogue State Handlers
==================================

Today's commercial voice and chat assistants guide users through a conversational interaction in order to find information or accomplish a task. The steps in each conversational interaction are called 'dialogue states'. A dialogue state defines the form of response which is appropriate for each step in an interaction as well as other logic that must be invoked to determine the desired response. Each application relies on a set of dialogue state handlers which define the logic and response required for every supported dialogue state. 

At the core of every conversational application resides a dialogue manager. The dialogue manager is responsible for analyzing each incoming request and assigning it to a specific dialogue state handler to execute required logic and return a response. The task of mapping each incoming request to the appropriate dialogue state is often referred to as 'dialogue state tracking'. While applying large-scale machine learning techiniques for dialogue state tracking is an active area of research today, nearly all commercial applications rely heavily on rule-based and pattern-matching approaches to map incoming requests to the correct dialogue state.

MindMeld Workbench provides advanced and flexible capabilities for dialogue state tracking. It offers a flexible syntax for defining the rules and patterns each incoming request must match in order to be assigned to a specific dialogue state. In addition, Workbench is fully extensible and can accommodate any custom logic to supplement MindMeld's built-in pattern matching capabilities.

Specify the Superset of Dialogue States
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before you can begin implementing your dialogue state handlers, it is first necessary to define the required dialogue states. For simple conversational interactions, the set of dialogue states can be very straightforward, as illustrated in the flow diagram below.

.. image:: images/simple_dialogue_states.png
    :width: 700px
    :align: center

In practice, however, the flow of dialogue states can be quite complex. Conversational applications are powerful since they provide few constraints on what a user can say during an interaction. This makes it easy for a user to shortcut to the specific functionality they may need. This also means that a user is free to change topics or otherwise throw a curve ball at your application at any point in the interaction without warning. Consequently, it is not uncommon for dialogue flows to be quite convoluted, as suggested below.

.. image:: images/complex_dialogue_states.png
    :width: 700px
    :align: center

Our goal here is to define the required dialogue states for the scripted conversational interaction in :ref:`section 3 <script-interactions>`. To capture the envisioned functionality, four different dialogue states will be needed: ``welcome``, ``send_store_hours``, ``send_nearest_store``, and ``say_goodbye``. The following diagram illustrates the conversation flow.

.. image:: images/quickstart_dialogue_states.png
    :width: 700px
    :align: center

As shown, each dialogue state prescribes the form of the system response. For most commercial applications today, the form of response consists of natural language templates to reply to the user or prompt for additional information. These templates are populated on-the-fly using contextual state information gleaned from the conversation.  Often, the response also includes additional information to render client-side interactive elements such as image carousels or quick reply buttons.

.. note::

  By convention, the dialogue state names should be verbs which describe the action your application should take at each point in the interaction.


Create the Application Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In MindMeld Workbench, the application container is a Python file which contains all of the logic and functionality for your application. This Python file is located in your project's root directory, and it enumerates all of the dialogue states and their associated handlers. To begin, create a file called 'my_app.py' with the following minimal implementation in your root directory. 

.. code:: python

  from mmworkbench import Application
  
  app = Application(__name__)
  
  @app.handle(intent='greet')
  def welcome():
      response = {
          'replies': [
              'Hello. I can help you find store hours ' +
              'for your local Kwik-E-Mart. How can I help?'
          ]
      }
      return response

Your directory structure should now resemble the following.

.. image:: images/directory2.png
    :width: 350px
    :align: center

The minimal code snippet shown above illustrates the conventions employed by Workbench to implement dialogue state tracking and dialogue state handling logic. It performs the following steps:

   1. It imports the Application class from the MindMeld Workbench package.
   2. It defines an Application instance to serve as the parent container for the application.
   3. It uses the :keyword:`@app.handle()` decorator to define a pattern which, when matched, will invoke the associated handler function.
   4. It specifies the handler function :keyword:`welcome()` which defines the ``welcome`` dialogue state and returns the desired response.

This application structure provides a straighforward mechanism to enumerate a variety of patterns along with their associated handlers which will comprise the core interaction logic for your application. 


Implement the Dialogue State Handlers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let us now define the dialogue handlers we would need for the interaction in :ref:`section 3 <script-interactions>`. In the process, we will introduce several new capabilities of Workbench which are described in depth later in the :ref:`User Guide <userguide>`.

To start, let's consider the handler for the ``welcome`` dialogue state.

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

As mentioned above, the name of the dialogue state is prescribed by the method name of the dialogue state handler, :keyword:`welcome()`. The :keyword:`@app.handle()` decorator specifies the pattern which must be matched to invoke the handler method. In this case, the pattern is specified simply as :keyword:`intent='greet'`. In other words, if the natural language processer predicts that the intent of the incoming request is ``greet``, the :keyword:`welcome()` handler will be invoked.

Every dialogue handler returns a :keyword:`response` object. This object specifies the natural language text as well as other data to be returned in the response. Note that the text strings contained in this response can utilize templated expressions, such as :keyword:`'Hello, {name}.'`. These templates rely on standard Python string formatting syntax. Templated expressions will be populated with real values before returning to the client. The :keyword:`slots` object is used to store the named string values which are used to populate the templates.

In the code snippet above, we also introduce the :keyword:`context` object. Workbench relies on the :keyword:`context` object to keep track of all of the state information associated with the current conversational interaction. In can contain output data from the natural language processing models, aggregated state from multiple previous interactions, as well as user and session information. The detailed information in the :keyword:`context` can be very useful for implementing custom dialogue state handling logic. More details can be found in the :ref:`User Guide <userguide>`.

Following this same approach, we can now also define handlers for the dialgue states ``send_store_hours``, ``send_nearest_store``, and ``say_goodbye``. The resulting my_app.py file now looks like the following.

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
      stores = qa.get(index='stores', sort='proximity', current_location=loc)
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
  
This code snippet introduces the QuestionAnswerer class. The QuestionAnswerer is the Workbench module responsible for creating and searching across a knowledge base of information relevant to your application. In this example, the ``send_nearest_store`` dialogue state relies on the QuestionAnswerer component to retrieve the closest retail store location from the knowledge base. The QuestionAnswerer and its associated knowledge base will be discussed in more detail below.

This simple example also illustrates the use of a default handler. The :keyword:`@app.handle()` decorator serves as a 'catchall' pattern which will return a default response if no other specified patterns are matched.

Now that we have our initial set of dialogue handlers in place, we can now proceed with building a knowledge base and training machine learning models to understand natural language requests.

