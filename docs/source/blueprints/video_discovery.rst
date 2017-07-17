Video Discovery
===============

In this step-by-step walkthrough, you'll build a conversational application that allows users to browse and find movies and TV shows from a huge content catalog, using the Workbench blueprint for this purpose.

1. The Use Case
^^^^^^^^^^^^^^^

Through a conversational interface, users should be able to browse through a content catalog containing movies and TV shows. They should be able to search using common filters for video content, such as: genre, cast, release year, etc... They should also be able to continue refining their search until they find the right content, and restart the search at any moment. The app should also support general actions such as greeting the user and handling basic humor.

2. Example Dialogue Interactions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The conversational flows for video discovery can be highly complex, depending on the desired app functionality and the amount of user guidance required at each step. Enumerating and finalizing all anticipated user interactions requires multiple iterations.

Here are some examples of scripted dialogue interactions for conversational flows.

.. image:: /images/video_dicovery_interactions.png
    :width: 700px
    :align: center

3. Domain-Intent-Entity Hierarchy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is the NLP model hierarchy for our video discovery application.

.. image:: /images/video_discovery_hierarchy.png
    :width: 700px
    :align: center


The 2 domains supported by this application are ``video_content`` and ``unrelated``:

  - ``video_content`` - Supports the functionality for browsing through the video catalog.
  - ``unrelated`` - Handles all the requests that are not related to the video catalog.

The ``video_content`` domain supports the following intents:

  - ``greet`` - User greets the app
  - ``help`` - User is confused or needs instructions on how to proceed
  - ``start_over`` — User wants to abandon current selections and restart the ordering process
  - ``exit`` - User wants to end the current conversation
  - ``browse`` - User is searching for content, either for a specific movie or TV show or for a set of results
  - ``unsupported`` - User is asking for information related to movies or TV shows, but the app does not support thoese questions. For example, asking when a movie was released, or when it will be on given channel

Similarly the ``unrelated`` domain supports the intents:

  - ``general`` - User asks general questions unrelated to video content. For example questions about weather, food, sports, etc...
  - ``compliment`` - User gives a compliment to the app
  - ``insult`` - User gives an insult to the app

For this app, only the ``browse`` intent requires entity recognition. This intent supports the following entity types:

   - ``cast`` — The name of an actor
   - ``country`` — The name of the country of origing of a movie or TV show
   - ``director`` — The name of a director
   - ``genre`` — The name of a video genre
   - ``sort`` — How the users want to sort the results: by most recent, most popular, etc...
   - ``title`` — The title of a video in the catalog
   - ``type`` — The type of video the user is looking for: `movie` or `TV show`

To train the different machine learning models in the NLP pipeline for this app, we need labeled training data that covers all our intents and entities. To download the data and code required to run this blueprint, run the command below in a directory of your choice.

.. code-block:: console

    $ python -c "import mmworkbench as wb; wb.blueprint('video_discovery');"

This should create a Workbench project folder called ``video_discovery`` in your current directory with the following structure:

<< ADD IMAGE HERE >>

4. Dialogue States
^^^^^^^^^^^^^^^^^^

To support the functionality we envision, our app needs one dialogue state for each intent, as shown in the table below.

+------------------+--------------------------+-------------------------------------------------+
| | Intent         | |  Dialogue State        | | Dialogue State                                |
| |                | |  name                  | | function                                      |
+==================+==========================+=================================================+
| | ``greet``      | | ``welcome``            | | Begin an interaction and welcome the user     |
+------------------+--------------------------+-------------------------------------------------+
| | ``browse``     | | ``show_content``       | | Show the user a set of results and            |
| |                | |                        | | refine them as the user provides more details |
+------------------+--------------------------+-------------------------------------------------+
| | ``start_over`` | | ``start_over``         | | Cancel the ongoing search                     | 
| |                | |                        | | and prompt the user for a new request         |
+------------------+--------------------------+-------------------------------------------------+
| | ``exit``       | | ``say_goodbye``        | | End the current interaction                   | 
+------------------+--------------------------+-------------------------------------------------+
| | ``help``       | | ``provide_help``       | | Provide help information                      | 
| |                | |                        | | in case the user gets stuck                   |
+------------------+--------------------------+-------------------------------------------------+
| | ``unsupported``| | ``handle_unsupported`` | | Inform user the app does not provide that     | 
| |                | |                        | | information and get them back to video search |
+------------------+--------------------------+-------------------------------------------------+
| | ``compliment`` | | ``say_something_nice`` | | Compliment the user back and promt the user   |
| |                | |                        | | to get back to food ordering                  | 
+------------------+--------------------------+-------------------------------------------------+
| | ``insult``     | | ``handle_insult``      | | Handle the insult and promt the user          | 
| |                | |                        | | to get back to food ordering                  |
+------------------+--------------------------+-------------------------------------------------+
| | others         | | ``default``            | | Prompt a user who has gone off-topic          | 
| |                | |                        | | to get back to food ordering                  |
+------------------+--------------------------+-------------------------------------------------+

All dialogue states and their associated handlers are defined in the ``app.py`` application container file at the top level of the blueprint folder.

Handler logic can be simple, complex, or in between. At one end of this spectrum, the handler simply returns a canned response, sometimes choosing randomly from a set of responses. A more sophisticated handler could execute knowledge base queries to fill in the slots of a partially-templatized response. And a handler that applies more complex business logic could call an external API, process what the API returns, and incorporate the result into the response template.

The handler logic is fairly straightforward for most of our dialogue states. The main actions are choosing from a set of pre-scripted natural language responses, and replying to the user. These simple states include ``welcome``, ``start_over``, ``say_goodbye``, ``provide_help``, ``handle_unsupported``, ``say_something_nice``, ``handle_insult`` and ``default``.

For example, here's the ``say_goodbye`` state handler, where we clear the :doc:`dialogue frame <../userguide/dm>` and use the :doc:`responder <../userguide/dm>` object to reply with one of our scripted "goodbye" responses:

.. code:: python

	@app.handle(intent='exit')
	def say_goodbye(context, slots, responder):
	    """
	    When the user ends a conversation, clear the dialogue frame and say goodbye.
	    """
	    context['frame'] = {}
	    goodbyes = ['Bye!', 'Goodbye!', 'Have a nice day.', 'See you later.']

	    responder.reply(goodbyes)

By contrast, the handler logic for the ``show_content`` dialogue state is more substantial, because it contains the core business logic for our application. In this dialogue state handler, we use the :doc:`Question Answerer <../userguide/kb>` and external API calls to process the transaction.

We can illustrate this with the general implementation of the ``show_content`` handler:

.. code:: python

	@app.handle(intent='browse')
	def show_content(context, slots, responder):
	    """
	    When the user looks for a movie or TV show, fetch the documents from the knowledge base
	    with all entities we have so far.
	    """
	    # Update the frame with the new entities extracted.
	    context['frame'] = update_frame(context['entities'], context['frame'])

	    # Fetch results from the knowledge base using all entities in frame as filters.
	    results = get_video_content(context['frame'])

	    # Fill the slots with the frame.
	    slots = fill_browse_slots(context['frame'], slots)

	    # Build response based on available slots and results.
	    reply, videos_client_action, prompt = build_browse_response(context, slots, results)

	    responder.reply(reply)

	    # Build and return the client action
	    videos_client_action = video_results_to_action(results)
	    responder.respond(videos_client_action)

This code follows a series of steps to build the final answer to the user: it updates the :doc:`dialogue frame <../userguide/dm>` with the new found entities, fetches results from the knowledge base (in the ``get_video_content`` method), builds a response with the new entities (done in ``fill_browse_slots`` and ``build_browse_response``) and sends a response to the user.

For more information on the ``show_content`` method and the functinos it calls, see the ``app.py`` file in the blueprint folder.

5. Knowledge Base
^^^^^^^^^^^^^^^^^