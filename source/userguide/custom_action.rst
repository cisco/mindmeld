Working with Custom Actions
===========================

MindMeld 4.3 provides the ability for external applications to integrate custom dialogue management logic with MindMeld applications.


Why Custom Actions?
-------------------

Suppose that a team is working to build a MindMeld application and integrate it the rest of their microservices.
Since the microservices are written in Java, the developers want to reuse as much logic as
possible. Since the MindMeld application is in Python, they would have to re-implement their Java code to Python code.

With Custom Actions, the developers can now shift the responsibility of fulfilling business logic to services that
are outside of MindMeld applications, and these services can be implemented in any language. The developers can specify
the exact conditions (for example, matching a certain domain or intent) to execute the custom actions and the MindMeld
application will interact with the custom action servers through http(s) requests.


OpenAPI Specification for client-server interaction
---------------------------------------------------

Our client – server protocol is documented `here <https://github.com/cisco/mindmeld/blob/master/mindmeld/openapi/custom_action.yaml>`_.
For easy viewing you can check out Swagger UI, which can also help to render OpenAPI protocol into client and server
stubs in a variety of languages.

Each request to the Custom Action Server includes two fields: ``request`` and ``responder``. The
``request`` field encapsulates information from the `Request <https://www.mindmeld.com/docs/apidoc/mindmeld.components.request.html#mindmeld.components.request.Request>`_
object (text, NLP information, etc.) and the ``responder`` field the `Responder <https://www.mindmeld.com/docs/apidoc/mindmeld.components.dialogue.html#mindmeld.components.dialogue.DialogueResponder>`_ object (directives, frame, slots, params).

In a normal MindMeld application, each object is passed into the application handler for processing.

.. code-block:: python

    @app.handle(intent='some_intent')
    def handle_intent(request, responder):
        # normal MindMeld application logic goes here

In an analogous process to ``@app.handle``, when we invoke the custom action, we are serializing the ``request`` and ``responder`` and
passing them into the body of the http request to the server. In the response, we simply expect it to contain all
the fields of the ``Responder`` object which will be deserialized and processed before returning to the end client.


Configure Custom Actions for your MindMeld application
------------------------------------------------------

You can specify custom action configuration in ``config.py`` with the ``url`` field:

.. code-block:: python

    CUSTOM_ACTION_CONFIG = {"url": "http://0.0.0.0:8080/action"}

Currently MindMeld also supports SSL encryption by specifying the following fields in
the ``CUSTOM_ACTION_CONFIG``: ``cert``, ``public_key`` and ``private_key``. Each field
is the path to the local location of the certificate, public and private key.

In your Dialogue Manager, you can access the application's custom action config by referencing the application's
property ``app.custom_action_config``.


A Sample Custom Action Server
-----------------------------

To understand the behavior of Custom Actions, let's take a look at example of a Custom Action Server. We have added a
sample server for Python 3.6+ in the github directory under `examples/custom_action/example_server`. This server is adapted from the
sample server that is auto-generated from our OpenAPI protocol using the `Swagger Online Editor <https://editor.swagger.io/>`_.

To setup the sample server, you can run the following command in the terminal:

.. code-block:: console

    pip install mindmeld[examples]

    # optionally, to view the MindMeld API in the local browser:
    pip install "connexion[swagger-ui]"


    git clone git@github.com:cisco/mindmeld.git

    cd mindmeld/examples/custom_action/example_server

    # to run the server
    python -m swagger_server


The server will run locally at address ``http://0.0.0.0:8080/action``.

In this example, the server simply returns a reply directive for each action request that includes the name of the action.

.. code-block:: python

    import connexion

    from ..models.data import Data
    from ..models.responder import Responder
    from ..models.directive import Directive


    def invoke_action(body):
        """Invoke an action

        This API accepts the serialized MindMeld Request and Responder and returns a serialized Responder

        :param body:
        :type body: dict | bytes

        :rtype: Responder
        """
        directives = []
        if connexion.request.is_json:
            data = Data.from_dict(body)

            msg = "Invoking {action} on custom server.".format(action=data.action)

            reply = Directive(name="reply", payload={"text": msg}, type="view")
            directives.append(reply)
        responder = Responder(directives=directives, frame={})
        return responder

To test the server, you can run the following code snippet:

.. code-block:: python

    from mindmeld.components.custom_action import CustomAction
    action_config = {'url': 'http://localhost:8080/action'}

    action = CustomAction(name='action_call_people', config=action_config)
    from mindmeld.components.request import Request
    from mindmeld.components.dialogue import DialogueResponder

    # should get 400 since the request fields are missing
    action.invoke(Request(), DialogueResponder())

    # synchronous case
    request = Request(text='some text', domain='some domain', intent='some intent')
    responder = DialogueResponder()

    # we should see a successful request and one reply directive
    action.invoke(request, responder)
    print('Directives:', responder.directives)

You can explore the implementation of the Request and Responder data objects in our sample server to return different
fields of MindMeld.


Using Custom Actions with MindMeld applications
-----------------------------------------------

Add a call to a custom action as follows:

.. code-block:: python

    app = Application(__name__)
    app.custom_action(intent='deny', action='action_restart')

In the above example, we are specifying that when `deny` intent is reached, the application
should make a call for ``action_restart`` to the URL specified in ``CUSTOM_ACTION_CONFIG``.

In our response, we should see one reply directive with the message: ``Invoking action_restart on custom server``.

If your application is asynchronous, you can specify the custom action to be executed
asynchronously with the ``async_mode`` flag.

.. code-block:: python

    app = Application(__name__, async_mode=True)
    app.custom_action(intent='deny', action='action_restart', async_mode=True)

If there are more than one custom action server, you can also choose to
specify the server by passing the custom action config directly into the application.

.. code-block:: python

    config = {"url": "http://0.0.0.0:8080/action"}
    app.custom_action(intent='deny', action='action_restart', config=config)

If you want to execute a sequence of custom actions, you can pass the list of actions into
the ``actions`` field.

.. code-block:: python

    app.custom_action(intent='ask_help', actions=['action_help', 'action_restart'])

In our response, we should see two replies: ``Invoking action_help on custom server``,
``Invoking action_restart on custom server``.

The default behavior for executing a sequence of custom actions is to merge all of their fields in the final
``responder``. If we set the ``merge`` flag to be ``False``, we will only keep the result of the last action.

.. code-block:: python

    app.custom_action(intent='ask_help', actions=['action_help', 'action_restart'], merge=False)

Here, in the final response, we will see only one reply: "Invoking action_restart on custom server".


Calling Individual Custom Actions inside a MindMeld application
---------------------------------------------------------------

You can invoke individual custom actions by calling the ``CustomAction`` object directly. You can access the current
application's custom action configuration from the application's property ``app.custom_action_config``.

.. code-block:: python

    @app.handle(intent='restart')
    def action_check_out(request, responder):
        from mindmeld.components import CustomAction
        CustomAction(name='action_restart', config=app.custom_action_config).invoke(request, responder)

Alternatively, you can define a new application's config and pass it directly into the ``CustomAction``.

.. code-block:: python

    config = {"url": "http://0.0.0.0:8080/action"}
    CustomAction(name='action_restart', config=config).invoke(request, responder)

The advantage of invoking a custom action manually is that you can further refine and process
the results from the custom actions. Here the resulting fields are merged into the ``responder``
object.

Similarly to the ``custom_action`` handler, we can pass the ``merge`` flag into the ``CustomAction``
object to set its behavior for handling the fields of the returned ``Responder``.

.. code-block:: python

    @app.handle(intent='restart')
    def action_check_out(request, responder):
        CustomAction(name='action_restart', config=config, merge=True).invoke(request, responder)

You can also invoke the CustomAction asynchronously as well:

.. code-block:: python

    @app.handle(intent='restart')
    async def action_check_out(request, responder):
        await CustomAction(name='action_restart', config=config).invoke(request, responder, async_mode=True)

We can pipe multiple custom actions easily in a sequence and mix this sequence with any operation
by the ``responder``.

.. code-block:: python

    @app.handle(intent='ask_help')
    def handle_ask_help(request, responder):
        responder.reply('I can help you')
        CustomAction(name='action_help', config=config).invoke(request, responder)
        CustomAction(name='action_restart', config=config).invoke(request, responder)

In the example above, first we choose to add a reply first, and then invoke two custom actions in sequence.

In the final result, we should see three replies: ``I can help you``, ``Invoking action_help on custom server``,
``Invoking action_restart on custom server``.

Instead of calling individual ``CustomAction`` in sequence, you can also use the ``CustomActionSequence`` class.

.. code-block:: python

    @app.handle(intent='ask_help')
    def handle_ask_help(request, responder):
        from mindmeld.components import CustomActionSequence

        responder.reply('I can help you')
        CustomActionSequence(actions=['action_help', 'action_restart'], config=config).invoke(request, responder)

For your convenience, we also provide helper functions (``invoke_custom_action``, ``invoke_custom_action_async``) which wrap
around the ``CustomAction`` class.

.. code-block:: python

    @app.handle(intent='restart')
    def action_check_out(request, responder):
        from mindmeld.components import invoke_custom_action
        invoke_custom_action('action_restart', config, request, responder)
