Webex Teams Integration
=======================


A conversational app built using MindMeld can be integrated seamlessly with Webex Teams.
In this tutorial, we will create a Webex Teams bot based on the :doc:`Food Ordering <../blueprints/food_ordering>` blueprint. The food ordering app is an example of a deep-domain conversational agent that uses a knowledge base of restaurants and menu items to understand complex user queries that reference dishes from different restaurant menus.

.. note::

   Please make sure to install the Webex Teams dependency by running in the shell: :code:`pip install mindmeld[bot]`.

Quick Start
-----------

In this quick start guide, we will use a webhook and `ngrok` to integrate the food ordering application to Webex Teams. By setting up a publicly accessible server, one can register the server's IP address to the webhook, thereby allowing Webex Teams to send messages to the food ordering application. This guide should take between five and fifteen minutes to complete.

1. Register Webex Bot
^^^^^^^^^^^^^^^^^^^^^

Create a `new Webex Bot integration <https://developer.webex.com/my-apps/new/bot>`_ and store the Bot's Access Token safely.

.. image:: /images/create_bot.png
    :width: 700px
    :align: center

Next, make sure you make the local server endpoint for port 8080 publicly accessible. This could be done through a system like Nginx.


2. Register Webex Webhook
^^^^^^^^^^^^^^^^^^^^^^^^^

Register a `new webhook <https://developer.webex.com/docs/api/v1/webhooks/create-a-webhook>`_. Set the ``targetUrl`` as ngrok's public URL, set ``resource`` to ``messages``, and set ``event`` to ``created``. Once you click `Run`, store the ``id`` field in the response. This will be your webhook ID.

.. image:: /images/webhook.png
    :width: 700px
    :align: center


3. Building the bot
^^^^^^^^^^^^^^^^^^^

You can import `WebexBotServer` to build a Webex bot application server with MindMeld.

First, you need to install the specific dependencies for bot integration.

.. code:: console

   pip install mindmeld[bot]

After that you can instantiate an WebexBotServer instance and pass your application logic and the Webhook ID and the Access Token into the server.

.. code:: python

   from mindmeld.components import NaturalLanguageProcessor
   from mindmeld.bot import WebexBotServer
   from mindmeld import configure_logs

   if __name__ == '__main__':
      # Create web hook here: https://developer.webex.com/docs/api/v1/webhooks/create-a-webhook
      WEBHOOK_ID = os.environ.get('WEBHOOK_ID')

      # Create bot access token here: https://developer.webex.com/my-apps/new
      ACCESS_TOKEN = os.environ.get('BOT_ACCESS_TOKEN')

      configure_logs()
      nlp = NaturalLanguageProcessor('.')
      nlp.build()

      server = WebexBotServer(name=__name__, app_path='.', nlp=nlp, webhook_id=WEBHOOK_ID,
                              access_token=ACCESS_TOKEN)

      port_number = 8080
      print('Running server on port {}...'.format(port_number))

      server.run(host='localhost', port=port_number)

You can download the food ordering blueprint or HR blueprint for an example implementation.

.. code:: python

   import mindmeld as mm
   mm.configure_logs()
   bp_name = 'food_ordering'
   mm.blueprint(bp_name)


4. Start the food ordering server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the following environment variables and start the flask server.

.. code:: console

   cd food_ordering
   export WEBHOOK_ID=<insert webhook id>
   export BOT_ACCESS_TOKEN=<insert bot access token>
   python webex_bot_server.py


5. Test the integration
^^^^^^^^^^^^^^^^^^^^^^^

Create a Webex Teams space and add the bot to the space. To trigger the webhook, simply @mention the bot and converse with it.

.. image:: /images/bot_interaction.png
    :width: 700px
    :align: center
