Webex Teams Integration
=======================


A conversational app built using MindMeld can be integrated seamlessly with Webex Teams.
In this tutorial, we will create a Webex Teams bot based on the :doc:`Food Ordering <../blueprints/food_ordering>` blueprint. The food ordering app is an example of a deep-domain conversational agent that uses a knowledge base of restaurants and menu items to understand complex user queries that reference dishes from different restaurant menus.

Quick Start
-----------

In this quick start guide, we will use a webhook and `ngrok` to integrate the food ordering application to Webex Teams. By setting up a publicly accessible server, one can register the server's IP address to the webhook, thereby allowing Webex Teams to send messages to the food ordering application. This guide should take between five and fifteen minutes to complete.

1. Register Webex Bot
^^^^^^^^^^^^^^^^^^^^^

Create a `new Webex Bot integration <https://developer.webex.com/my-apps/new/bot>`_ and store the Bot's Access Token safely.

.. image:: /images/create_bot.png
    :width: 700px
    :align: center


2. Use Ngrok to point your local server port as a public URL
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We use `ngrok <https://ngrok.com/>`_ to set up a publicly accessible tunnel to the local food ordering flask server at port 8080.

.. code:: console

  ./ngrok http 8080

.. image:: /images/ngrok.png
    :width: 700px
    :align: center


3. Register Webex Webhook
^^^^^^^^^^^^^^^^^^^^^^^^^

Register a `new webhook <https://developer.webex.com/docs/api/v1/webhooks/create-a-webhook>`_. Set the ``targetUrl`` as ngrok's public URL, set ``resource`` to ``messages``, and set ``event`` to ``created``. Once you click `Run`, store the ``id`` field in the response. This will be your webhook ID.

.. image:: /images/webhook.png
    :width: 700px
    :align: center


4. Download
^^^^^^^^^^^

Download the food ordering blueprint.

.. code:: python

   import mindmeld as mm
   mm.configure_logs()
   bp_name = 'food_ordering'
   mm.blueprint(bp_name)


5. Start the food ordering server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the following environment variables and start the flask server.

.. code:: console

   cd food_ordering
   export WEBHOOK_ID=<insert webhook id>
   export BOT_ACCESS_TOKEN=<insert bot access token>
   python webex_bot_server.py


6. Test the integration
^^^^^^^^^^^^^^^^^^^^^^^

Create a Webex Teams space and add the bot to the space. To trigger the webhook, simply @mention the bot and converse with it.

.. image:: /images/bot_interaction.png
    :width: 700px
    :align: center
