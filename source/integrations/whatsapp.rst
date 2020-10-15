WhatsApp Integration
====================

A conversational app built using MindMeld can be integrated seamlessly with WhatsApp through the Twilio sandbox.
In this tutorial, you will create a WhatsApp bot based on the :doc:`Human Resource <../blueprints/hr_assistant>` blueprint. The HR assistant app is an example of a deep-domain conversational agent that uses a knowledge base of human resources and policies to understand complex user queries that reference employees within a company.

.. note::

   Please make sure to install the Twilio dependency by running in the shell: :code:`pip install mindmeld[bot]`. You also need to register `a sandbox account with Twilio <https://www.twilio.com/console/sms/whatsapp/sandbox>`_.

Quick Start
-----------

In this quick start guide, you will use a Twilio sandbox and `ngrok` to integrate the human resource application to WhatsApp. By setting up a publicly accessible server, one can register the server's IP address to the Twilio developer sandbox, thereby allowing WhatsApp to send messages to the human resource application through a phone number registered with the sandbox. This guide should take between five and fifteen minutes to complete.


1. Register a sandbox
^^^^^^^^^^^^^^^^^^^^^

Login or create a `new Twilio account <https://www.twilio.com>`_.


Next, make sure you `register the sandbox and confirm it on WhatsApp <https://www.twilio.com/console/sms/whatsapp/learn>`_.


2. Building the bot
^^^^^^^^^^^^^^^^^^^

First, you need to install the specific dependencies for bot integration.

.. code:: console

   pip install mindmeld[bot]

After that you can instantiate a WhatsappBotServer instance. A sample implementation is provided in the HR blueprint.

.. code:: console

   mindmeld blueprint hr_assistant

After downloading the HR blueprint, you can inspect the implementation in `whatsapp_bot_server.py`.

The code sample is also available in our repo under `examples/whatsapp <https://github.com/cisco/mindmeld/tree/master/examples/whatsapp>`_.

3. Start the HR assistant app server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the following environment variables and start the Flask server.

.. code:: console

   python -m hr_assistant build
   cd hr_assistant
   python whatsapp_bot_server.py


4. Test the integration
^^^^^^^^^^^^^^^^^^^^^^^

Start the ngrok channel. You can download the ngrok application from the Internet and then unzip it in a safe location.

.. code:: console

   ./ngrok http 8080

   Session Status                online
   Session Expires               7 hours, 59 minutes
   Update                        update available (version 2.3.35, Ctrl-U to update)
   Version                       2.3.29
   Region                        United States (us)
   Web Interface                 http://127.0.0.1:4041
   Forwarding                    http://be84be34.ngrok.io -> http://localhost:8080
   Forwarding                    https://be84be34.ngrok.io -> http://localhost:8080

   Connections                   ttl     opn     rt1     rt5     p50     p90
                                 0       0       0.00    0.00    0.00    0.00

After running the ngrok application, copy the ngrok URL and paste into the Twilio sandbox's configuration.

.. image:: /images/whatsapp_sandbox.png
    :width: 700px
    :align: center

Now you can converse with HR assistant on WhatsApp!

.. image:: /images/whatsapp_chat.png
    :width: 700px
    :align: center
