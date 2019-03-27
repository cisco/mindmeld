Step 10: Deploy Trained Models
==============================

Once your application has been built, Workbench makes it easy to test locally. In :doc:`Step 4 <04_define_the_dialogue_handlers>`, we created an application container for your dialogue state handler logic. This was in the ``__init__.py`` file in the application root directory. To provide the necessary interface to manage deployment, we now create the ``__main__.py`` file with the following three lines of code.

.. code:: python

  if __name__ == "__main__":
      from . import app
      app.cli()

We can now test our application locally.


If you have not already, you must build your models before the application can be run. To build your models use the ``build`` command:

.. code-block:: console

    cd $HOME
    python -m my_app build

.. code-block:: console

    Building application my_app...complete.

To launch the web service use the ``run`` command:

.. code-block:: console

    cd $HOME
    python -m my_app run

.. code-block:: console

    Numerical parser running, PID 20248
    Loading intent classifier: domain='store_info'
    ...
     * Running on http://0.0.0.0:7150/ (Press CTRL+C to quit)

To test using any REST client (such as Postman or Advanced Rest Client), send `POST` requests to the web service endpoint at ``http://localhost:7150/parse``. Alternately, you can use a :manpage:`curl` command from your terminal as follows:

.. code-block:: console

  curl -X POST -H 'Content-Type: application/json' -d '{"text": "hello world"}' "http://localhost:7150/parse" | jq .

.. code-block:: console

  {
    "dialogue_state": "welcome",
    "directives": [
      {
        "name": "reply",
        "payload": {
          "text": "Hello. I can help you find store hours for your local Kwik-E-Mart. How can I help?"
        },
        "type": "view"
      },
      {
        "name": "listen",
        "type": "action"
      }
    ],
    "frame": {},
    "history": [
      {
        "dialogue_state": "welcome",
        "directives": [
          {
            "name": "reply",
            "payload": {
              "text": "Hello. I can help you find store hours for your local Kwik-E-Mart. How can I help?"
            },
            "type": "view"
          },
          {
            "name": "listen",
            "type": "action"
          }
        ],
        "frame": {},
        "params": {
          "allowed_intents": [],
          "dynamic_resource": {},
          "previous_params": {
            "allowed_intents": [],
            "dynamic_resource": {},
            "previous_params": null,
            "target_dialogue_state": null,
            "time_zone": null,
            "timestamp": 0
          },
          "target_dialogue_state": null,
          "time_zone": null,
          "timestamp": 0
        },
        "request": {
          "confidences": {},
          "context": {},
          "domain": "store_info",
          "entities": [],
          "frame": {},
          "history": [],
          "intent": "greet",
          "nbest_aligned_entities": [],
          "nbest_transcripts_entities": [],
          "nbest_transcripts_text": [],
          "params": {
            "allowed_intents": [],
            "dynamic_resource": {},
            "previous_params": {
              "allowed_intents": [],
              "dynamic_resource": {},
              "previous_params": null,
              "target_dialogue_state": null,
              "time_zone": null,
              "timestamp": 0
            },
            "target_dialogue_state": null,
            "time_zone": null,
            "timestamp": 0
          },
          "text": "hello world"
        },
        "slots": {}
      }
    ],
    "params": {
      "allowed_intents": [],
      "dynamic_resource": {},
      "previous_params": {
        "allowed_intents": [],
        "dynamic_resource": {},
        "previous_params": null,
        "target_dialogue_state": null,
        "time_zone": null,
        "timestamp": 0
      },
      "target_dialogue_state": null,
      "time_zone": null,
      "timestamp": 0
    },
    "request": {
      "confidences": {},
      "context": {},
      "domain": "store_info",
      "entities": [],
      "frame": {},
      "history": [],
      "intent": "greet",
      "nbest_aligned_entities": [],
      "nbest_transcripts_entities": [],
      "nbest_transcripts_text": [],
      "params": {
        "allowed_intents": [],
        "dynamic_resource": {},
        "previous_params": {
          "allowed_intents": [],
          "dynamic_resource": {},
          "previous_params": null,
          "target_dialogue_state": null,
          "time_zone": null,
          "timestamp": 0
        },
        "target_dialogue_state": null,
        "time_zone": null,
        "timestamp": 0
      },
      "text": "hello world"
    },
    "request_id": "38dd4c17-1440-492c-8d4b-eeacd7e108e6",
    "slots": {},
    "response_time": 0.018073081970214844,
    "version": "2.0"
  }

The web service responds with a JSON data structure containing the application response along with the detailed output for all of the machine learning components of the Workbench platform.

.. See the :ref:`User Guide <userguide>` for more about the Workbench request and response interface format.

.. Cloud Deployment
.. ~~~~~~~~~~~~~~~~~~~~~~~~~

.. Coming Soon
