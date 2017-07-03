Step 10: Deploy Trained Models To Production
============================================

Once your application has been built, Workbench makes it easy to test locally and then deploy into production. In :doc:`Step 4 <04_define_the_dialogue_handlers>`, we created an application container for your dialogue state handler logic. This was the ``app.py`` file in the application root directory. To provide the necessary interface to manage deployment, we now append the following two lines of code to this file.

.. code:: python

  if __name__ == "__main__":
      app.cli()

Our ``app.py`` now looks something like the following.

.. code:: python

  import sys
  from mmworkbench import Application
  app = Application(__name__)

  @app.handle(intent='greet')
  def welcome():
      return {'replies': ['Hello there!']}
  ...

  if __name__ == "__main__":
      app.cli()

We can now test our application locally, or deploy remotely into a production environment.

Local Deployment
~~~~~~~~~~~~~~~~

If you have not already, you must build your models before the application can be run. To build
your models use the ``build`` command:

.. code-block:: console

    $ python app.py build
    Building application app.py...complete.

To launch the web service use the ``run`` command:

.. code-block:: console
    $ python app.py run
    Numerical parser running, PID 20248
    Loading intent classifier: domain='store_info'
    ...
     * Running on http://0.0.0.0:7150/ (Press CTRL+C to quit)

To test using any REST client (such as Postman or Advanced Rest Client), send `POST` requests to the web service endpoint at ``http://localhost:7150/parse``. Alternately, you can use a :keyword:`curl` command from your terminal as follows:

.. code-block:: console

  $ curl -X POST -d '{"text": "hello world"}' "http://localhost:7150/parse" | jq .
  {
    "client_actions": [
      {
        "message": {
          "text": "Hello. I can help you find store hours for your local Kwik-E-Mart. How can I help?"
        },
        "name": "show-prompt"
      }
    ],
    "dialogue_state": "welcome",
    "domain": "store_info",
    "entities": [],
    "frame": {},
    "history": [],
    "intent": "greet",
    "query_id": "e84d2e28-0228-4a16-8841-8598a1cb550a",
    "request": {
      "session": {},
      "text": "hello world"
    },
    "response_time": 0.06281208992004395,
    "version": "2.0"
  }

The web service responds with a JSON data structure containing the application response along with the detailed output for all of the machine learning components of the Workbench platform.

See the :ref:`User Guide <userguide>` for more about the Workbench request and response interface format.

MindMeld Cloud Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

Coming Soon
