Step 10: Deploy Trained Models To Production
============================================

Apart from serving as a library of Machine Learning tools for building powerful conversational interfaces, MindMeld Workbench also provides functionality for easily deploying the models to a server for testing and serving live user queries. This is done using a lightweight micro web framework built into MindMeld Workbench for local deployment. For deploying to production, running a single command is sufficient to trigger a remote process on a production-ready backend infrastructure (more details below). Finally, a simple "Test Console" UI is provided along with the deployment setup to visualize the end-to-end flow for any given request.

Local Deployment
~~~~~~~~~~~~~~~~

To test your end-to-end application on a local machine, spin up a local server by running the following command:

.. code-block:: text

  python my_app.py deploy --environment local

This loads all the NLP models and any other dependencies. Once the local server process is up and running, you should see the following message on the terminal:

.. code-block:: text

  2017-01-21 03:12:26,870  * Running on http://0.0.0.0:7150/ (Press CTRL+C to quit)

If there are no other errors reported on ``stdout``, it means that the server startup was successful. You can now run a query through the end-to-end system using the Test Console UI. The test console can then be accessed on a browser at the following URL:

* http://localhost:7150/test.html

The following HTML page opens up. The different tabs show various details on the steps involved in Natural Langauge Processing, Question Answering and Dialog Management for any input query. This is helpful for debugging the each of the components for specific queries.

.. image:: images/test_console.png
    :align: center

The web framework also exposes a ``/parse`` API endpoint that accepts POST requests. You can use any REST client (such as Postman or Advanced Rest Client) to trigger the endpoint with a ``query`` parameter. Alternately, you can use curl:

.. code-block:: text

  curl -X POST http://localhost:7150/parse?query="Hi"

This returns a JSON response containing all the necessary information:

.. code-block:: javascript

  {
    "query": "Hi",
    "domain": "store_information",
    "intent": "greet",
    "entities": {},
    "numeric-entities": {},
    "dialogue_context": {
      "dialogue_id": "c4898837-9dca-4b43-9e93-3c72e3b4c35c",
      "history": [],
      "state": "welcome",
      "response_type": "answer"
    },
    "reply": "Hello, Bart. I can help you find store hours for your local Kwik-E-Mart. How can I help?",
    "request_context": {
      "query_id":"68f07386-fef6-49e9-b0b6-cf07405e607c"
    }
  }

Production Deployment
~~~~~~~~~~~~~~~~~~~~~

MindMeld provides a "Backend As A Service" platform exclusively for MindMeld Workbench applications. All the cloud infrastructure setup needed to deploy production-grade conversational applications is available off-the-shelf with a valid MindMeld license. So with a few simple configurations, you can go from a standalone research project to a full-blown production application within minutes!

The overall process starts with a developer (or build process) invoking a single command to send all the catalog data, NLP models, deployment configurations, requirements.txt and authorization keys to a **MindMeld Deployment Service**. The MindMeld Deployment Service takes these inputs, verifies the auth keys and uses the deployment config to setup any required infrastructure on a Virtual Private Cloud (VPC) in the MindMeld backend. At a high level, this includes setting up the Knowledge Base, server clusters, load balancer and DNS aliases for the application URL with relevant endpoints exposed. Here is a diagram that shows the basic process:

.. image:: images/deployment.png
    :align: center

In order to setup a production deployment of your app, you need to first obtain the ``MINDMELD_CLIENT_KEY`` and ``MINDMELD_CLIENT_SECRET`` credentials from MindMeld Sales. These need to be setup as OS environment variables on your local machine (or the build process used for deploying the app):

.. code-block:: text

  export MINDMELD_CLIENT_KEY='my_mindmeld_client_key'
  export MINDMELD_CLIENT_SECRET='my_mindmeld_client_secret'

Next, you can define a "deployment config" file. This file specifies all the operational configurations for your production deployment, such as number of servers, datacenter region, load balancers etc. Here is an example deployment configuration for the Kwik-E-Mart Stores application:

File **deployment_config.json**

.. code-block:: javascript

  {
    "app_name": 'kwik-e-mart',
    "region": 'us-east-1',
    "num_instances": 10,
    "instance_type": 'm3.xlarge',
    "use_ssl": True,
    "use_lb": True
  }

You can then run the following command to set off the deployment:

.. code-block:: text

  python my_app.py deploy --environment production --data_path '/path/to/stores.json' --deployment_config deployment_config.json

And you're done! Once the deployment is complete (and no errors are encountered) you should see the following message on stdout:

.. code-block:: text

  Name: kwik-e-mart
  Description: None
  Creation Date: 2017-01-09T02:57:50+00:00
  URL: https://kwik-e-mart.mindmeld.com/

  Deployment successful!

You can then fire up the production app test console and the ``/parse`` API endpoint on the following links:

.. code-block:: text

  https://kwik-e-mart.mindmeld.com/test.html
  https://kwik-e-mart.mindmeld.com/parse?q="Hello"

Congratulations. You have learned how to build the most advanced conversational interfaces. Happy chatting!
