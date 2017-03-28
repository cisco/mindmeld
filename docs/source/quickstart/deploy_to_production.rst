Step 10: Deploy Trained Models To Production
============================================

Once your application has been built, Workbench makes it easy to test locally and then deploy into production. In :doc:`Step 4 </define_the_dialogue_handlers>`, we created an application container for your dialogue state handler logic. This was the :keyword:`my_app.py` file in the application root directory. To provide the necessary interface to manage deployment, we now append the following two lines of code to this file.

.. code:: python

  if __name__ == "__main__":
      app.run(sys.argv)

Our :keyword:`my_app.py` now looks something like the following.

.. code:: python

  import sys
  from mmworkbench import Application
  app = Application(__name__)

  @app.handle(intent='greet')
  def welcome():
      return {'replies': ['Hello there!']}
  ...

  if __name__ == "__main__":
      app.run(sys.argv)

We can now test our application locally, or deploy remotely into a production environment.

Local Deployment
~~~~~~~~~~~~~~~~

To run the application locally, navigate to your application root directory in a terminal window, and enter the following command.

.. code-block:: console

    $ python my_app.py
    Building application my_app.py...complete.
    Running application my_app.py on http://localhost:7150/ (Press CTRL+C to quit)

This command first builds all of the machine learning models required in your application. Once this training is complete, a local web service is launched for your application. To test your running application with a browser-based console, navigate to ``http://localhost:7150/test.html``. To test using any REST client (such as Postman or Advanced Rest Client), send `POST` requests to the web service endpoint at ``http://localhost:7150/parse``. Alternately, you can use a :keyword:`curl` command from your terminal as follows:

.. code-block:: console

  $ curl -X POST -d '{"query": "hello world"}' "http://localhost:7150/parse"
  {
    'request': {...},
    'domain': {'target': 'store_info'},
    'intent': {'target': 'greet', 'probs': [...]},
    'dialogue_state': 'welcome',
    'entities': [],
    'parse_tree': [],
    'frame': {},
    'history': [],
    'client_actions': [
      'replies': ['Hello there!']
    ]
  }

The web service responds with a JSON data structure containing the application response along with the detailed output for all of the machine learning components of the Workbench platform.

To build your models *without* launching a web service, use the :keyword:`build` command-line argument:

.. code-block:: console

    $ python my_app.py build
    Building application my_app.py...complete.


To launch a web service without building models beforehand, use the :keyword:`run` command-line argument:

.. code-block:: console

    $ python my_app.py run
    Running application my_app.py on http://localhost:7150/ (Press CTRL+C to quit)

See the :ref:`User Manual <userguide>` for more about the Workbench request and response interface format.


MindMeld Cloud Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~

**MindMeld Cloud**, a cloud-based managed service offering, streamlines the deployment and scaling of production conversational applications. Every application deployed on the MindMeld Cloud is hosted in a dedicated, secure, private environment. Each MindMeld Cloud environment supports flexible scaling options which can accommodate complex applications with billions of monthly queries. Some of the largest global enterprises rely on MindMeld Cloud to host and serve their mission-critical conversational applications.

To get started with MindMeld Cloud, please `contact MindMeld sales <mailto:info@mindmeld.com>`_ to request a production license and your deployment credentials. Once you have received your deployment key and secret, you can configure Workbench with your credentials using the Python shell as follows.

.. code:: python

  >>> from mmworkbench import MindMeldCloud as mmCld
  >>> mmCld.config({'key': 'my-access-key', 'secret': 'my-access-secret'})
  >>> mmCld.dump()

Next, run your application with the :keyword:`deploy` command-line argument in a terminal window.

.. code-block:: console

    $ python my_app.py deploy
    Name: kwik-e-mart
    Description: None
    Creation Date: 2017-01-21T02:57:50+00:00
    URL: https://kwik-e-mart.mindmeld.com/

    Deployment successful!

This command launches a secure private environment in the MindMeld Cloud. This environment can include a cluster of load-balanced instances that automatically scale to handle application load. The environment includes a replica of your knowledge bases, suitable for production operation, as well as all of your trained machine learning models, optimized for low-latency execution. Completing each cloud deployment can take several seconds to a few minutes. When deployment completes, an HTTPS web service becomes available at the unique production URL displayed in the console.

By default, the unique production URL includes the application name associated with your deployment key. In this example, the production URL is ``https://kwik-e-mart.mindmeld.com/``. Along the same lines as the local deployment procedure above, you can view a web-based test console by loading ``https://kwik-e-mart.mindmeld.com/test.html`` into your browser. Alternately, you can use the :keyword:`/parse` web service endpoint as illustrated in the following :keyword:`curl` command.

.. code-block:: console

  $ curl -X POST -d '{"query": "hello world"}' "https://kwik-e-mart.mindmeld.com/parse"
  {
    'request': {...},
    'domain': {'target': 'store_info'},
    'intent': {'target': 'greet', 'probs': [...]},
    'dialogue_state': 'welcome',
    'entities': [],
    'parse_tree': [],
    'frame': {},
    'history': [],
    'client_actions': [
      'replies': ['Hello there!']
    ]
  }

The MindMeld Cloud provides flexible configuration and deployment options to handle applications of any complexity and query volume. See the :ref:`User Manual <userguide>` for details.
