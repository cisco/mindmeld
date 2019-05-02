Getting Started
===============

These instructions explain how to install MindMeld on a Unix-based system and set up your first MindMeld project. Users of other operating systems like Windows can use Docker to get started.

.. note::

  MindMeld requires Python 3.4, 3.5, or 3.6.


Install MindMeld
----------------

You must choose the mechanism by which you install MindMeld. The supported choices are:

  - :ref:`Docker <getting_started_docker_setup>`
  - :ref:`virtualenv <getting_started_virtualenv_setup>`

If you're going to be using MindMeld often, **we recommend you do the virtualenv installation** and setup all dependencies locally. That will provide the optimal performance and experience. But if you want to get a taste of MindMeld with minimal effort, you can get started quickly using Docker.


.. _getting_started_docker_setup:

Install with Docker
^^^^^^^^^^^^^^^^^^^

The ``Dockerfile`` provided by MindMeld contains MindMeld and all its dependencies. Follow these steps to get started using Docker:

#. First, `install Docker <https://www.docker.com/community-edition#/download>`_, and run it.
#. Then, open a terminal (shell) and run this command:

.. code-block:: shell

   docker pull mindmeldworkbench/mindmeld
   docker run -p 0.0.0.0:7150:7150 mindmeldworkbench/mindmeld -ti -d

The Docker container contains Elasticsearch, the numerical parsing service, the MindMeld library and the Home Assistant application for you to test. The container will build and serve the application on port 7150 which is exposed to the external environment.

The application code and data is located at directory ``/root/home_assistant`` on the docker container.

For more information on the Home Assistant application, see :ref:`Home Assistant <home_assistant>` blueprint application.

To test the application inside docker, you can make a request:

.. code-block:: shell

   curl -X POST http://localhost:7150/parse -H 'Content-Type: application/json' -d '{"text":"good morning"}'

The output should be as follows:

.. code-block:: console

   {
     "directives": [
       {
         "name": "reply",
         "payload": {
           "text": "Hi, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances."
         },
         "type": "view"
       }
     ],
     .
     .
     .

Editing application with Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to make edits to the Home Assistant application in the running docker container, we can enter the docker container, modify the files, commit the changes and spin off a new container with the changes. Here is an example code snippet on how do do it:

Find the container id of the docker image running Home Assistant.

.. code-block:: shell

   docker ps

.. code-block:: console

   CONTAINER ID        IMAGE                         COMMAND                  CREATED             STATUS              PORTS                                        NAMES
   d696c64e9924        mindmeldworkbench/mindmeld    "/bin/sh -c 'export …"   7 minutes ago       Up 7 minutes        9200/tcp, 0.0.0.0:7150->7150/tcp, 9300/tcp   nervous_panini

With the container's ID as d696c64e9924, we connect to the docker's bash environment.

.. code-block:: shell

   docker exec -it d696c64e9924 bash


Now, open the ``home_assistant/greeting.py`` file in a text editor to make a change to one of the natural language responses. Instead of the agent replying ``Hi, I am your home assistant..``, we will replace the text to ``Hi Alice, I am your home assistant..``. Here is what the edited ``home_assistant/greeting.py`` file would look like:

.. code-block:: shell

   # -*- coding: utf-8 -*-
   """This module contains the dialogue states for the 'greeting' domain
   in the MindMeld home assistant blueprint application
   """
   from .root import app


   @app.handle(intent='greet')
   def greet(request, responder):
       responder.reply('Hi Alice, I am your home assistant. I can help you to check weather, set temperature'
                       ' and control the lights and other appliances.')

   @app.handle(intent='exit')
   def exit(request, responder):
       responder.reply('Bye!')


Make sure you save the file and quit the docker shell.

.. code-block:: shell

   exit

Commit the edited docker file system, stop the existing running container and restart the edited docker container.

.. code-block:: shell

   docker commit d696c64e9924 mindmeldworkbench/mindmeld:edited
   docker stop d696c64e9924
   docker run -p 0.0.0.0:7150:7150 mindmeldworkbench/mindmeld:edited -ti -d

Now issue the curl request again.

.. code-block:: shell

   curl -X POST http://localhost:7150/parse -H 'Content-Type: application/json' -d '{"text":"good morning"}'


In the output json, notice the payload reflect the ``Alice`` text change we made:

.. code-block:: console

   {
     "directives": [
       {
         "name": "reply",
         "payload": {
           "text": "Hi Alice, I am your home assistant. I can help you to check weather, set temperature and control the lights and other appliances."
         },
         "type": "view"
       }
     ],
     .
     .
     .


.. note::

  Using ``docker commit`` makes a copy of the existing docker container, adding several gigabytes to your file system. Consider pruning your docker containers on regular intervals using the command ``docker system prune``.


If you encounter any issues, see :ref:`Troubleshooting <getting_started_troubleshooting>`.

Proceed to :ref:`Begin New Project <getting_started_begin_new_project>`.


.. _getting_started_virtualenv_setup:

Install with virtualenv
^^^^^^^^^^^^^^^^^^^^^^^

1. Install prerequisites
""""""""""""""""""""""""

On a Ubuntu 16/18 machine, you can install the dependencies for MindMeld and set up the necessary configuration files with the `mindmeld_init.sh script <https://devcenter.mindmeld.com/scripts/mindmeld_init.sh>`_.

.. note::

   The script installs the following components after a confirmation prompt: ``docker``, ``python3.6``, ``python-pip``, ``virtualenv`` and Elasticsearch 6.7.

If you are using a Ubuntu 16/18 machine, when you're ready to go, open a terminal (shell) and run this command:

.. code-block:: shell

  bash -c "$(curl -s  https://devcenter.mindmeld.com/scripts/mindmeld_init.sh)"

If you encounter any issues, see :ref:`Troubleshooting <getting_started_troubleshooting>`.

For macOS users, a recent (April 16th 2019) change in licensing policy of Java prevents us from creating an automatic script to download and run it. Java is necessary for Elasticsearch 6.7 to run. Assuming you have Oracle Java or OpenJDK installed, please download the following libraries:

macOS:

+---------------+--------------------------------------------------------------------------------------------------------+
|    Component  |    Command                                                                                             |
+===============+========================================================================================================+
| brew          |  ``/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"``|
+---------------+--------------------------------------------------------------------------------------------------------+
| python3       |  ``brew install python3``                                                                              |
+---------------+--------------------------------------------------------------------------------------------------------+
| pip           |  ``sudo -H easy_install pip``                                                                          |
+---------------+--------------------------------------------------------------------------------------------------------+
| virtualenv    |  ``sudo -H pip install --upgrade virtualenv``                                                          |
+---------------+--------------------------------------------------------------------------------------------------------+
| Elasticsearch |  See instructions below to download Elasticsearch 6.7                                                  |
+---------------+--------------------------------------------------------------------------------------------------------+

.. code-block:: shell

   curl https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-6.7.0.tar.gz -o elasticsearch-6.7.0.tar.gz
   tar -zxvf elasticsearch-6.7.0.tar.gz
   cd elasticsearch-6.7.0/bin
   ./elasticsearch-6.7.0/bin/elasticsearch


Ubuntu:

+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|    Component  |    Command                                                                                                                                                                                                   |
+===============+==============================================================================================================================================================================================================+
| python3       |  ``sudo apt-get install python3.6``                                                                                                                                                                          |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| pip           |  ``sudo apt install python-pip``                                                                                                                                                                             |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| virtualenv    |  ``sudo apt install virtualenv``                                                                                                                                                                             |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Elasticsearch |  ``sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:6.7.0 && sudo docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:6.7.0``|
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


.. note::

  We use docker for Elasticsearch in Ubuntu since provisioning it for the Ubuntu OS is convoluted. See here for more details if you want to set up
  Elasticsearch from scratch on `Linux <https://www.digitalocean.com/community/tutorials/how-to-install-elasticsearch-logstash-and-kibana-elastic-stack-on-ubuntu-18-04>`_.:

2. Set up a virtual environment
"""""""""""""""""""""""""""""""

To prepare an isolated environment for MindMeld installation using ``virtualenv``, follow the following steps.

- Create a folder for containing all your MindMeld projects, and navigate to it:

.. code-block:: shell

  mkdir my_mm_workspace
  cd my_mm_workspace

- Setup a virtual environment by running one of the following commands:

.. code-block:: shell

   virtualenv -p python3 .

- Activate the virtual environment:

.. code-block:: shell

  source bin/activate


Later, when you're done working with MindMeld, you can deactivate the virtual environment with the ``deactivate`` command.

.. code-block:: shell

  deactivate


3. Install the MindMeld package
"""""""""""""""""""""""""""""""

Now that your environment is set up, you can install MindMeld just as you would any other Python package. This may take a few minutes.

.. code-block:: shell

  pip install mindmeld

If you see errors here, you likely entered incorrect credentials during :ref:`Setup <getting_started_virtualenv_setup>`. Make sure you use your credentials for the MindMeld Learning Center.

To verify your setup is good, run this command. If there is no error, the installation was successful:

.. code-block:: shell

  mindmeld


.. _duckling:

4. Start the numerical parser
"""""""""""""""""""""""""""""

MindMeld uses a Haskell-based numerical parser for detecting certain numeric expressions like times, dates, and quantities in user queries. Start the numerical parser with this command:

.. code-block:: shell

  mindmeld num-parse --start

If you encounter an error like ``OS is incompatible with duckling executable``, it means that
your operating system is not compatible with the pre-compiled numerical parser binary distributed
with MindMeld. You instead need to run the numerical parser using Docker as shown below.

.. code-block:: shell

   docker pull mindmeldworkbench/duckling:master && docker run -p 0.0.0.0:7151:7151 mindmeldworkbench/duckling -ti -d


.. warning::

   The numerical parser is a critical component that MindMeld relies on. **Do not skip this step**
   .


.. _getting_started_begin_new_project:

Begin New Project
-----------------

With the setup out of the way, you are now ready to get your feet wet. You can proceed in one of two ways:

#. Try out a :ref:`blueprint application <getting_started_blueprint>`. This is the **recommended approach** for beginners to familiarize themselves with MindMeld. This is also a good starting point if your use case matches one of the :doc:`blueprint scenarios <../blueprints/overview>`.

#. Start a :ref:`brand new project <getting_started_template>`. This is the approach to take if your specific use case isn't covered by an existing blueprint, or if you prefer to build out your app from scratch.

MindMeld is designed so you can keep using the tools and coding patterns that are familiar to you. Some of the very basic operations can be performed in your command-line shell using the ``mindmeld`` command. But to really take advantage of the power of MindMeld, the Python shell is where all the action is at. The examples in this section are accompanied by code samples from both shells.


.. _getting_started_blueprint:

Start with a blueprint
^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Blueprints are simple example apps that are intentionally limited in scope. They provide you with a baseline to bootstrap upon for common conversational use cases. To improve upon them and convert them into production-quality apps, follow the exercises in the :doc:`individual blueprint sections <../blueprints/overview>`.


Using the command-line
""""""""""""""""""""""

To try out the :doc:`Food Ordering blueprint<../blueprints/food_ordering>`, run these commands on the command line:

.. code-block:: shell

  mindmeld blueprint food_ordering
  python -m food_ordering build   # this will take a few minutes
  python -m food_ordering converse

.. code-block:: console

 Loading intent classifier: domain='ordering'
 ...
 You:

The ``converse`` command loads the machine learning models and starts an interactive session with the "You:" prompt.
Here you can enter your own input and get an immediate response back. Try "hi", for example, and see what you get.


Using the Python shell
""""""""""""""""""""""

To try out the :doc:`Home Assistant blueprint<../blueprints/home_assistant>`, run these commands in your Python shell:

.. code-block:: python

    import mindmeld as mm
    mm.configure_logs()
    blueprint = 'home_assistant'
    mm.blueprint(blueprint)

    from mindmeld.components import NaturalLanguageProcessor
    nlp = NaturalLanguageProcessor(blueprint)
    nlp.build()

    from mindmeld.components.dialogue import Conversation
    conv = Conversation(nlp=nlp, app_path=blueprint)
    conv.say('Hello!')


MindMeld provides several different blueprint applications to support many common use cases for
conversational applications. See :doc:`MindMeld Blueprints<../blueprints/overview>` for more usage examples.


.. _getting_started_template:

Start with a new project
^^^^^^^^^^^^^^^^^^^^^^^^

There is a special ``template`` blueprint that sets up the scaffolding for a blank project. The example below creates a new empty project in a local folder named ``my_app``.

Using the command-line
""""""""""""""""""""""

.. code-block:: shell

  mindmeld blueprint template myapp


Using the Python shell
""""""""""""""""""""""

.. code-block:: python

  import mindmeld as mm
  mm.configure_logs()
  mm.blueprint('template', 'my_app')

The :doc:`Step-By-Step guide <../quickstart/00_overview>` walks through the methodology for building conversational apps using MindMeld.


Upgrade Mindmeld
----------------

To upgrade to the latest version of MindMeld, run ``pip install mindmeld --upgrade``

Make sure to run this regularly to stay on top of the latest bug fixes and feature releases.

.. note::

   - As of version 3.3, we have moved the MindMeld package from the MindMeld-hosted PyPI to Cisco’s PyPI server. If you are using the old ``~/.pip/pip.conf``, please re-run :ref:`Step 1 <getting_started_virtualenv_setup>` to update your installation path.

   - Before re-downloading a :doc:`blueprint <../blueprints/overview>` using an upgraded version of MindMeld, please remove the blueprint cache by running this command: ``rm -r ~/.mindmeld/blueprints/*``


.. _cli:

Command-Line Interfaces
-----------------------

MindMeld has two command-line interfaces for some of the common workflow tasks you'll be doing often:

#. ``mindmeld``
#. ``python -m <app_name>``

Built-in help is available with the standard :option:`-h` flag.

mindmeld
^^^^^^^^

The command-line interface (CLI) for MindMeld can be accessed with the ``mindmeld`` command.
This is most suitable for use in an app-agnostic context.

The commands available are:

#. ``blueprint`` : Downloads all the training data for an existing :doc:`blueprint <../blueprints/overview>` and sets it up for use in your own project.
#. ``num-parse`` : Starts or stops the numerical parser service.


python -m <app_name>
^^^^^^^^^^^^^^^^^^^^

When you're in the context of a specific app, ``python -m <app_name>`` is more appropriate to use.

The commands available are:

#. ``build`` : Builds the artifacts and machine learning models and persists them.
#. ``clean`` : Deletes the generated artifacts and takes the system back to a pristine state.
#. ``converse`` : Begins an interactive conversational session with the user at the command line.
#. ``evaluate`` : Evaluates each of the classifiers in the NLP pipeline against the test set.
#. ``load-kb`` : Populates the knowledge base.
#. ``predict`` : Runs model predictions on queries from a given file.
#. ``run`` : Starts the MindMeld service as a REST API.


Configure Logging
-----------------

MindMeld adheres to the standard `Python logging mechanism <https://docs.python.org/3/howto/logging.html>`_.
The default logging level is ``WARNING``, which can be overridden with a config file or from code.
The ``INFO`` logging level can be useful to see what's going on:

.. code-block:: python

  import logging
  logging.getLogger('mindmeld').setLevel(logging.INFO)

There is a handy ``configure_logs()`` function available that wraps this and accepts 2 parameters:

#. :data:`format`: The `logging format <https://docs.python.org/3/howto/logging.html#changing-the-format-of-displayed-messages>`_.
#. :data:`level`: The `logging level <https://docs.python.org/3/howto/logging.html#logging-levels>`_.

Here's an example usage:

.. code-block:: python

  import mindmeld as mm
  mm.configure_logs()


.. _getting_started_troubleshooting:

Troubleshooting
---------------

+---------------+---------------------------------------------+-----------------------------------------------+
|    Context    |    Error                                    |    Resolution                                 |
+===============+=============================================+===============================================+
| any           | Code issue                                  | Upgrade to latest build:                      |
|               |                                             | ``pip install mindmeld -U``                   |
+---------------+---------------------------------------------+-----------------------------------------------+
| Elasticsearch | ``KnowledgeBaseConnectionError``            | Run ``curl localhost:9200`` to                |
|               |                                             | verify that Elasticsearch is                  |
|               |                                             | running.                                      |
|               |                                             | If you're using Docker, you can               |
|               |                                             | increase memory to 4GB from                   |
|               |                                             | *Preferences | Advanced*.                     |
+---------------+---------------------------------------------+-----------------------------------------------+
| Numerical     | ``OS is incompatible with duckling binary`` | Run the numerical parser via                  |
| Parser        |                                             | Docker.                                       |
|               |                                             | :ref:`More details <duckling>`.               |
+---------------+---------------------------------------------+-----------------------------------------------+
| Blueprints    | ``ValueError: Unknown                       | Run the mindmeld_init.sh found                |
|               | error fetching archive`` when running       | :ref:`here <getting_started_virtualenv_setup>`|
|               | ``mm.blueprint(bp_name)``                   |                                               |
+---------------+---------------------------------------------+-----------------------------------------------+
| Blueprints    | ``JSONDecodeError: Expecting value: line 1  | Remove the cached version of the app:         |
|               | column 1 (char 0)``                         | ``rm ~/.mindmeld/blueprints/bp_name`` and     |
|               |                                             | re-download the blueprint.                    |
+---------------+---------------------------------------------+-----------------------------------------------+

Environment Variables
---------------------

.. _parallel_processing:

MM_SUBPROCESS_COUNT
^^^^^^^^^^^^^^^^^^^
MindMeld supports parallel processing via process forking when the input is a list of queries, as is the case when :ref:`leveraging n-best ASR transcripts for entity resolution <nbest_lists>`. Set this variable to an integer value to adjust the number of subprocesses. The default is ``4``. Setting it to ``0`` will turn off the feature.
