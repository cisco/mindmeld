Getting Started
===============

These instructions explain how to install MindMeld on a Unix-based system and set up your first MindMeld project. Users of other operating systems like Windows can use Docker to get started.

.. note::

  MindMeld requires Python 3.6 or 3.7.


Install MindMeld
----------------

You must choose the mechanism by which you install MindMeld. The supported choices are:

  - :ref:`Docker <getting_started_docker_setup>`
  - :ref:`virtualenv <getting_started_virtualenv_setup>`

If you're going to be using MindMeld often, **we recommend you do the virtualenv installation** and setup all dependencies locally. That will provide the optimal performance and experience. But if you want to get a taste of MindMeld with minimal effort, you can get started quickly using Docker.


.. _getting_started_docker_setup:

Install with Docker
^^^^^^^^^^^^^^^^^^^

In the Docker setup, we use Docker only for running MindMeld dependencies, namely, Elasticsearch and the Numerical Parser service. The developer will then use their local machine to run the MindMeld project.

1. Pull and run Docker container
""""""""""""""""""""""""""""""""

.. note::

  The following instruction references our docker container with Elasticsearch 7 which is a requirement for leveraging semantic embedding. ``mindmeldworkbench/dep:latest`` is still available with an older version of Elasticsearch.

#. First, `install Docker <https://www.docker.com/community-edition#/download>`_, and run it.
#. Then, open a terminal (shell) and run these commands:

.. code-block:: shell

   docker pull mindmeldworkbench/dep:es_7
   docker run -ti -d -p 0.0.0.0:9200:9200 -p 0.0.0.0:7151:7151 -p 0.0.0.0:9300:9300 mindmeldworkbench/dep:es_7

2. Install prerequisites
""""""""""""""""""""""""

Next, we install ``python``, ``pip`` and ``virtualenv`` on the local machine using a script we made. These are pre-requisite libraries needed for most python projects. Currently, the script works for Mac and Ubuntu 16/18 operating systems:

.. code-block:: shell

  bash -c "$(curl -s  https://raw.githubusercontent.com/cisco/mindmeld/master/scripts/mindmeld_lite_init.sh)"

If you encounter any issues, see :ref:`Troubleshooting <getting_started_troubleshooting>`.

3. Set up a virtual environment
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


4. Install the MindMeld package
"""""""""""""""""""""""""""""""

Now that your environment is set up, you can install MindMeld just as you would any other Python package. This may take a few minutes.

.. code-block:: shell

  pip install mindmeld

If you see errors here, make sure that your ``pip`` package is up to date and your connection is active. If the error is a dependency error (tensorflow, scikitlearn, etc), you can try to install/reinstall the specific dependency before installing MindMeld.

To verify your setup is good, run this command. If there is no error, the installation was successful:

.. code-block:: shell

  mindmeld

A few of our dependencies are optional since they are not required for the core NLU functions. If you are interested in developing for Cisco Webex Teams, you can install the Webex Teams dependency by typing in the shell:

.. code-block:: shell

  pip install mindmeld[bot]

If you are interested in using the LSTM entity recognizer, you will need to install the Tensorflow dependency:

.. code-block:: shell

  pip install mindmeld[tensorflow]

If you are interested in leveraging pretrained BERT embedders for question answering, you will need to install the following dependency:

.. code-block:: shell

  pip install mindmeld[bert]

.. _getting_started_virtualenv_setup:

Install with virtualenv
^^^^^^^^^^^^^^^^^^^^^^^

1. Install prerequisites
""""""""""""""""""""""""

On a Ubuntu 16/18 machine, you can install the dependencies for MindMeld and set up the necessary configuration files with the `mindmeld_init.sh script <https://raw.githubusercontent.com/cisco/mindmeld/master/scripts/mindmeld_init.sh>`_.

.. note::

   The script installs the following components after a confirmation prompt: ``docker``, ``python3.6``, ``python-pip``, ``virtualenv`` and Elasticsearch 7.8.

If you are using a Ubuntu 16/18 machine, when you're ready to go, open a terminal (shell) and run this command:

.. code-block:: shell

  bash -c "$(curl -s  https://raw.githubusercontent.com/cisco/mindmeld/master/scripts/mindmeld_init.sh)"

If you encounter any issues, see :ref:`Troubleshooting <getting_started_troubleshooting>`.

For macOS users, a recent (April 16th 2019) change in licensing policy of Java prevents us from creating an automatic script to download and run it. Java is necessary for Elasticsearch 7.8 to run. Assuming you have Oracle Java or OpenJDK installed, please download the following libraries:

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
| Elasticsearch |  See instructions below to download and run Elasticsearch 7.8 natively or using docker                 |
+---------------+--------------------------------------------------------------------------------------------------------+

Native onboarding:

.. code-block:: shell

   curl https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.8.0-darwin-x86_64.tar.gz -o elasticsearch-7.8.0.tar.gz
   tar -zxvf elasticsearch-7.8.0.tar.gz
   ./elasticsearch-7.8.0/bin/elasticsearch


Docker onboarding:

.. code-block:: shell

  sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:7.8.0 && sudo docker run -ti -d -p 0.0.0.0:9200:9200 -p 0.0.0.0:9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.8.0


Ubuntu:

+---------------+-------------------------------------------------------------------------------+
|    Component  |    Command                                                                    |
+===============+===============================================================================+
| python3       |  ``sudo apt-get install python3.6``                                           |
+---------------+-------------------------------------------------------------------------------+
| pip           |  ``sudo apt install python-pip``                                              |
+---------------+-------------------------------------------------------------------------------+
| virtualenv    |  ``sudo apt install virtualenv``                                              |
+---------------+-------------------------------------------------------------------------------+
| Elasticsearch |  See instructions below to download Elasticsearch 7.8 natively or using docker|
+---------------+-------------------------------------------------------------------------------+

Native onboarding:

.. code-block:: shell

   wget -O elasticsearch-7.8.0.tar.gz  https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.8.0-linux-x86_64.tar.gz
   tar -zxvf elasticsearch-7.8.0.tar.gz
   ./elasticsearch-7.8.0/bin/elasticsearch


Docker onboarding:

.. code-block:: shell

  sudo docker pull docker.elastic.co/elasticsearch/elasticsearch:7.8.0 && sudo docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.8.0


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

If you see errors here, make sure that your ``pip`` package is up to date and your connection is active. If the error is a dependency error (tensorflow, scikitlearn, etc), you can try to install/reinstall the specific dependency before installing MindMeld.

To verify your setup is good, run this command. If there is no error, the installation was successful:

.. code-block:: shell

  mindmeld

A few of our dependencies are optional since they are not required for the core NLU functions. If you are interested in developing for Cisco Webex Teams, you can install the Webex Teams dependency by typing in the shell:

.. code-block:: shell

  pip install mindmeld[bot]

If you are interested in using the LSTM entity recognizer, you will need to install the Tensorflow dependency:

.. code-block:: shell

  pip install mindmeld[tensorflow]


.. _duckling:

4. Start the numerical parser
"""""""""""""""""""""""""""""

MindMeld uses a Haskell-based numerical parser for detecting certain numeric expressions like times, dates, and quantities in user queries. The numerical parser is locally started on default port 7151 with this command:

.. code-block:: shell

  mindmeld num-parse --start

You can start the numerical parser on a different port using the ``-p`` command-line flag, for example, ``mindmeld num-parse --start -p 9000`` starts the service on port 9000.
If you encounter an error like ``OS is incompatible with duckling executable``, it means that your operating system is
not compatible with the pre-compiled numerical parser binary distributed with MindMeld. You instead need to run the
numerical parser using Docker as shown below.

.. code-block:: shell

   docker pull mindmeldworkbench/duckling:master && docker run -ti -d -p 0.0.0.0:7151:7151 mindmeldworkbench/duckling:master


.. note::

   The numerical parser is an optional component of MindMeld. To turn off the numerical parser, in ``config.py``, set ``NLP_CONFIG = {"system_entity_recognizer": {}}``.


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


Upgrade MindMeld
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

.. _ES docs: https://www.elastic.co/guide/en/elasticsearch/reference/current/indices-delete-index.html

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
| Elasticsearch | ``KnowledgeBaseError``                      | If error is due to maximum shards open, run   |
|               |                                             | ``curl -XDELETE 'http://localhost:9200/_all'``|
|               |                                             | to delete all existing shards from all apps.  |
|               |                                             | Alternatively, to delete an app specific      |
|               |                                             | indices, run                                  |
|               |                                             | ``curl -XDELETE 'localhost:9200/<app_name>*'``|
|               |                                             | For example, to delete indices of             |
|               |                                             | a hr_assistant application, one can run       |
|               |                                             | ``curl -XDELETE localhost:9200/hr_assistant*``|
|               |                                             | For more details, see `ES docs`_              |
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

MM_SYS_ENTITY_REQUEST_TIMEOUT
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This variable sets the request timeout value for the :ref:`system entity recognition service <configuring-system-entities>` . The default float value is ``1.0 seconds``.

MM_QUERY_CACHE_IN_MEMORY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MindMeld maintains a cache of preprocessed training examples to speed up the training process.  This variable controls whether the query cache is maintained in-memory or only on disk.  Setting this to ``0`` will save memory during training, but will negatively impact performance for configurations with slow disk access.  Defaults to ``1``.

MM_QUERY_CACHE_WRITE_SIZE
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
This variable works in conjunction with ``MM_QUERY_CACHE_IN_MEMORY``.  If the in-memory cache is enabled, this variable sets the number of in-memory cached examples that are batched up before they are synchronized to disk.  This allows for better write performance by doing bulk rather than individual writes.  Defaults to ``1000``.

MM_CRF_FEATURES_IN_MEMORY
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The CRF model used by MindMeld can generate very large feature sets.  This can cause high memory usage for some datasets.  This variable controls whether these feature sets are stored in-memory or on disk. Defaults to ``1``
