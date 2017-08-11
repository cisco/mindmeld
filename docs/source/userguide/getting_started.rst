Getting Started
===============

These instructions explain how to install MindMeld Workbench on Mac OS X and set up your first Workbench project. Platforms other than Mac OS may work too but are not currently supported.

.. note::

   For this release of MindMeld Workbench, you should

   - be a seasoned Python developer with machine learning (ML) knowhow and experience with one or more popular ML frameworks or libraries

   - have a Developer Token for the MindMeld Learning Center.

Installation
------------

You must choose the mechanism by which you install Workbench. The supported choices are:

  - :ref:`Docker <getting_started_docker_setup>`
  - :ref:`virtualenv <getting_started_virtualenv_setup>`

If you're going to be using Workbench often, **we recommend you do the virtualenv installation** and setup all dependencies locally. That will provide the optimal performance and experience. But if you want to get a taste of Workbench with minimal effort, you can get started quickly using Docker.


.. _getting_started_docker_setup:

Install with Docker
^^^^^^^^^^^^^^^^^^^

The ``Dockerfile`` provided by MindMeld contains Workbench and all its dependencies. Follow these steps to get started using Docker:

#. First, `install Docker <https://www.docker.com/community-edition#/download>`_, and run it.
#. Then, open a terminal (shell) and run this command:

.. code-block:: shell

  bash -c "$(curl -s  https://mindmeld.com/docs/scripts/docker_mmworkbench_init.sh)"

You will be prompted for your Developer Token. The build will take a few minutes. It sets up all dependencies and drops you inside a container.

If you encounter any issues, see :ref:`Troubleshooting <getting_started_troubleshooting>`.

Proceed to :ref:`Begin New Project <getting_started_begin_new_project>`.


.. _getting_started_virtualenv_setup:

Install with virtualenv
^^^^^^^^^^^^^^^^^^^^^^^

1. Install prerequisites
""""""""""""""""""""""""

On a Mac OS machine, you can install the dependencies for MindMeld Workbench and set up the necessary configuration files with the `mmworkbench_init.sh script <https://mindmeld.com/docs/scripts/mmworkbench_init.sh>`_.

.. note:: 
   
   A few things to note before you run the script:

   - The script installs the following components after a confirmation prompt: ``brew``, ``python``, ``pip``, ``virtualenv``, Java 8 and Elasticsearch. 
   - You will be prompted to enter your Developer Token.
   - Two configuration files will be created: ``~/.pip/pip.conf`` and ``~/.mmworkbench/config``. **Previous files are overwritten.**

When you're ready to go, open a terminal (shell) and run this command:

.. code-block:: shell

  bash -c "$(curl -s  https://mindmeld.com/docs/scripts/mmworkbench_init.sh)"

If you encounter any issues, see :ref:`Troubleshooting <getting_started_troubleshooting>`.

Here are the commands run by the script to install the required components:

+---------------+--------------------------------------------------------------------------------------------------------+
|    Component  |    Command                                                                                             |
+===============+========================================================================================================+
| brew          |  ``/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"``|
+---------------+--------------------------------------------------------------------------------------------------------+
| python        |  ``brew install python``                                                                               |
+---------------+--------------------------------------------------------------------------------------------------------+
| pip           |  ``sudo -H easy_install pip``                                                                          |
+---------------+--------------------------------------------------------------------------------------------------------+
| virtualenv    |  ``sudo -H pip install --upgrade virtualenv``                                                          |
+---------------+--------------------------------------------------------------------------------------------------------+
| Java 8        |  ``brew tap caskroom/cask && brew cask install java``                                                  |  
+---------------+--------------------------------------------------------------------------------------------------------+
| Elasticsearch |  ``brew install elasticsearch && brew services start elasticsearch``                                   |
+---------------+--------------------------------------------------------------------------------------------------------+


2. Set up a virtual environment
"""""""""""""""""""""""""""""""

To prepare an isolated environment for Workbench installation using ``virtualenv``, follow the following steps.

- Create a folder for containing all your Workbench projects, and navigate to it:

.. code-block:: console

  mkdir my_wb_workspace
  cd my_wb_workspace

- Setup a virtual environment by running one of the following commands:

.. code-block:: console

  $ virtualenv -p python3 .  # for Python 3.x (recommended)
  $ virtualenv .             # for Python 2.7

- Activate the virtual environment:

.. code-block:: console

  source bin/activate


Later, when you're done working with MindMeld Workbench, you can deactivate the virtual environment with the ``deactivate`` command.

.. code-block:: console

  deactivate


3. Install the Workbench package
""""""""""""""""""""""""""""""""

Now that your environment is set up, you can install MindMeld Workbench just as you would any other Python package. This may take a few minutes.

.. code-block:: console

  pip install mmworkbench

If you see errors here, you likely entered incorrect credentials during :ref:`Setup <getting_started_virtualenv_setup>`. Make sure you use your credentials for the MindMeld Learning Center.

To verify your setup is good, run this command. If there is no error, the installation was successful:

.. code-block:: console

  mmworkbench


4. Start the numerical parser
"""""""""""""""""""""""""""""

Workbench uses a Java-based numerical parser for detecting certain numeric expressions like times, dates, and quantities in user queries. Start the numerical parser with this command:

.. code-block:: console

  mmworkbench num-parse --start

.. warning::

   The numerical parser is a critical component that Workbench relies on. **Do not skip this step**.


.. _getting_started_begin_new_project:

Begin New Project
-----------------

With the setup out of the way, you are now ready to get your feet wet. You can proceed in one of two ways:

#. Try out a :ref:`blueprint application <getting_started_blueprint>`. This is the **recommended approach** for beginners to familiarize themselves with Workbench. This is also a good starting point if your use case matches one of the :doc:`blueprint scenarios <../blueprints/overview>`.

#. Start a :ref:`brand new project <getting_started_template>`. This is the approach to take if your specific use case isn't covered by an existing blueprint, or if you prefer to build out your app from scratch.

MindMeld Workbench is designed so you can keep using the tools and coding patterns that are familiar to you. Some of the very basic operations can be performed in your command-line shell using the ``mmworkbench`` command. But to really take advantage of the power of Workbench, the Python shell is where all the action is at. The examples in this section are accompanied by code samples from both shells.


.. _getting_started_blueprint:

Start with a blueprint
^^^^^^^^^^^^^^^^^^^^^^

.. note::

   Blueprints are simple example apps that are intentionally limited in scope. They provide you with a baseline to bootstrap upon for common conversational use cases. To improve upon them and convert them into production-quality apps, follow the exercises in the :doc:`individual blueprint sections <../blueprints/overview>`.


Using the command-line
""""""""""""""""""""""

To try out the :doc:`Food Ordering blueprint<../blueprints/food_ordering>`, run these commands on the command line:

.. code-block:: console

  $ mmworkbench blueprint food_ordering
  $ cd food_ordering
  $ python app.py build   # this will take a few minutes
  $ python app.py converse
 Loading intent classifier: domain='ordering'
 ...
 You:

The ``converse`` command loads the machine learning models and starts an interactive session with the "You:" prompt.
Here you can enter your own input and get an immediate response back. Try "hi", for example, and see what you get.


Using the Python shell
""""""""""""""""""""""

To try out the :doc:`Home Assistant blueprint<../blueprints/home_assistant>`, run these commands in your Python shell:

.. code-block:: python

    import mmworkbench as wb
    wb.configure_logs()
    blueprint = 'home_assistant'
    wb.blueprint(blueprint)

    from mmworkbench.components import NaturalLanguageProcessor
    nlp = NaturalLanguageProcessor(blueprint)
    nlp.build()

    from mmworkbench.components.dialogue import Conversation
    conv = Conversation(nlp=nlp, app_path=blueprint)
    conv.say('Hello!')


Workbench provides several different blueprint applications to support many common use cases for
conversational applications. See :doc:`Workbench Blueprints<../blueprints/overview>` for more usage examples.


.. _getting_started_template:

Start with a new project
^^^^^^^^^^^^^^^^^^^^^^^^

There is a special ``template`` blueprint that sets up the scaffolding for a blank project. The example below creates a new empty project in a local folder named ``my_app``.

Using the command-line
""""""""""""""""""""""

.. code-block:: console

  mmworkbench blueprint template myapp


Using the Python shell
""""""""""""""""""""""

.. code-block:: python

  import mmworkbench as wb
  wb.configure_logs()
  wb.blueprint('template', 'my_app')

The :doc:`Step-By-Step guide <../quickstart/00_overview>` walks through the methodology for building conversational apps using Workbench.


Upgrade Workbench
-----------------

To upgrade to the latest version of Workbench, run ``pip install mmworkbench --upgrade``

Make sure to run this regularly to stay on top of the latest bug fixes and feature releases.


Command-Line Interfaces
-----------------------

MindMeld Workbench has two command-line interfaces for some of the common workflow tasks you'll be doing often:

#. ``mmworkbench``
#. ``python app.py``

Built-in help is available with the standard :option:`-h` flag.

mmworkbench
^^^^^^^^^^^

The command-line interface (CLI) for MindMeld Workbench can be accessed with the ``mmworkbench`` command.
This is most suitable for use in an app-agnostic context.

The commands available are:

#. ``blueprint`` : Downloads all the training data for an existing :doc:`blueprint <../blueprints/overview>` and sets it up for use in your own project.
#. ``num-parse`` : Starts or stops the numerical parser service.


python app.py
^^^^^^^^^^^^^

When you're in the context of a specific app, ``python app.py`` is more appropriate to use.

The commands available are:

#. ``build`` : Builds the artifacts and machine learning models and persists them.
#. ``clean`` : Deletes the generated artifacts and takes the system back to a pristine state.
#. ``converse`` : Begins an interactive conversational session with the user at the command line.
#. ``load-kb`` : Populates the knowledge base.
#. ``run`` : Starts the Workbench service as a REST API.


Configure Logging
------------------

Workbench adheres to the standard `Python logging mechanism <https://docs.python.org/3/howto/logging.html>`_. 
The default logging level is ``WARNING``, which can be overridden with a config file or from code. 
The ``INFO`` logging level can be useful to see what's going on:

.. code-block:: python
  
  import logging
  logging.getLogger('mmworkbench’).setLevel(logging.INFO)

There is a handy ``configure_logs()`` function available that wraps this and accepts 2 parameters: 

#. :data:`format`: The `logging format <https://docs.python.org/3/howto/logging.html#changing-the-format-of-displayed-messages>`_.
#. :data:`level`: The `logging level <https://docs.python.org/3/howto/logging.html#logging-levels>`_.

Here's an example usage:
  
.. code-block:: python

  import mmworkbench as wb  
  wb.configure_logs()


.. _getting_started_troubleshooting:

Troubleshooting
---------------

+---------------+----------------------------------+-----------------------------------+
|    Context    |    Error                         |    Resolution                     |
+===============+==================================+===================================+
| pip install   | Could not find a version         | Verify your credentials for the   |
|               | that satisfies the               | MindMeld Learning Center.         |
|               | requirement mmworkbench          |                                   |
+---------------+----------------------------------+-----------------------------------+
| any           | Code issue                       | Upgrade to latest build:          |
|               |                                  | ``pip install mmworkbench -U``    |
+---------------+----------------------------------+-----------------------------------+
| Elasticsearch | ``KnowledgeBaseConnectionError`` | Run ``curl localhost:9200`` to    |
|               |                                  | verify that Elasticsearch is      |
|               |                                  | running.                          |
|               |                                  | If you're using Docker, you can   |
|               |                                  | increase memory to 4GB from       |
|               |                                  | *Preferences | Advanced*.         |
+---------------+----------------------------------+-----------------------------------+
