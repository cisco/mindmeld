Getting Started
===============

These instructions install MindMeld Workbench on your personal machine and set up your first Workbench project.

Pre-requisites
--------------

For this release of MindMeld Workbench, you should

 - be a seasoned Python developer with machine learning (ML) knowhow and experience with one or more popular ML frameworks or libraries

 - have a username and password for the MindMeld Learning Center

The rest of this section describes how to install software pre-requisites.

Optional: Docker
^^^^^^^^^^^^^^^^

If you're going to be using Workbench often, we recommend you do the full install and setup all dependencies locally. That will provide the optimal performance and experience.
But if you want to get a quick taste of workbench, you can get setup with a provided ``Dockerfile`` and these steps:

#. First, `install Docker <https://www.docker.com/community-edition#/download>`_, and run it.
#. Then, run these commands:

.. code-block:: shell

  curl -s https://mindmeld.com/docker/wb3.tar.gz | tar xzvf -
  cd wb3
  ./buildme.sh
  ./runme.sh

You will be prompted for your mindmeld.com username and password. The build will take a few minutes. It setups all dependencies and drops you inside a container.

Proceed to :ref:`Begin New Project <getting_started_begin_new_project>`.

.. _getting_started_automated_setup:

Automated Setup
^^^^^^^^^^^^^^^^^

If you are on a Mac OS machine, you can install dependencies for MindMeld Workbench and 
setup configuration files with the `mmworkbench_init.sh script <https://mindmeld.com/docs/scripts/mmworkbench_init.sh>`_.

A few things to note before you run the script:

- The script installs the following components will be installed after a confirmation prompt: ``brew``, ``python``, ``pip``, ``virtualenv``, Java 8 and ElasticSearch. 
- You will be prompted to enter your mindmeld.com username and password. 
- Two configuration files will be created: ``~/.pip/pip.conf`` and ``~/.mmworkbench/config``. Previous files are overwritten.

When you're ready to go, run this command:

.. code-block:: shell

  bash -c "$(curl -s  https://mindmeld.com/docs/scripts/mmworkbench_init.sh)"


If you encounter any issues, look in :ref:`Troubleshooting <getting_started_troubleshooting>`.


Install Workbench
-----------------

Virtual Environment
^^^^^^^^^^^^^^^^^^^^

To prepare to install Workbench in an isolated environment using ``virtualenv``, follow the following steps.

- Create your project folder and navigate to it:

.. code-block:: console

  $ mkdir ~/mmworkbench
  $ cd $_

- Setup a virtual environment by running one of the following commands:

.. code-block:: console

  $ virtualenv .             # for Python 2.7
  $ virtualenv -p python3 .  # for Python 3.x

- Activate the virtual environment:

.. code-block:: console

  $ virtualenv bin/activate


Later, when you're done working with MindMeld Workbench, you can deactivate the virtual environment with the ``deactivate`` command.

.. code-block:: console

  $ deactivate


pip install
^^^^^^^^^^^^

Now that your environment is set up, you can install MindMeld Workbench just as you would any other
Python package. This may take a few minutes.

.. code-block:: console

  $ pip install mmworkbench

If you see errors here, you likely entered incorrect credentials during :ref:`Setup <getting_started_automated_setup>`. Make sure you use your credentials for the MindMeld Learning Center.

To verify your setup is good, run this command. If there is no error, the installation was successful:

.. code-block:: console

    $ mmworkbench

Numerical Parser
^^^^^^^^^^^^^^^^^

Start the numerical parser with this command:

.. code-block:: console

  $ mmworkbench num-parse --start

The numerical parser is a critical component that relies on Java 8. **Do not skip this step**.

.. _getting_started_begin_new_project:

Begin New Project
-----------------

With the setup out of the way, you are now ready to get your feet wet. MindMeld Workbench is designed so you can
keep using the tools and coding patterns that are familiar to you. Some of the very basic operations can be performed in
your command-line shell using the ``mmworkbench`` command. But to really take advantage of the power of Workbench,
the Python shell is where all the action is at.


Command Line
^^^^^^^^^^^^

To try out the :doc:`Food Ordering blueprint<../blueprints/food_ordering>`, run these commands on the command line:

.. code-block:: console

  $ mmworkbench blueprint food_ordering
  $ cd $_
  $ python app.py build   # this will take a few minutes
  $ python app.py converse
 Loading intent classifier: domain='ordering'
 ...
 You:

The ``converse`` command loads the machine learnings models and starts an interactive session with the "You:" prompt.
Here you can enter your own input and get an immediate response back. Try "hi", for example, and see what you get.


Python Shell
^^^^^^^^^^^^

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


Upgrade Workbench
-----------------

To upgrade to the latest version of Workbench, run ``pip install mmworkbench --upgrade``

Make sure to run this regularly to stay on top of the latest bug fixes and feature releases.


Command-Line Interfaces
-----------------------

MindMeld Workbench has two command-line interfaces for some of the common workflow tasks you'll be doing often:

#. mmworkbench
#. python app.py

Builtin help is available with the standard `-h` flag.

mmworkbench
^^^^^^^^^^^

The command-line interface (CLI) for MindMeld Workbench can be accessed with the `mmworkbench` command.
This is most suitable for use in an app-agnostic context.

The commands available are:

#. ``blueprint`` : Downloads all the training data for an existing :doc:`blueprint <../blueprints/overview>` and sets it up for use in your own project.
#. ``num-parse`` : Starts or stops the numerical parser service.

Also, there is a special ``template`` blueprint that sets up the scaffolding for a blank project:

.. code-block:: console

  $ mmworkbench blueprint template myapp

Similarly, in the python shell, the ``template`` blueprint sets up the scaffolding for a blank project:

.. code-block:: python

    import mmworkbench as wb
    wb.configure_logs()    
    wb.blueprint('template', 'my_app')


python app.py
^^^^^^^^^^^^^

When you're in the context of a specific app, `python app.py` is more appropriate to use.

The commands available are:

#. ``build`` : Builds the artifacts and machine learning models and persists them.
#. ``clean`` : Deletes the generated artifacts and takes the system back to a pristine state.
#. ``converse`` : Begins an interactive conversational session with the user at the command line.
#. ``load-kb`` : Populates the knowledge base.
#. ``run`` : Starts the workbench service as a REST API.


Configure Logging
------------------

Workbench adheres to the standard `Python logging mechanism <https://docs.python.org/3/howto/logging.html>`_. 
The default logging level is ``WARNING``, which can be overridden with a config file or from code. 
The INFO logging level can be useful to see what's going on:

.. code-block:: python
  
  import logging
  logging.getLogger('mmworkbench’).setLevel(logging.INFO)

configure_logs()
^^^^^^^^^^^^^^^^

There is a handy ``configure_logs()`` function available that wraps this and accepts 2 parameters: 

#. `format message <https://docs.python.org/3/howto/logging.html#changing-the-format-of-displayed-messages>`_
#. `logging level <https://docs.python.org/3/howto/logging.html#logging-levels>`_: in increasing order of severity, they are ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR`` and ``CRITICAL``.

The method signature is:

.. code-block:: python
 
   configure_logs(format="%(message)s", level=logging.WARNING)


Sample Code
^^^^^^^^^^^^
  
.. code-block:: python

  import mmworkbench as wb  
  wb.configure_logs()


.. _getting_started_troubleshooting:

Troubleshooting
---------------


+-------------+---------------------------+-----------------------------------+
|    Context  |    Error                  |    Resolution                     |
+=============+===========================+===================================+
| pip install | Could not find a version  | Verify your credentials for the   |
|             | that satisfies the        | MindMeld Learning Center.         |
|             | requirement mmworkbench   |                                   |
+-------------+---------------------------+-----------------------------------+
