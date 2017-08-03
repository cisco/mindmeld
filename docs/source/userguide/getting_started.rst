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


Automated Setup
^^^^^^^^^^^^^^^^^

If you are on a Mac OS machine, you can install dependencies for MindMeld Workbench and 
setup configuration files by running this command:

.. code-block:: shell

  bash -c "$(curl -s  https://mindmeld.com/docs/scripts/mmworkbench_init.sh)"


Notes:

#. You will be prompted to enter your mindmeld.com username and password.
#. This creates two configuration files: ``~/.pip/pip.conf`` and ``~/.mmworkbench/config``. Previous files will be overwritten.
#. If you enter an incorrect password, run the command again.


The following components will be installed:

- brew
- pyenv
- pyenv-virtualenv
- Java 8
- ElasticSearch

For other platforms, or if you run into issues, you can try the :ref:`manual setup <getting_started_manual_setup>`.


Install Workbench
-----------------

Configure a Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To prepare to install Workbench in an isolated environment using pyenv and virtualenv, run these or equivalent commands:

.. code-block:: console

  # this is the parent of your Workbench project folder.
  mkdir workbench-development
  cd $_

  # install Python 3.6.1
  pyenv install 3.6.1

  # create a new virtual environment using Python 3.6.1
  pyenv virtualenv 3.6.1 workbench

  # automatically activate the environment upon entering this directory
  pyenv local workbench


pip install
^^^^^^^^^^^^

Now that your environment is set up, you can install MindMeld Workbench just as you would any other
Python package. This may take a few minutes.

.. code-block:: console

  $ pip install mmworkbench

If the following command returns no error, the installation was successful:

.. code-block:: console

    $ mmworkbench

Start the Numerical Parser
^^^^^^^^^^^^^^^^^^^^^^^^^^

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


Troubleshooting
---------------



.. _getting_started_manual_setup:

Manual Setup
^^^^^^^^^^^^

Python
"""""""

Since Workbench is a Python-based machine learning library, you need Python on your system.

Run ``python --version`` to verify that Python is installed.

In Workbench,

 - Python 3.4 and newer are actively supported
 - The latest version of Python 3 is recommended
 - Python 2.7+ should work, but is deprecated

You can:

 - use `pyenv <https://github.com/pyenv/pyenv>`_ to manage multiple versions of Python
 - obtain Python at `python.org <https://www.python.org/>`_  if it is not already installed

Pyenv and Virtualenv
""""""""""""""""""""""""""""

It is strongly recommended that you install Workbench in an isolated environment. This way, you can work
on different projects without having conflicting library versions, and keep Workbench separate from your other work.

One solution is to use `virtualenv with pyenv <https://github.com/pyenv/pyenv-virtualenv>`_.

On Mac OS systems with `homebrew <https://brew.sh/>`_, 

 - install pyenv with: ``brew install pyenv``.
 - install virtualenv with: ``brew install pyenv-virtualenv``. You'll also have to append these lines to your bash profile:

.. code-block:: console
 
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"

If your system cannot run virtualenv with pyenv, you will need to find alternatives to some instructions in this document.


Java 8
"""""""

Workbench requires Java 8 or newer.

Run ``java -version`` to verify that Java 8 or newer is installed. 

If the command fails, or your Java version begins with 1.7, install Java 8.

On Mac OS systems with `homebrew <https://brew.sh/>`_, run ``brew cask install java``

Visit `java.com <https://www.java.com/inc/BrowserRedirect1.jsp?locale=en>`_ for detailed instructions.

Elasticsearch
""""""""""""""

Workbench requires Elasticsearch 5.0 or newer.

`Elasticsearch <https://www.elastic.co/products/elasticsearch>`_ is a highly scalable open-source
full-text search and analytics engine. It can be installed locally or you can use a remote instance if you have access to one.

On Mac OS systems with `homebrew <https://brew.sh/>`_, install and run Elasticsearch with these commands:

.. code-block:: console

  brew install elasticsearch
  brew services start elasticsearch

For other systems, or for more information on configuring Elasticsearch, go
`here <https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html>`_.


