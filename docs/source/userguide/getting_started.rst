Getting Started
===============

These instructions install MindMeld Workbench on your personal machine and set up your first Workbench project.

Pre-requisites
--------------

For this release of MindMeld Workbench, you should

 - be a seasoned Python developer with machine learning (ML) knowhow and experience with one or more popular ML frameworks or libraries

 - have a username and password for the MindMeld Learning Center

The rest of this section describes how to install software pre-requisites.

Homebrew on Mac OS
^^^^^^^^^^^^^^^^^^

If you are on a Mac OS machine, it is recommended that you install `homebrew <https://brew.sh/>`_, a package manager that some of these instructions use.

Run ``brew --version`` to verify that homebrew is installed.

This is not a requirement, but if your system cannot run homebrew you will need to find alternatives to some instructions in this document.

Python
^^^^^^

Since Workbench is a Python-based machine learning library, you need Python on your system.

Run ``python --version`` to verify that Python is installed.

In Workbench,

 - Python 3.4 and newer are actively supported
 - The latest version of Python 3 is recommended
 - Python 2.7+ should work, but is deprecated

You can

 - use `pyenv <https://github.com/pyenv/pyenv>`_ to manage multiple versions of Python
 - obtain Python at `python.org <https://www.python.org/>`_  if it is not already installed

Pyenv and Virtualenv
^^^^^^^^^^^^^^^^^^^^

It is strongly recommended that you install Workbench in an isolated environment. This way, you can work
on different projects without having conflicting library versions, and keep Workbench separate from your other work.

One solution is to use `virtualenv with pyenv <https://github.com/pyenv/pyenv-virtualenv>`_.

Run ``pyenv --version`` to verify that Pyenv is installed.

Run ``virtualenv --version`` to verify that Virtualenv is installed.

Again, this is not a requirement, but if your system cannot run pyenv you will need to find alternatives to some instructions in this document.

Java 8
^^^^^^^

Workbench requires Java 8 or newer.

Run ``java -version`` to verify that Java 8 or newer is installed. 

If the command fails, or your Java version begins with 1.7, install Java 8.

 - On Mac OS systems with `homebrew <https://brew.sh/>`_, run ``brew cask install java``

Visit `java.com <https://www.java.com/inc/BrowserRedirect1.jsp?locale=en>`_ for detailed instructions.

Elasticsearch
^^^^^^^^^^^^^

Workbench requires Elasticsearch 5.0 or newer. Version 5.5.x is recommended.

`Elasticsearch <https://www.elastic.co/products/elasticsearch>`_ is a highly scalable open-source
full-text search and analytics engine. It can be installed locally or you can use a remote instance if you have access to one.

  - On Mac OS systems with `homebrew <https://brew.sh/>`_, run ``brew install elasticsearch``

For other systems, or for more information on configuring Elasticsearch, go
`here <https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html>`_.

After Elasticsearch has been configured simply run ``elasticsearch`` to start the process.

Prepare your system
---------------------

The configuration steps described in this section are all either required or strongly recommended.

Configure a Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To prepare to install Workbench in an isolated environment using pyenv and virtualenv, run these or similar commands:

.. code-block:: console

  # this is the parent of your Workbench project folder. It can be anywhere you want.
  mkdir workbench-development
  cd $_

  # install Python 3.6.1
  pyenv install 3.6.1

  # create a new virtual environment using Python 3.6.1
  pyenv virtualenv 3.6.1 workbench

  # automatically activate the environment upon entering this directory
  pyenv local workbench

Configure Pip and Workbench
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Workbench installation relies on ``pip``, a Python packaging system included by default with the Python binary installers.

Run ``pip --version`` to verify that pip is installed.

 - You should have pip version 8 or 9

 - If you need upgrade pip module, run ``pip install --upgrade pip``


Automatic Configuration
"""""""""""""""""""""""

You can configure pip and workbench easily by running these commands in a command-line terminal:

.. highlight:: python
   :linenothreshold: 2

.. code-block:: shell

  export USERNAME=<Username>
  curl -s https://mindmeld.com/docs/scripts/mmworkbench_init.sh > mmworkbench_init.sh
  source mmworkbench_init.sh


Notes:

#. Remember to modify the ``Username`` to your actual mindmeld.com username.
#. You will be prompted to enter your mindmeld.com password.
#. This creates two configuration files: ``~/.pip/pip.conf`` and ``~/.mmworkbench/config``. Previous files will be overwritten.
#. If you enter an incorrect password, run the last command again.


Proceed to :ref:`Install Workbench <getting_started_install_workbench>`.


Manual Configuration
""""""""""""""""""""

Pip
'''

The next two steps are for Mac OS. If you need information about configuring pip on a different OS, see the `pip documentation <http://pip.readthedocs.io/en/latest/user_guide/#configuration>`_.

Run ``ls -l ~/.pip`` to verify that there is a ``~/.pip`` folder on your system.

 - Create the folder if it does not exist

Run ``ls -l ~/.pip/pip.conf`` to verify that there is a ``~/.pip/pip.conf`` file on your system.

 - Create the file if it does not exist

Add the following lines to your ``pip.conf`` file, substituting your username and password.

.. code-block:: text

  [global]
  extra-index-url = https://{YOUR_USERNAME}:{YOUR_PASSWORD}@mindmeld.com/pypi/
  trusted-host = mindmeld.com

These configuration changes enable pip to work with the MindMeld private Python Package Index (PyPI). MindMeld Workbench is not publicly available, and can only be installed from the MindMeld PyPI, which is hosted at https://mindmeld.com/pypi/.

Workbench
''''''''''

Workbench reads your credentials from its configuration file, located at
``~/.mmworkbench/config``, when performing actions that require authentication, such as accessing
:doc:`blueprints <../blueprints/overview>`.

Create the ``~/.mmworkbench`` folder.

Create the ``~/.mmworkbench/config`` file and add the following lines, substituting your username and password:

.. code-block:: text

  [mmworkbench]
  mindmeld_url = https://mindmeld.com
  username = {YOUR_USERNAME}
  password = {YOUR_PASSWORD}


.. _getting_started_install_workbench:

Install Workbench
-----------------

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

The numerical parser is the component that relies on Java 8.

Begin New Project
-----------------

With the setup out of the way, you are now ready to get your feet wet. MindMeld Workbench is designed so you can
keep using the tools and coding patterns that are familiar to you. Some of the very basic operations can be performed in
your command-line shell using the ``mmworkbench`` command. But to really take advantage of the power of Workbench,
the Python shell is where all the action is at.


Command Line
^^^^^^^^^^^^

You can use the ``blueprint`` command in ``mmworkbench`` to begin a new project. This enables you to use one of the 
already built example apps as a baseline for your project. 

The `template` blueprint sets up the scaffolding for a blank project:

.. code-block:: console

  $ mmworkbench blueprint template myapp


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

The `template` blueprint sets up the scaffolding for a blank project:

.. code-block:: python

    import mmworkbench as wb
    wb.configure_logs()    
    wb.blueprint('template', 'my_app')


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

#. ``blueprint`` : Downloads all the training data for an existing blueprint and sets it up for use in your own project.
#. ``num-parse`` : Starts or stops the numerical parser service.

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

There is a handy ``configure_logs()`` function available that wraps this and sets log format:
  
.. code-block:: python

  import mmworkbench as wb  
  wb.configure_logs()
