.. meta::
    :scope: private

Getting Started
===============

This will get you started with installing MindMeld Workbench on your personal machine
and setting up your first Workbench project.

Pre-requisites
--------------

This release of MindMeld Workbench is targeted towards seasoned Python developers with machine 
learning knowhow and experience with one or more popular machine learning frameworks and libraries.

The username and password used to access the MindMeld Learning Center will be required in this process.

Install Software
----------------

Python
^^^^^^

Workbench is a Python-based machine learning library. To use Workbench, you will need to have
Python installed. If Python is not already installed on your system, you can get it at
`python.org <https://www.python.org/>`_ or use `pyenv <https://github.com/pyenv/pyenv>`_ to
manage multiple versions of Python. For workbench, Python 3.4 and newer are actively supported.
The latest version of Python 3 is recommended. Python 2.7+ should work fine too, but it is
deprecated.


Configure Pip
"""""""""""""

The simplest way to install workbench is using pip, Python’s packaging system which is included by
default with the Python binary installers (since Python 2.7.9). You can check to see if pip is
installed by typing the following command:

.. code-block:: console

    $ pip --version
    pip 8.1.2 from [...]/lib/python3.5/site-packages (python 3.5)

You should make sure you have a recent version of pip installed, at the very least >1.4 to support
binary module installation (a.k.a. wheels). To upgrade the pip module, type:

.. code-block:: console

    $ pip install --upgrade pip
    Collecting pip
    [...]
    Successfully installed pip-9.0.1

MindMeld Workbench is not publicly available, and can only be installed from MindMeld's private
Python Package Index (PyPI). Once you have confirmed pip is installed, you need to configure it
so that it will work with the MindMeld PyPI. On macOS the pip config file is located at
``~/.pip/pip.conf``. You can read more about configuring pip on your platform, including where
config files are located in the
`pip documentation <http://pip.readthedocs.io/en/latest/user_guide/#configuration>`_.

The MindMeld PyPI is hosted at https://mindmeld.com/pypi/. In order to access it you will
need to authenticate using your username and password. Add the following lines to your pip
config file, substituting your username and password where appropriate.

.. code-block:: text

  [global]
  extra-index-url = https://{YOUR_USERNAME}:{YOUR_PASSWORD}@mindmeld.com/pypi/
  trusted-host = mindmeld.com


Using a Virtual Environment
"""""""""""""""""""""""""""

If you would like to work in an isolated environment (which is strongly recommended so you can work
on different projects without having conflicting library versions), you can use `virtualenv with pyenv
<https://github.com/pyenv/pyenv-virtualenv>`.

Java 8
^^^^^^

MindMeld Workbench has a numerical parsing component that requires Java 8 or newer. To check whether
Java 8 is already installed on your system, use the following command:

.. code-block:: console

    $ java -version
    java version "1.8.0_131"
    Java(TM) SE Runtime Environment (build 1.8.0_131-b11)
    Java HotSpot(TM) 64-Bit Server VM (build 25.131-b11, mixed mode)

If the command fails, or your java version begins with 1.7, you need to install Java 8. Visit
`java.com <https://www.java.com/inc/BrowserRedirect1.jsp?locale=en>`_ for detailed instructions.


Elasticsearch
^^^^^^^^^^^^^

`Elasticsearch <https://www.elastic.co/products/elasticsearch>`_ is a highly scalable open-source
full-text search and analytics engine. It allows you to store, search, and analyze big volumes of
data quickly and in near real time. Workbench leverages Elasticsearch for information retrieval. 
Generally, the latest version of Elasticsearch is recommended, but 5.0 or newer is required.

For the best developer experience with smaller applications, Elasticsearch should be installed locally. On
macOS systems with `homebrew <https://brew.sh/>`_ installed, the simplest way to install
Elasticsearch is with the following set of commands.

.. code-block:: console

    $ brew update
    $ brew install elasticsearch

For other systems, or for more information on configuring Elasticsearch, go
`here <https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html>`_.

After Elasticsearch has been configured simply run ``elasticsearch`` to start the process.


Setup Workbench
---------------

Now, we are ready to install Workbench.


Configure Workbench
^^^^^^^^^^^^^^^^^^^

Certain MindMeld Workbench capabilities, such as accessing
:doc:`blueprints <../blueprints/overview>` require authenticating using your MindMeld username and
password. Workbench will read your credentials from its configuration file, located at
``~/.mmworkbench/config``. Add the following lines to the Workbench configuration file,
substituting your username and password where appropriate.

.. code-block:: text

  [mmworkbench]
  mindmeld_url = https://mindmeld.com
  username = {YOUR_USERNAME}
  password = {YOUR_PASSWORD}


Configure your Virtual Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You may want to install Workbench in an isolated environment so you can keep it separate from your other work.
Here is one way of doing this using pyenv and virtualenv:

.. code-block:: console

  mkdir workbench-development
  cd $_

  # install Python 3.6.1
  pyenv install 3.6.1

  # create a new virtual environment using Python 3.6.1 
  pyenv virtualenv 3.6.1 workbench

  # automatically activate the environment upon entering this directory
  pyenv local workbench


Install Workbench
^^^^^^^^^^^^^^^^^

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


Begin New Project
-----------------

With the setup out of the way, you are now ready to get your feet wet. MindMeld Workbench is designed so you can 
keep using the tools and coding patterns that are familiar to you. Some of the very basic operations can be performed in 
your command-line shell using the ``mmworkbench`` command. But to really take advantage of the power of Workbench, 
the Python shell is where all the action is at.


Command Line
^^^^^^^^^^^^

You can use ``blueprint`` command in ``mmworkbench`` to begin a new project. This enables you to use one of the 
already built example apps as a baseline for your project.

To try out the :doc:`Food Ordering blueprint<../blueprints/food_ordering>`, run these commands on the command line:

.. code-block:: console

  $ mmworkbench blueprint food_ordering
  $ cd $_
  $ python app.py build   # this will take a few minutes
  $ python app.py converse
 Loading intent classifier: domain='ordering'
 ...
 You:    

The *converse* command loads the machine learnings models and starts an interactive session with the "You:" prompt. 
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


Upgrade Workbench
-----------------

To upgrade to the latest version of Workbench, you can run:

.. code-block:: console

  $ pip install mmworkbench --upgrade

Make sure to run this regularly to stay on top of the latest bug fixes and feature releases.

