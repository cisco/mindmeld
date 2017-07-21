.. meta::
    :scope: private

Getting Started
===============

This will get you started with installing MindMeld Workbench on your personal machine
and setting up your first Workbench project.

Pre-requisites
--------------

This release of MindMeld Workbench is targeted towards seasoned Python developers with machine 
learning knowhow and experience with one or more popular frameworks and libraries. If you don't fit
that profile, we suggest you wait or look elsewhere.

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
on different projects without having conflicting library versions), you can use pyenv.

Check out the `pyenv command reference <https://github.com/pyenv/pyenv/blob/master/COMMANDS.md>`_
for more details on how to use pyenv in general and the
`pyenv-virtualenv usage <https://github.com/pyenv/pyenv-virtualenv#usage>`_ for usage with
virtualenv specifically.


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

You may want to install Workbench in an isolate environment so you can keep it separate from your other work.
Here is one way of doing this using pyenv and virtual-env:

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


Upgrade Workbench
^^^^^^^^^^^^^^^^^

To upgrade to the latest version of Workbench, you can run:

.. code-block:: console

  $ pip install mmworkbench --upgrade


Begin a New Project
^^^^^^^^^^^^^^^^^^^

To begin a new project, you can use workbench's built-in ``blueprint`` command, which uses one of the 
already built example apps for you to use as a baseline.

To try out the :doc:`Food Ordering blueprint<../blueprints/food_ordering>`, run these commands:

.. code-block:: console

  $ mmworkbench blueprint food_ordering
  $ cd $_
  $ mmworkbench num-parse --start     # starts the numerical parser
  $ python app.py build   # this will take a few minutes
  $ python app.py converse
 Loading intent classifier: domain='ordering'
 ...
 You:    

The *converse* command loads the machine learnings models and starts an interactive session with the "You:" prompt. 
Here you can enter your own input and get an immediate response back. Try "hi", for example, and see what you get.


Workbench provides several different blueprint applications to support many common use cases for
conversational applications.


Begin Your Journey
------------------

You are now ready to begin training and evaluating machine learning models for your application.
The following sections describe the modules and functionality available in Workbench to build and
evaluate state-of-the-art models to understand language, answer questions and power an advanced
conversational interface.
