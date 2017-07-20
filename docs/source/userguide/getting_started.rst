.. meta::
    :scope: private

Getting Started
===============

MindMeld Workbench is a commercial software product which leverages a number of technologies. This
page serves as a guide to installing Workbench on your personal machine and
setting up your first Workbench project. The username and password used to access the MindMeld Learning
Center will be required in this process.

Install Java 8
--------------

MindMeld Workbench has a numerical parsing component that runs in Java. Java 8 or newer is
required. To check whether Java 8 is already installed on your system, use the following command:

.. code-block:: console

    $ java -version
    java version "1.8.0_131"
    Java(TM) SE Runtime Environment (build 1.8.0_131-b11)
    Java HotSpot(TM) 64-Bit Server VM (build 25.131-b11, mixed mode)

If the command fails, or your java version begins with 1.7, you need to install Java 8. Visit
`java.com <https://www.java.com/inc/BrowserRedirect1.jsp?locale=en>`_ for detailed instructions.

Set up Elasticsearch
---------------------

`Elasticsearch <https://www.elastic.co/products/elasticsearch>`_ is a highly scalable open-source
full-text search and analytics engine. It allows you to store, search, and analyze big volumes of
data quickly and in near real time. Workbench leverages Elasticsearch for information retrieval
based components. Generally, the latest version of Elasticsearch is recommended, but 5.0 or newer
is required.

Local Elasticsearch Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the best developer experience with smaller applications, Elasticsearch should be installed locally. On
macOS systems with `homebrew <https://brew.sh/>`_ installed, the simplest way to install
Elasticsearch is with the following set of commands.

.. code-block:: console

    $ brew update
    $ brew install elasticsearch

For other systems, or for more information on configuring Elasticsearch, go
`here <https://www.elastic.co/guide/en/elasticsearch/reference/current/_installation.html>`_.

After Elasticsearch has been configured simply run ``elasticsearch`` to start the process.

Remote Elasticsearch Cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have configured a remote Elasticsearch cluster to host Elasticsearch, you will need to set
the ``MM_ES_HOST`` environment variable accordingly. Add this to your shell's profile
(``.bash_profile`` or equivalent):

.. code-block:: bash

    export MM_ES_HOST="my-es-host.com"

Before attempting to use workbench, make sure to re-source so the environment variable is set.

.. code-block:: console

    source ~/.bash_profile


Install Python
--------------

Workbench is a Python-based machine learning library. To use Workbench, you will need to have
Python installed. If Python is not already installed on your system, you can get it at
`python.org <https://www.python.org/>`_ or use `pyenv <https://github.com/pyenv/pyenv>`_ to
manage multiple versions of Python. For workbench, Python 3.4 and newer are actively supported.
The latest version of Python 3 is recommended. Python 2.7+ should work fine too, but it is
deprecated.

Using a Virtual Environment
---------------------------

If you would like to work in an isolated environment (which is strongly recommended so you can work
on different projects without having conflicting library versions) you can use venv or virtualenv.

Virtual Environments with Pyenv
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are using pyenv you can easily create a virtual environment for any environment. For example
to create an environment using Python 3.6.1 named ``my-env`` use the following command:


.. code-block:: console

    $ pyenv virtualenv 3.6.1 my-env

If you would like to automatically activate the environment upon entering a project directory, use
the following command while at the root of the project directory:

.. code-block:: console

    $ pyenv local my-env

If instead you would like to use the environment (or any particular Python version) for the active
shell session only, use the following command:

.. code-block:: console

    $ pyenv shell my-env

Check out the `pyenv command reference <https://github.com/pyenv/pyenv/blob/master/COMMANDS.md>`_
for more details on how to use pyenv in general and the
`pyenv-virtualenv usage <https://github.com/pyenv/pyenv-virtualenv#usage>`_ for usage with
virtualenv specifically.

Manual Virtual Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you would like to manage your virtual environment directly, the method for doing so varies
slightly on various versions of Python as documented for Python 3.3+
`here <https://docs.python.org/3/library/venv.html>`_ and for earlier versions
`here <https://virtualenv.pypa.io>`_. In Python 3.6 and later you can use the following command.

.. code-block:: console

    $ python -m venv /path/to/venv/

Now every time you want to activate this environment, just open a terminal and type:

.. code-block:: console

    $ cd $WB_APP_ROOT
    $ source /path/to/venv/env/bin/activate

While the environment is active, any package you install using pip will be installed in this
isolated environment, and python will only have access to these packages.

Configure Pip
-------------

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

The MindMeld PyPI is hosted at https://pypi.mindmeld.com/simple/. In order to access it you will
need to authenticate using your username and password. Add the following lines to your pip
config file, substituting your username and password where appropriate.

.. code-block:: text

  [global]
  extra-index-url = https://{YOUR_USERNAME}:{YOUR_PASSWORD}@pypi.mindmeld.com/simple/

Configuring Workbench
---------------------

Certain MindMeld Workbench capabilities, such as accessing
:doc:`blueprints <../blueprints/overview>` require authenticating using your MindMeld username and
password. Workbench will read your credentials from its configuration file, located at
``~/.mmworkbench/config``. Add the following lines to the Workbench configuration file,
substituting your username and password where appropriate.

.. code-block:: text

  [mmworkbench]
  username = {YOUR_USERNAME}
  password = {YOUR_PASSWORD}


Install Workbench
-----------------

Now that your environment is set up, you can install MindMeld Workbench just as you would any other
Python package. This may take a few minutes if some of workbench's larger dependencies such as
`NumPy <http://www.numpy.org>`_, `SciPy <http://www.scipy.org>`_, and
`scikit-learn <http://scikit-learn.org/>`_ have not previously been installed.

.. code-block:: console

  $ pip install mmworkbench

If the following command returns no error, the installation was successful:

.. code-block:: console

    $ mmworkbench

Install Jupyter Notebook (Optional)
-----------------------------------

The recommended way of interacting with Workbench is via `Jupyter Notebook <http://jupyter.org/>`_,
an open-source web application that allows you to create and share documents with code,
visualizations, and explanatory text. You can install Jupyter using the following command.

.. code-block:: console

  $ pip install jupyter


Next, you can confirm the installation was successful using the following command.

.. code-block:: console

  $ jupyter notebook

Begin a New Project
-------------------

To begin a new project, you can use workbench's built-in blueprint functionality. There is a
``template`` blueprint which will create a new app with the basic skeleton, ready for you to extend.

The blueprint utility is available via the command line or from the python shell.

.. code-block:: console
    :caption: Console

    $ mmworkbench blueprint template my_app


.. code-block:: python
    :caption: Python shell

    >>> import mmworkbench
    >>> mmworkbench.blueprint('template', 'my_app')

Workbench provides several different blueprint applications to support many common use cases for
conversational applications.

Now you can fire up a Jupyter interactive workbook by typing:

.. code-block:: console

    $ cd my_app
    $ jupyter notebook
    [I 13:00 NotebookApp] Writing notebook server cookie secret to [...]
    [I 13:00 NotebookApp] Serving notebooks from local directory: [...]
    [I 13:00 NotebookApp] 0 active kernels
    [I 13:00 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=[...]
    [I 13:00 NotebookApp] Use Control-C to stop this server and shut down all kernels [...]
    [...]

A Jupyter notebook server is now running in your terminal, listening to port 8888. You can visit
this server by opening your Web browser to the URL displayed in the console readout (this usually
happens automatically when the server starts). You should see your workspace root directory
populated with the directories and files of your application blueprint.

.. image:: ../images/jupyter1.png
    :width: 700px
    :align: center

Now create a new Python notebook by clicking on the “New” button and selecting the appropriate
Python version. This will create a new notebook file called Untitled.ipynb in your workspace.
Click on the notebook title to change the name to something like 'my_app'.

A notebook contains a list of cells. Each cell can contain executable code or formatted text.
Right now the notebook contains only one empty code cell, labeled “In [1]:”. Try typing
``print("Hello world!")`` in the cell, and click on the play button or press Shift-Enter. This sends
the current cell to this notebook’s python kernel, which runs it and returns the output. The result
is displayed below the cell. Since we reached the end of the notebook, a new cell is
automatically created after cell execution. Go through the User Interface Tour from Jupyter’s Help menu to learn
the basics.

You are now ready to begin training and evaluating machine learning models for your application.
The following sections describe the modules and functionality available in Workbench to build and
evaluate state-of-the-art models to understand language, answer questions and power an advanced
conversational interface.
