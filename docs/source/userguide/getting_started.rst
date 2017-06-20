.. meta::
    :scope: private

Getting Started
===============

Describe how to get a license to workbench, how to install the package, and how to set up a new project with the correct directory structure and dependencies.
4.2.1 Obtain a license
4.2.2 Preparation: Install dependencies, setup environment (virtualenv, etc.)
4.2.3 Install Workbench
4.2.4 Create a Workbench project (from blueprint)


[ARCHIVED CONTENT BELOW]


Obtain a License
~~~~~~~~~~~~~~~~

**TODO: Revisit this section -- talk more about user name and password**

MindMeld Workbench is a commercial software product which requires a license to use. For more information or to inquire about obtaining a license for MindMeld Workbench, please `contact MindMeld sales <mailto:info@mindmeld.com>`_. You will recieve credentials which can be used to install MindMeld Workbench and related files.

Install Java 8
~~~~~~~~~~~~~~

**TODO: Add section about installing Java 8.**

Set Up Your Environment
~~~~~~~~~~~~~~~~~~~~~~~

Workbench is a Python-based machine learning library. To use Workbench, you will need to have Python installed. If Python is not already installed on your system, you can get it at `www.python.org <https://www.python.org/>`_ or use `pyenv <https://github.com/pyenv/pyenv>`_ to manage multiple versions of Python. The latest version of Python 3 is recommended. Python 2.7+ should work fine too, but it is deprecated.

The simplest way to install workbench is using pip, Python’s packaging system which is included by default with the Python binary installers (since Python 2.7.9). You can check to see if pip is installed by typing the following command:

.. code-block:: console

    $ pip --version
    pip 8.1.2 from [...]/lib/python3.5/site-packages (python 3.5)


You should make sure you have a recent version of pip installed, at the very least >1.4 to support binary module installation (a.k.a. wheels). To upgrade the pip module, type:

.. code-block:: console

    $ pip install --upgrade pip
    Collecting pip
    [...]
    Successfully installed pip-9.0.1

Once you have confirmed pip is installed, you need to modify the config file so it will work with MindMeld's private Python Package Index (PyPI). On Mac OS X the pip config file is located at ``~/.pip/pip.conf``. You can read more about configuring pip on your platform, including where config files are located in the `pip documentation <http://pip.readthedocs.io/en/latest/user_guide/#configuration>`_.

MindMeld's private Python Package Index is hosted at https://pypi.mindmeld.com/simple/. In order to access it you will need to authenticate using your username and password. Add the following lines to your pip config file.

.. code-block:: text

  [global]
  extra-index-url = https://username:password@pypi.mindmeld.com/simple/


Install Workbench
~~~~~~~~~~~~~~~~~

Now that your environment is set up, you can install MindMeld Workbench just as you would any other Python package. This may take a few minutes if some of workbench's larger dependencies such as `NumPy <http://www.numpy.org>`_, `SciPy <http://www.scipy.org>`_, and `scikit-learn <http://scikit-learn.org/>`_ have not previously been installed.


.. code-block:: console

  $ pip install mmworkbench

You can check that installation was successful if the following command returns no error:

.. code-block:: console

    $ mmworkbench
    Usage: mmworkbench [OPTIONS] COMMAND [ARGS]...

      Command line interface for mmworkbench.

    Options:
      -V, --version        Show the version and exit.
      -v, --verbosity LVL  Either CRITICAL, ERROR, WARNING, INFO or DEBUG
      -h, --help           Show this message and exit.

    Commands:
      blueprint
      build         Builds the app with default config
      clean         Delete all built data, undoing `build`
      converse      Starts a conversation with the app
      create-index  Create a new question answerer index
      load-index    Load data into a question answerer index
      num-parse     Starts or stops the numerical parser service
      run           Starts the workbench service


Install Jupyter Notebook (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The recommended way of interacting with Workbench is via `Jupyter Notebook <http://jupyter.org/>`_, an open-source web application that allows you to create and share documents with code, visualizations, and explanatory text. You can install Jupyter using the following command.

.. code-block:: console

  $ pip install jupyter


Next, you can confirm the installation was successful using the following command.

.. code-block:: console

  $ jupyter notebook

**TODO: Add more detail on benefits of Jupyter Notebook**


Begin a New Project
~~~~~~~~~~~~~~~~~~~

**TODO: Revisit this section**

To begin a project, the first step is to create your application's root directory.

.. code-block:: console

    $ export WB_APP_ROOT="$HOME/my_app"
    $ mkdir -p $WB_APP_ROOT
    $ cd $WB_APP_ROOT

Your new project is now empty. The fastest way to set up the directory structure and data files for your project is to use one of Workbench's pre-configured blueprint applications. To set up a basic application skeleton, you can use the ``blueprint()`` method to set up a baseline configuration:

.. code-block:: console

    $ mmworkbench blueprint baseline

Workbench provides several different blueprint applications to support many common use cases for converational applications.

Now you can fire up a Jupyter interactive workbook by typing:

.. code-block:: console

    $ jupyter notebook
    [I 13:00 NotebookApp] Writing notebook server cookie secret to [...]
    [I 13:00 NotebookApp] Serving notebooks from local directory: [...]
    [I 13:00 NotebookApp] 0 active kernels
    [I 13:00 NotebookApp] The Jupyter Notebook is running at: http://localhost:8888/?token=[...]
    [I 13:00 NotebookApp] Use Control-C to stop this server and shut down all kernels [...]
    [...]

A Jupyter server is now running in your terminal, listening to port 8888. You can visit this server by opening your Web browser to the URL displayed in the console readout (this usually happens automatically when the server starts). You should see your workspace root directory populated with the directories and files of your application blueprint.

.. image:: images/jupyter1.png
    :width: 700px
    :align: center

Now create a new Python notebook by clicking on the “New” button and selecting the appropriate Python version. This will create new notebook file called Untitled.ipynb in your workspace. Click on the notebook title to change the name to something like 'my_app'.

A notebook contains a list of cells. Each cell can contain executable code or formatted text. Right now the notebook contains only one empty code cell, labeled “In [1]:”. Try typing print("Hello world!") in the cell, and click on the play button or type Shift-Enter. This sends the current cell to this notebook’s python kernel, which runs it and returns the output. The result is displayed below the cell, and since we reached the end of the notebook, a new cell is automatically created. Go through the User Interface Tour from Jupyter’s Help menu to learn the basics.

You are now ready to begin training and evaluating machine learning models for your application. The following sections describe the modules and functionality available in Workbench to build and evaluate state-of-the-art models to understand language, answer questions and power a conversational interface.

Using a Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**TODO: Revisit this section**

If you would like to work in an isolated environment (which is strongly recommended so you can work on different projects without having conflicting library versions), you should install virtualenv by running the following pip command:

.. code-block:: console

    $ pip install --user --upgrade virtualenv
    Collecting virtualenv
    [...]
    Successfully installed virtualenv

Now you can create an isolated python environment by typing:

.. code-block:: console

    $ cd $WB_APP_ROOT
    $ virtualenv env
    Using base prefix '[...]'
    New python executable in [...]/env/bin/python3.5
    Also creating executable in [...]/env/bin/python
    Installing setuptools, pip, wheel...done.

Now every time you want to activate this environment, just open a terminal and type:

.. code-block:: console

    $ cd $WB_APP_ROOT
    $ source env/bin/activate

While the environment is active, any package you install using pip will be installed in this isolated environment, and python will only have access to these packages (if you also want access to the system’s site packages, you should create the environment using virtualenv’s --system-site-packages option). Check out virtualenv’s documentation for more information.

