Getting Started
===============

Workbench was build specifically to accommodate a new generation of advanced conversational assistants which need to by powered by large, dynamic custom data sets.  As a result MindMeld is optimized for the following guiding principles:

    •	be adaptable to handle the diverse requirements of large applications
    •	facilitate large, custom data sets and rather than pre-trained, generic models
    •	favor power and flexibility rather than simplicity and ease-of-use
    •	favor flexible SDKs rather than hardcoded, cloud-based APIs
    •	support versatile command-line utilities rather than rigid GUIs
    •	provide robust capabilities for large on-premise or VPC production deployments
  
MindMeld is intended to be a powerful tool for data scientists to build some of the most advanced voice and chat assistants possible today.


Obtain a License
~~~~~~~~~~~~~~~~

MindMeld Workbench is a commercial software product which requires a license to use. For more information or to inquire about obtaining a license for MindMeld Workbench, please `contact MindMeld sales <mailto:info@mindmeld.com>`_.



Set Up Your Environment
~~~~~~~~~~~~~~~~~~~~~~~

Workbench is a python-based machine learning library. To use Workbench, you will need to have python installed. If python is not already installed on your system, you can get it at `www.python.org <https://www.python.org/>`_. The latest version of Python 3 is recommended. Python 2.7+ should work fine too, but it is deprecated.

You will need a number of python modules: Jupyter, NumPy, Pandas, and Scikit-Learn. If you already have Jupyter running with all these modules installed, you can skip this section. If you don’t have them yet, there are many ways to install them (and their dependencies): you can use your system’s packaging system (eg. apt-get on Ubuntu, or MacPorts or HomeBrew on MacOSX), or install a Scientific Python distribution such as Anaconda and use its packaging system, or you can just use Python’s own packaging system which is included by default with the Python binary installers (since Python 2.7.9): pip. You can check to see if pip is installed by typing the following command:

.. code-block:: console

    $ pip3 --version
    pip 8.1.2 from [...]/lib/python3.5/site-packages (python 3.5)


You should make sure you have a recent version of pip installed, at the very least >1.4 to support binary module installation (a.k.a. wheels). To upgrade the pip module, type:

.. code-block:: console

    $ pip3 install --upgrade pip
    Collecting pip
    [...]
    Successfully installed pip-9.0.1

Now you can install all the required modules and their dependencies using this simple pip command:

.. code-block:: console

    $ pip3 install --upgrade jupyter numpy pandas scipy scikit-learn
    Collecting jupyter
      Downloading jupyter-1.0.0-py2.py3-none-any.whl
    Collecting numpy
      [...]

To check your installation, try to import every module like this:

.. code-block:: console

    $ python3 -c "import jupyter, numpy, pandas, scipy, sklearn"

There should be no output and no error. 


Install Workbench
~~~~~~~~~~~~~~~~~

Now that your environment is set up, you can install MindMeld Workbench using the following command. (TO DO: figure out the easiest, secure way to do this.)

.. code-block:: console

  $ pip install --upgrade --no-cache-dir https://get.mindmeld.com/Workbench/3.0/my-email/my-license-key/Workbench.tar.gz

You will need your registered email address as well as your registered license key in order to perform this installation. You can check that installation was successful if the following command returns no output or error:

.. code-block:: console

    $ python3 -c "import mmworkbench"


Begin a New Project
~~~~~~~~~~~~~~~~~~~

To begin a project, the first step is to create your application's root directory.

.. code-block:: console

    $ export WB_APP_ROOT="$HOME/my_app"
    $ mkdir -p $WB_APP_ROOT
    $ cd $WB_APP_ROOT

Your new project is now empty. The fastest way to set up the directory structure and data files for your project is to use one of Workbench's pre-configured blueprint applications. To set up a basic application skeleton, you can use the :keyword:`blueprint()` method to set up a baseline configuration:

.. code-block:: console

    $ python3 -c "import mmworkbench as wb; wb.blueprint('baseline');"

Workbench provides several different blueprint applications to support many common use cases for converational applications.

Now you can fire up a Jupyter intractive workbook by typing:

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

If you would like to work in an isolated environment (which is strongly recommended so you can work on different projects without having conflicting library versions), you should install virtualenv by running the following pip command:

.. code-block:: console

    $ pip3 install --user --upgrade virtualenv
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

