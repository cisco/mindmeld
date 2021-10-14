DVC & DAGsHub for Model Tracking
================================

We use Data Version Control, or DVC, in MindMeld in order to track trained models in our applications. DVC is an
open-source version control system for data science and machine learning projects that enables versioning of large
files and directories in concert with Git without storing the data itself in Git.

DAGsHub is a platform built on top of Git & DVC that enables users to host, manage and collaborate on their code,
data, models and experiments. We use DAGsHub in MindMeld as a central model registry, to make our models accessible from anywhere and easily shareable.

You can find more info about DVC in the `official documentation <https://dvc.org/doc>`_.

You can find more info about DAGsHub in the `DAGsHub documentation <https://dagshub.com/docs>`_.

Install DVC using the following command.

.. code-block:: console

   pip install 'mindmeld[dvc]'

The functionality of the DVC & DAGsHub command within MindMeld is shown in the table below.

Available Options in MindMeld
-----------------------------

+-----------------+------------------------------------------------------------------------+
| **Option**      | **Description**                                                        |
+-----------------+------------------------------------------------------------------------+
| --init          | Initializes DVC within a repository. Must be run before other options. |
+-----------------+------------------------------------------------------------------------+
| --setup_dagshub | Setup a central model registry with DAGsHub[1][2].                     |
+-----------------+------------------------------------------------------------------------+
| --save          | Saves models using DVC. Run after model building finishes.             |
+-----------------+------------------------------------------------------------------------+
| --checkout HASH | Checks out the repo state and models corresponding to a git hash.      |
|                 | The --save option must have been run before committing the hash for    |
|                 | model checkout to work.                                                |
+-----------------+------------------------------------------------------------------------+
| --destroy       | Remove all files associated with DVC from a directory.                 |
|                 | Must be run in the directory that contains the .dvc folder             |
+-----------------+------------------------------------------------------------------------+
| --help          | Show the available options and their descriptions.                     |
+-----------------+------------------------------------------------------------------------+

- [1] The DAGsHub central repository requires you `sign up to DAGsHub <https://dagshub.com/user/sign_up>`_ (it's free for personal projects), and create a project – you can `connect an existing GitHub project <https://dagshub.com/repo/connect>`_, `migrate a project from any git hosting <https://dagshub.com/repo/migrate>`_ or `create a project from scratch <https://dagshub.com/repo/create>`_.

- [2] `--setup_dagshub` option only works after running `--init`.

We will use the Home Assistant blueprint to demonstrate the various command options.


Example: Home Assistant Blueprint
---------------------------------

Let's take a look at how model tracking works using the Home Assistant blueprint. Below we'll walk through each of
the necessary steps starting from an empty directory.

1. **Download the Home Assistant blueprint.**

.. code-block:: console

   mindmeld blueprint home_assistant

2. **Start a new Git repository and add Home Assistant files to Git.**

.. code-block:: console

   git init
   git add home_assistant/
   git commit -m "Added home_assistant blueprint"

DVC requires a source code management (SCM) tool such as Git to operate.

3. **Initialize DVC in the directory.**

.. code-block:: console

   python -m home_assistant dvc --init

This will create all the files necessary for DVC to operate.

.. code-block:: console

  On branch master
  Changes to be committed:
    (use "git reset HEAD <file>..." to unstage)

        new file:   .dvc/.gitignore
        new file:   .dvc/config
        new file:   .dvc/plots/confusion.json
        new file:   .dvc/plots/default.json
        new file:   .dvc/plots/scatter.json
        new file:   .dvc/plots/smooth.json
        new file:   .dvcignore

4. **[Optional] Setup a central model registry with DAGsHub**

.. code-block:: console

   python -m home_assistant dvc --setup_dagshub

You will be prompted to input the URL of the DAGsHub project `you created <https://dagshub.com/repo/create>`_, your user name, and your `password (access token) <https://dagshub.com/user/settings/tokens>`_.

You're DVC remote will then point to your DAGsHub storage, with credentials set up for you to save your models directly to your project.

We recommend committing your code to Git and pushing to your connected GitHub project/DAGsHub project using ``git push`` so that you can see your entire project in the DAGsHub UI.

5. **Build the models, save them using DVC, and commit the new files to Git.**

.. code-block:: console

   python -m home_assistant build
   python -m home_assistant dvc --save
   git commit -m "Track models with dvc"

The save command creates a file (.generated.dvc) that tracks the trained models.

.. code-block:: console

  On branch master
  Changes to be committed:
    (use "git reset HEAD <file>..." to unstage)

        new file:   home_assistant/.generated.dvc

6. **Add new training data and follow the same commands in Step 4.**

.. code-block:: console

   # New training data and/or intents added
   python -m home_assistant build -i
   python -m home_assistant dvc --save
   git commit -m "Updated models with new training data"


7. **Switch between different trained models and repo states using the 'checkout' flag.**

.. code-block:: console

   # Use git log to get the git commit hash you want to checkout
   python -m home_assistant dvc --checkout [HASH]
