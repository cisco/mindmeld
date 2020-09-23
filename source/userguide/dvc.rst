DVC for Model Tracking
======================

We use Data Version Control, or DVC, in Mindmeld in order to track trained models in our applications. DVC is an
open-source version control system for data science and machine learning projects that enables versioning of large
files and directories in concert with Git without storing the data itself in Git.

You can find more info about DVC in the `official documentation <https://dvc.org/doc>`_.


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

4. **Build the models, save them using DVC, and commit the new files to Git.**

.. code-block:: console

   python -m home_assistant build
   python -m home_assistant dvc --save
   git commit -m "Track models with dvc"

The save command creates a file (.generated.dvc) that tracks the trained models.

5. **Add new training data and follow the same commands in Step 4.**

.. code-block:: console

   # New training data and/or intents added
   python -m home_assistant build -i
   python -m home_assistant dvc --save
   git commit -m "Updated models with new training data"


6. **Switch between different trained models and repo states using the 'checkout' flag.**

.. code-block:: console

   # Use git log to get the git commit hash you want to checkout
   python -m home_assistant dvc --checkout_hash [HASH]
