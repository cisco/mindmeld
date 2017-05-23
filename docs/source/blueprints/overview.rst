Workbench Blueprints
====================

MindMeld Workbench provides example applications, called *Workbench Blueprints*, that cover many common conversational use cases. Each blueprint has a pre-configured application structure and a pre-built set of code samples and data sets for a particular conversational scenario. You can use these blueprints to quickly build and test a fully working conversational app without having to write code or collecting any training data. Once you have this end-to-end setup ready and tested, you can use the blueprint as a baseline for further improvements and customization by adding data and logic specific to your business or application needs.

The Workbench blueprints thus have a dual purpose:

  #. They serve as useful tutorials showcasing how to build practical applications using the Workbench toolkit.

  #. They provide a fast and easy way to bootstrap your app on Workbench for many typical use cases.

In :doc:`Step 3 <../quickstart/define_the_hierarchy>` of the :ref:`Step-by-Step Guide <quickstart>`, we briefly saw how the :keyword:`Store Assistant` blueprint could be used to quickly set up the Workbench app directory structure for our Kwik-E-Mart store information app. In addition to the :keyword:`Store Assistant` use case, Workbench provides blueprints for a growing list of common use cases:

== ===
1  :doc:`Food Ordering <food_ordering>`
2  :doc:`Music Search and Discovery <music_discovery>`
3  :doc:`Meeting Assistant <meeting_assistant>`
== ===

These blueprints are covered in depth in the following sections. The goal of each section is to guide you through all the steps involved in using Workbench for building an app that addresses a specific real-world use case.

For each blueprint, you will learn how to:

  - define the app scope clearly in terms of domains, intents, entities and roles
  - setup a Workbench project by creating the app directory structure
  - create the knowledge base for the app
  - acquire and annotate training data for the statistical NLP models
  - train, tune and test the NLP models
  - configure the language parser
  - design and implement the dialogue states
  - deploy the app to a conversational platform or device


Blueprint Setup
~~~~~~~~~~~~~~~

To get started with any of the blueprints, you must have Workbench and all of its required dependencies installed. Please refer to the :doc:`Getting Started <../userguide/getting_started>` page for more information on acquiring the Workbench toolkit and installing it on your system.

With Workbench installed, you can use the :keyword:`blueprint()` method to set up a blueprint application.

.. code-block:: console

    $ python -c "import mmworkbench as wb; wb.blueprint('BLUEPRINT_NAME');"

Here, :keyword:`BLUEPRINT_NAME` refers to one of the Workbench-supported blueprints. Running the above command will set up a Workbench project in your current directory for working with the requested blueprint app. Refer to the individual blueprint sections for more details.

