.. _define-hierarchy:

Define the Domain, Intent, Entity and Role Hierarchy
====================================================

Conversational applications rely on a hierarchy of machine learning classifiers in order to model and understand natural language. More broadly defined as Natural Language Processing, this family of machine learning models sits at the core all conversational assistants in widespread production use today. While there are many different ways that machine learning techniques can be enlisted to dissect and understand human language, a set of best practices has been adopted in recent years to systematize the sometimes challenging task of building accurate and useful natural language processing systems. Today, nearly all commercial conversational applications rely on the hierarchy of machine learning models illustrated below.

.. image:: images/hierarchy.png
    :width: 700px
    :align: center

The topmost layer in the model hierarchy is the domain classifier. The domain classifier is responsible for performing a first-pass classification to assign incoming queries into set of pre-defined buckets or 'domains'. For any given domain, there may be one or more pre-defined intents. Each intent defines a specific action or answer type to invoke for a given request. The intent classifier models are responsible for deciding which intent is most likely associated with a given request. Once the request is categorized into a specific intent, the entity recognition models are employed to discern the important words and phrases in each query that must be identified in order to understand and fulfill the request. These identified words and phrases are called 'entities', and each intent may have zero or more types of entities which must be recognized. For some types of entities, a fourth and final classification step, called role classification, may be required. The role classifiers are responsible for adding differentiating labels to entities of the same type. Refer to the :ref:`User Guide <userguide>` for a more in-depth treatment of the natural language processing classifier hierarchy utilized by MindMeld Workbench. 

For our simple conversational application which can help us find store information for our local Kwik-E-Mart, the natural language processing model hierarchy can be designed as illustrated below.

.. image:: images/hierarchy2.png
    :width: 700px
    :align: center

As illustrated, this rudimentary application has a single domain, 'store_info', which encompasses all of the functionality required to find information about Kwik-E-Mart retail stores. In addition, this domain supports five initial intents:

   - ``greet`` Begins an interaction and welcomes the user.
   - ``get_store_hours`` Returns the open and close time for the specified store.
   - ``find_nearest_store`` Returns the closest store to the user.
   - ``exit`` Ends the current interaction.
   - ``help`` Provides help information in case the user gets stuck.

.. note::

  By convention, intent names should always be verbs which describe what the user is trying accomplish.

In this basic example, only the ``get_store_hours`` intent requires entity recognition. This intent supports the two defined entity types listed below.

   - ``store_name`` The name of a specific retail store location.
   - ``date`` The calendar date or day of the week.

Neither of these two entity types will require role classification in this simple example.

.. note::

  By convention, entity names should always be nouns which describe the entity type.

The design of the domain, intent, entity and role hierarchy for this example application is now complete, and we can begin implementing this application using MindMeld Workbench. Every Workbench application begins with a root folder. The root folder contains all of the training data files, configuration files and custom code required in each Workbench application. For our simple example, lets first define a root directory called 'my_app'. 

.. code-block:: console

    $ export WB_APP_ROOT="$HOME/my_app"
    $ mkdir -p $WB_APP_ROOT
    $ cd $WB_APP_ROOT

To define the domain and intent hierarchy for your application, create a subfolder called 'domains'. Inside the 'domains' folder, create a subfolder for the name of each different domain in your application. Then, inside each domain folder, create another subfolder with the name of each individual intent in that domain. These folders are used to organize the training data for your machine learning models to understand natural language.

.. code-block:: console

    $ mkdir domains
    $ cd domains
    $ mkdir store_info
    $ cd store_info
    $ mkdir greet
    $ mkdir get_store_hours
    ...

Similarly, inside the root folder, create another subdirectory called 'entities'. Inside the entities folder, create a subdirectory for the name of every different entity type required in your application. These folders organize the data files used by the entity recognizer, role classifier and entity resolver models. 

.. code-block:: console

    $ cd $WB_APP_ROOT
    $ mkdir entities
    $ cd entities
    $ mkdir store_name

Workbench provides a faster way to create your application structure for common use cases. These are called application 'blueprints'. A blueprint is a pre-configured application structure. Starting with an empty root directory, you can set up your initial application structure using the :keyword:`blueprint()` method, as shown below.

.. code-block:: console

    $ python3 -c "import mmworkbench as wb; wb.blueprint('quickstart');"

For our simple example application, the resulting root directory structure is illustrated below. 

.. image:: images/directory.png
    :width: 400px
    :align: center

Refer to the :ref:`User Manual <userguide>` for more details about available blueprints as well as the organization and structure of the application root directory.


Notice that there is no folder for the ``date`` entity. In this case, ``date`` is a 'system' entity, which is already built in to the Workbench platform. Workbench provides several different 'system' entity types for common, domain-independent entities; see the Workbench :ref:`User Guide <userguide>` for details.  

Given this defined hierarchy, we would expect our trained natural language processing models to yield the following results for the user requests in the simple interaction proposed in the preceding section.

.. image:: images/quickstart_parse_output.png
    :width: 600px
    :align: center

The following sections of the step-by-step guide will describe how to introduce training data to the defined directories in order to build machine learning models to parse and understand user requests, as shown above.

