Building a conversational interface in 11 steps
===============================================

This section outlines the step-by-step approach to building conversational interfaces using MindMeld workbench. This section provides four real-world examples (tutorials) to illustrate each of the steps in practice.


This section provides a high-level overview introducing the steps.

Select the right use case
-------------------------
Selecting the right use case is critical. Selecting an unrealistic or incorrect use case will render even the smartest voice or chat assistant dead on arrival. 
The best user cases today typically mimic an existing, familiar real-world interaction. This is the best way to ensure that users will know what they can ask.
Any use case must offer a practical path for collecting the training data required for the app.
The best use cases provide a clear value or utility where voice can be a faster way to find information or accomplish a task.

Define the dialog state flows
-----------------------------
In my view, this is where we define the natural language response templates which should be returned at each dialogue state in an interaction. We should illustrate a simple flow in a flow chart and then in a snippet of python code which illustrates how the logic is implemented in the dialogue manager.

Define the domain, intent, entity and role hierarchy
----------------------------------------------------
Show and describe a simple diagram which illustrates the domain, intent, entity and role hierarchy.  Show the directory structure which captures this hierarchy for a simple example.

Create the knowledge base
-------------------------
Describe what a knowledge base is and what it is used for. Describe options for storing the knowledge base. Show a simple example of creating a knowledge base by uploading a JSON file to ES via Workbench.

Generate representative training data
-------------------------------------
This section should introduce the topic of training data and why it is critical. It should describe different approaches for collecting training data (human data entry, mine the web, crowdsourcing, operational logs, etc.).  It should emphasize that training data needs to be representative and very high quality since it serves as the ground truth.

Show examples which illustrate how MindMeld can be used to generate training data for 
domain and intent classification data
entity recognition data
entity resolution and disambiguation data
answer ranking data
In particular, illustrate how the knowledge graph can be used to create micro-targeted crowdsourcing tasks which cover the full extend of a large product or content catalog.

Train the domain and intent models
----------------------------------
Introduce the topic of loading training data, training text classification models, measuring CV and held-out performance.

Train the entity and role recognizers
-------------------------------------
Introduce the topic of loading training data, training entity and role classification models, measuring CV and held-out performance.

Train the entity resolvers
--------------------------
Introduce the topic of loading training data, training entity resolution models, measuring CV and held-out performance, performing disambiguation.

Implement the semantic parser
-----------------------------
Introduce the topic of semantic and dependency parsing. Illustrate a simple example of a rule-based or grammar-based parser which groups entities into a tree data structure.

Optimize question answering
---------------------------
Introduce the topic of ranking for answer recommendations.

Deploy trained models to production
-----------------------------------
Show a simple example of the steps required to deploy to production