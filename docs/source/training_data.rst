Training Data
================

.. raw:: html

    <style> .red {color:red} </style>

.. raw:: html

    <style> .green {color:green} </style>

.. raw:: html

    <style> .blue {color:blue} </style>

.. raw:: html

    <style> .orange {color:orange} </style>

.. role:: red
.. role:: green
.. role:: blue
.. role:: orange

Data is the fuel that powers any machine learning-based application. Many components in our Deep Domain Conversational AI System utilize supervised Machine Learning models to parse the user's query and derive meaning out of it. Typically, we require on the order of thousands to millions of queries to train powerful models. 

This chapter focuses on general best practices around working with training data. In particular, we will take a look at:

* Acquiring the right kind of training data for your application.

    + The data must be *highly representative* of real-world usage of the system
    + The spec for generating example queries should very **precise** to your use-case
    + It needs to cover all conversational scenarios that you expect your app to handle
    + The data should include both formal and informal language constructs, ideally including popular slangs, abbreviations and other unconventional language expressions
    + The people generating this data should be drawn from a real-world, representative and unbiased pool of non-experts

* Annotating it appropriately to provide meaningful labels.

    + The guidelines for annotation must be simple, consistent and unambiguous. This enables us to use the wisdom of crowds to generate annotations at scale
    + The annotation format must be lightweight. Text-editor based workflows usually provide excellent velocity as compared to vaguely designed GUIs


Collecting Data At Scale
------------------------

.. _Amazon Mechanical Turk: https://www.mturk.com

For a varied pool of users, crowdsourcing is a great way to bootstrap your data collection process. Platforms such as `Amazon Mechanical Turk`_ are popular and relatively inexpensive to get an initial dataset. For enforcing greater standards and niche use-cases, several alternate services are available at a higher price point. As mentioned above, the spec to send out to "Turkers" should be highly precise and specific to the use-case. Lack of clarity or specificity can lead to noisy data, which hurts training accuracy.

Example specs:

Use case - :red:`"Music Information about artists, albums and release year of songs/albums."`

.. code-block:: text

  Scenario: You are interested in getting information about music
  
  Task: Search for music information by voice. You can ask it for information about which artist
  created a song, what album a song appears in, and when (or what year) a song/album came out.
  
  Examples:
  
  When was the album Thriller made?
  Who sang American Pie?
  Please tell me what year The Beatles released the album Yellow Submarine.
  What album of Pink Floyd does Comfortably Numb appear in?
  Can you tell me which year Stairway To Heaven came out?
  Who wrote Embraceable You?
  Who was the composer of the Eroica Symphony?
  
  Please provide 3 additional queries below and please try to vary your phrasing on each query.


Use case - :red:`"Order Food or Beverages from Starbucks"`

.. code-block:: text

  Imagine that you have a Starbucks voice app that you can talk to, like it's a human assistant.
  You can ask the app to order you food, drinks or bagged coffee from a Starbucks store.
  
  Here are some examples:
  
      "Can I get a grande iced coffee with vanilla syrup?"
      "I would like an egg and cheddar sandwich and a tall coffee."
      "Please order me a bag of the Guatemala Antigua beans."
      "Get me a chocolate smoothie with nonfat milk."
      "One venti decaf blonde roast and one tall Americano with room for cream."
      "I'm in the mood for a strawberry creme frap and a blueberry scone."
      "A venti caramel macchiato with extra whip."
  
  Please provide five queries in your own words and your own style, DIFFERENT FROM THE QUERIES ABOVE.
  Please try to vary the way you ask questions, so your queries do not all follow the same pattern.
  Here's a menu for reference: http://www.starbucks.com/menu/


Design Dialogue Flowchart
~~~~~~~~~~~~~~~~~~~~~~~~~

For conversational agents heavy in multi-part interactions in a dialogue sequence, a useful exercise is to design a Dialogue Flowchart for your application. The flowchart should describe the various dialogue states that your system can be in and the transitions between each of those states. This exercise will help you to think through all the different ways the conversation could branch while starting from a clean slate and progressing towards the final goal state.

For instance, here's a sample flowchart for a coffee-ordering application (click to enlarge):

.. image:: images/dialog_flow.png
   :scale: 25%
   :target: _images/dialog_flow.png

Once the Dialogue Flowchart has been designed, you can then work through each dialogue state and generate user queries per state. The goal is to imagine yourself in the shoes of a user who is at a particular place in the flow and predict what would be the most likely things that the user could say to the system at that point. This exercise will help you in understanding the needs and functionality of your conversational interface better, and guides you towards rapidly collecting highly relevant data.

In the case of a coffee-ordering application, at the very initial state, the user could either begin with a greeting to the system or launch straight into placing an order. Below are a few example queries of possible greetings that a user could begin with:

.. code-block:: text

  Hello
  Hi
  Good morning
  ...

You could launch a simple crowdsourcing task specific to this **greeting** state and collect hundreds or thousands of queries quickly. 

Next, we want to specify the task of ordering a coffee.

.. code-block:: text

  I want a chai tea latte.
  Just a shot of espresso please.
  Get me a grande mocha with whipped cream on top.
  Can I get a tall cappuccino with skimmed milk, one venti caramel macchiato, two chocolate chip muffins and a banana?

Following along the rest of the dialogue flows, you can see how each branch in the state-flow diagram guides us in determining how to choose task specifications to collect queries. In a similar vein, we can collect queries for additional use cases such as reordering from history, checking order status etc.

Define Domains, Intents And Entities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Based on the use cases and data collected, we can develop an intuition for the types of domains and intents we should create in order to build a robust conversational system. Intents refer to the different possible intentions a user might have in each state and determine which branch gets selected next in the dialogue flow. Domains are collections of related intents sharing a common vocabulary. While there is no "one size fits all" approach to modeling domains and intents, we can follow some basic guidelines and recommended best practices for modeling these concepts.

* Defining domains is not necessary for an app that only needs to handle a small number of intents. In such a case, all intents can be considered as belonging to a single de-facto domain.
* When you have more than a dozen intents, consider grouping intents into domains based common language patterns and vocabulary.
* Queries in any domain should be semantically and syntactically different from all other domains.
* Defining the right set of intents for each domain is critical to ensure good accuracy and broad coverage of user queries. Each domain should have no more than a few dozen different intents.
* To optimize parser accuracy, the queries for each intent should should be semantically and syntactically different from the queries for all other intents

Some example intents for the coffee-ordering app would be:

* greeting
* order
* reorder
* check_order_status

Once the domains and intents are established, we go ahead and start defining the Entities. Each intent has its own set of entities which provide structure to the language patterns. For instance, the **order** intent above can have entities such as :blue:`drink_name`, :green:`size`, :orange:`quantity` etc. So when a user says :red:`"a tall grande mocha"`, the system extracts the relevant entities from the query and fills the corresponding entities: 

* :blue:`drink_name` = "mocha"
* :green:`size` = "tall"
* :orange:`quantity` = "1"

Defining these constructs is necessary in order to annotate your data appropriately, so it is important to establish these with clarity and flexibility in order to minimize future re-annotations.

Annotate the Training Data
--------------------------

The MindMeld Entity Recognizer extracts relevants entities from user queries and assigns them to appropriate entity types. For instance, when a user says :red:`"I want to order a tall decaf latte"`, the Entity Recognizer identifies :green:`tall` as a :green:`size`, :orange:`decaf` as an :orange:`option` and :blue:`latte` as a :blue:`drink_name`.

In order to do this accurately at runtime, the Entity Recognizer needs to be trained to recognize and classify entities in free-form text. We do this by adding annotations to our training data to identify all the entities within our collected queries. We mark up the parts of the query that correspond to a slot, i.e. provide further information to the app about the current user intent.

Here are some examples:

.. code-block:: text

  {cinnamon dolce latte|name} {venti|size} {extra whip|option}
  let me get a {double shot on ice|name}
  {medium|size} {iced coffee|name} with {low fat milk|option}

Annotation Markup
~~~~~~~~~~~~~~~~~

The annotation markup syntax is fairly straightforward:

* Enclose the entity in curly braces
* Follow the entity with its type
* Use the pipe character as separator

A useful strategy is to annotate a few hundred queries, train an initial Entity Recognizer using those examples and then use the trained Entity Recognizer to annotate new queries. That way, you will only need to fix errors made by the Entity Recognizer as opposed to generating annotations from scratch. You can retrain the Entity Recognizer at regular intervals and as the model gets better, you'll have lesser errors to correct while annotating new data.

Be consistent with your entity annotations and ensure that the annotated entity span matches with the entries in your Entity Map. Queries without any slot information should be left unannotated. E.g. "Sure", "Hello", "Yes, please".


Organize the Training Data
--------------------------

Workbench expects all your training data to be in the following hierarchical directory structure:

.. code-block:: text

  data_dir_root/
    domain_1/
      intent_1/
        train_file_1.txt
        train_file_2.txt
        ...
      intent_2/
        train_file_1.txt
        train_file_2.txt
        ...
      .
      .
      .
      intent_n/
        train_file_1.txt
        train_file_2.txt
        ...
    domain_2/
      intent_1/
        train_file_1.txt
        train_file_2.txt
        ...
      .
      .
      .
      intent_n/
        train_file_1.txt
        train_file_2.txt
        ...
    .
    .
    .
    domain_n/
      ...

At the top level, there is a folder for each domain, and within each domain folder, there's a subfolder for each intent belonging to that domain. The text files containing the annotated training data for each intent should be placed in their respective intent folders.

Here's how Workbench determines the data to use for training each of our classification models:

=================  =================================================================================
Model              Training Data Used
=================  =================================================================================
Domain Classifier  Data across all domains and intents: :code:`data_dir_root/*`
Intent Classifier  Data across all intents in a particular domain: :code:`data_dir_root/domain_i/*`
Entity Recognizer  Data restricted to a particular intent: :code:`data_dir_root/domain_i/intent_j/*`
Role classifier    Data restricted to a particular intent: :code:`data_dir_root/domain_i/intent_j/*`
=================  =================================================================================
