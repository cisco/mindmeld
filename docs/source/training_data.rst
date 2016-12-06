Training Data
================

Data is the fuel that powers any machine learning-based application. Many components in our Deep Domain Conversational AI System utilize machine learned models to parse the user's query and derive meaning out of it. It's fair to say that half the intelligence of the system comes from the machine learning algorithms themselves, but the other half comes from the data used to train those models.

This chapter focuses on general best practices around working with training data. In particular, we'll take a look at:

#. Acquiring the right kind of training data for your application
#. Annotating it appropriately to provide meaningful labels for the machine learning models to learn from


Generating Representative Training Data
---------------------------------------

The performance of machine learning systems depends on both the quantity and the quality of training data. Of the two, quantity, being directly measurable is an easier concept to understand. Generally speaking, the more data you use to train your model, the more accurate its predictions are. However, that's just half the story. Not all data is created equal. The quality of your data will greatly determine your success in training these advanced machine learning models and by extension, building a successful conversational interface.

Machine learning models learn to "fit" the data you provide during training. In other words, they analyze the examples given during training and build a predictive model based on the patterns they learn from those examples. It's hence crucial that your training data be highly representative of the real-world usage of the system. It needs to cover all the conversational scenarios that you expect your app to handle. It also needs to closely match the language (in style and format) that you expect real users to use. You may therefore need to include both formal and informal language constructs, popular slangs, abbreviations and other unconventional language expressions in your training data to ensure that the models powering your system can generalize across a variety of users.


Design Dialogue Flowchart
~~~~~~~~~~~~~~~~~~~~~~~~~

The first step towards training data generation is designing the Dialogue Flowchart for your application. The flowchart should describe the various dialogue states that your system can be in and the transitions between each of those states. This exercise will help you to think through all the different ways the conversation could branch while starting from a clean slate and progressing towards the final goal state.

For instance, here's a sample flowchart for a coffee-ordering application (click to enlarge):

.. image:: images/dialog_flow.png
   :scale: 25%
   :target: _images/dialog_flow.png


Author Archetypal Queries
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the Dialogue Flowchart has been designed, the next step is to work through each dialogue state, and author a set of archetypal user queries per state. The goal is to imagine yourself in the shoes of a user who's at a particular place in the dialogue flow and predict what would be the most likely things that the user could say to the system at that point. This exercise will help you in clearly understanding the needs and functionality of your conversational interface better. It will also provide you with an initial set of seed examples you can later expand upon to build out a representative training set.

In the case of a coffee-ordering application, at the very initial state, the user could either begin with a greeting to the system or launch straight into placing an order.

Below are a few example queries of possible greetings that a user could begin with:

#. Hello!
#. Hi!
#. Good morning!

While the number of greeting-like queries is a fairly small set, the number of ways in which a user could place an order are far more and can get arbitrarily complex.

E.g.

#. I want a chai tea latte.
#. Just a shot of espresso please.
#. Get me a grande mocha with whipped cream on top. 
#. Can I get a tall cappuccino with skimmed milk, one venti caramel macchiato, two chocolate chip muffins and a banana?

The representative set for "ordering" would theefore have to be a lot larger, since it would need to cover all the different examples of users specifying an order with varying amounts of detail. 

In addition, if you expect the system to handle other kinds of queries at the start state (e.g. reordering from history, checking order status, etc.), you would accordingly have to author more examples covering those as well.

We next look at adding more structure to the training data generation process by defining domains, intents and slots.


Defining Intents, Slots and Domains
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An intelligent conversational interface gives its users the flexibility to control and direct the conversation flow rather than forcing them to follow a fixed predetermined sequence of prompts. At every state in the dialogue, users are free to express whatever they want and the system responds by following the appropriate branch in the dialogue flowchart. 

We saw earlier that a user at the start state of a coffee-ordering app could either greet the system, place a new order, reorder from history, check his order status or say something else altogether. Each of these is called an Intent and corresponds to one of the possible intentions that the user may have at a given state. The intent chooses the transition from the current dialogue state to the next and subsequently determines the type of action to be taken by the system.

By convention, intent names are verbs since they denote some action that the user is trying to accomplish. 

Some example intents for the coffee-ordering app would be:

#. greeting
#. order
#. reorder
#. check_order_status

Apart from determining the user intent, the system would usually need more details from the user in order to fulfill their request. These additional pieces of information are called Slots. Each intent has its own set of related slots.

For instance, the "order" intent above can have slots such as "drink_name", "size", "quantity", etc. So when a user says "a tall grande mocha", the system extracts the relevant entities from the query and fills the corresponding slots: drink_name="mocha", size="tall", quantity="1".

Once you have identified all the intents across all dialogue states for your application, the next step is then to define the slots for each intent. While some slots could be mandatory, i.e. the system can't reach the goal state without filling those slots, others can be optional.

The last step is to group all related intents into a high level category called the Domain. Each domain has its own vocabulary, that is shared among  all the intents belonging to that domain. While it is possible to have all the intents under one single de-facto domain, the MindMeld Parser perfoms better when intent groups with vastly different language patterns and terminology are modeled under separate domains.

For instance, the conversational interface for a multimedia console may choose to model "Music", "Movies" and "Games" as separate domains. But a coffee-ordering application may only have a single domain called "Food and Drinks".


Generating Representative Queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clearly establishing the domain-intent-slots hierarchy for your conversational interface is crucial before you can start generating good quality training data. A good strategy is to work on one intent at a time and generate as many representative training examples for it as possible before moving on to the next one. The example queries should reflect actual in-the-wild usage and capture all expected syntax variations.

Here are a few examples for the "order" intent in a coffee-ordering app:

| # Keyword searches
| One tall peppermint hot chocolate
| Pumpkin spice latte

| # Declarative sentences
| I want a grande iced americano decaf.
| I want to order a tall white chocolate mocha.
| I feel like having a hot chocolate with lemon pound cake.

| # Imperative sentences
| Get me a chai tea latte.
| Order the pumpkin spice latte.
| Place an order for two bagels and a coffee.

| # Interrogative sentences
| Could I please have a grande gingerbread latte with extra whipped cream?
| Could you get me a chocolate croissant?
| How about a Pike Place roast?

You should aim to collect at least a few hundred queries by generating them yourself and getting help from friends and colleagues. Having others contribute ensures more variety in the training data. Once you have an initial set of queries covering all intents, you can follow the remainder of the chapters in the "Build" section to train an early version of your conversational interface.

While this early prototype may not have great accuracy or be ready for real-world usage, it will still be a great tool for you to interact with and start getting a feel for what the end product would look like. It will help you identify any missing dialogue states, intents or slots that you additionally need to capture. It will also help inform the training data generation process since you will be able to think from the perspective of a user in each dialogue state.

Once the end-to-end conversation flow is fixed and all dialogue states are precisely defined, you can train and pay human annotators to generate a lot more training data. A cheaper alternative to using skilled trained annotators is employing crowdsourcing platforms like Amazon Mechanical Turk. The success of such techniques depends on how clearly you can describe the data collection task to evoke the right responses from the crowd. It's hence important for you to have interacted with the prototype yourself, so you can define these tasks in a more informed way.

The end goal is to have a representative set of queries covering all intents, totalling up to at least tens of thousands, if not hundreds of thousands instances.



Annotate the Training Data
--------------------------

The MindMeld Entity Recognizer extracts relevants entities from user queries and assigns them to appropriate slot types. For instance, when a user says "I want to order a **tall** **decaf**  **latte**", the Entity Recognizer identifies **tall** as a size, **decaf** as an option and **latte** as a drink_name. 

In order to do this accurately at runtime, the Entity Recognizer needs to be trained to recognize and classify entities in free-form text. We do this by adding annotations to our training data to identify all the entities within our collected queries. We mark up the parts of the query that correspond to a slot, i.e. provide further information to the app about the current user intent.

Here are some examples:

{cinnamon dolce latte|name} {venti|size} {extra whip|option}

let me get a {double shot on ice|name}

{medium|size} {iced coffee|name} with {low fat milk|option}

The annotation markup syntax is fairly simple:

* Enclose the entity in curly braces
* Follow the entity with its type
* Use the pipe character as separator

A good strategy is to annotate a few hundred queries, train an initial Entity Recognizer using those examples and then use the trained Entity Recognizer to annotate new queries. That way, you will only need to fix errors made by the Entity Recognizer as opposed to generating annotations from scratch. You can retrain the Entity Recognizer at regular intervals and as the model gets better, you'll have lesser errors to correct while annotating new data.

Be consistent with your entity annotations and ensure that the annotated entity span matches with the entries in your Entity Map. Queries without any slot information should be left unannotated. E.g. "Sure", "Hello", "Yes, please".

Once you have a large representative training set that's annotated for entities and categorized by intent type, you're finally ready to build a commercial-grade conversational AI system



Tips and Best Practices
-----------------------

* In general, queries in any domain should be semantically and/or syntactically different from queries in all other domains. Each application should contain somewhere between one and several different domains.

* Defining the right set of intents for each domain is critical to ensure good accuracy and broad coverage of user queries. Each domain should have no more than a few dozen different intents.

* To optimize parser accuracy, the queries for each intent should should be semantically and/or syntactically different from the queries for all other intents

* Training data should capture all required intents and the queries themselves should capture all expected syntax variations, reflecting actual in-the-wild usage

* Entity annotations should be consistent and their spans should match the entries in the Entity Map file.

* Use trained Entity Recognizers to annotate and new queries and save human annotation time.


