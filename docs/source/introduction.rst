Introduction to Conversational Applications
===========================================

History of Conversational AI technology
---------------------------------------

People have been working on this technology for over 50 years.
It has only taken off in the past 12 months due to recent breakthroughs in AI.
Great experiences are now possible, but this is one of the most complex technology areas in computer science.

Current uses of Conversational AI
---------------------------------

Conversational AI applications are starting to appear everywhere:
built into native apps and websites
built into messaging platforms like FB Messenger, Skype, iMessage, etc.
invoked through virtual assistants like Google Assistant, Siri, Cortana & Alexa
available on devices like Amazon Echo, Google Home, etc.
available in the connected home via Google Home, Apple HomeKit, Amazon AVS, etc.
available in the car while driving via Android Auto and Apple Car Play
available via in-store kiosks
built into enterprise collaboration platforms like Slack, Spark, etc.
built into customer service software like Service Cloud, Dynamics 365, etc.

Conversational AI is not for everything, but there are certain situations where it can be quite useful:  when hands are busy, when on the go, when navigating through large content collections, when a touchscreen is not handy.

Describe the domains where Conversational AI can be useful: messaging, calendaring, alarms, weather, entertainment discovery, traffic,  general knowledge, ordering and booking, collaboration, customer support, etc.

Over the next couple years, hundreds of specialized conversational assistants will emerge to help us with a wide range of daily tasks.


Anatomy of a conversational AI interaction
------------------------------------------

Illustrate and describe the key technologies which are required at each point in a typical conversational AI interaction:
Step 1: Wake-up word activates device.
Describe that this is done using low-power, embedded keyword spotting technology.
Step 2: Speech recognition converts spoken utterance into a text transcript.
Note that ASR is available via a variety of paid cloud servers or available for free in all major OSs and browsers.
Step 3: Machine learning classifiers determine the domain and intent of the transcript.
Step 4: Entity recognizer models identify and resolves important entities in the transcript.
Step 5: Semantic parser determines the relationships between entities in the transcript.
Step 6: Question answerer validates, retrieves and ranks candidate answers for the request.
Step 7: Dialogue manager determines the correct form of response.
Step 8: Natural language generator converts response to human-language transcript.
Step 9: Text-to-speech technology speaks the response to the user.
Also note that this is typically build into the OS.
Step 10: On-screen display shows answers visually in real time.

Machine learning for language understanding 
-------------------------------------------

Describe the potential and challenges associated with using machine learning to understand language.  
Current production applications rely on supervised learning models trained on data. 
Deep learning can be used if you have lots and lots of data. 
For many NLP domains, it is difficult or impossible to get the training data you need.  
NLP models are very susceptable to the curse of dimensionality, and therefore, large-scale measurement and analytics are the only way you can tell if your app will work sufficiently well across the long tail of user queries.


Different approaches for building conversational applications
--------------------------------------------------------------

Describe different strategies for building apps. For very simple bots, rule-based approaches suffice. For more sophisticated and useful bots, rule-based approaches break down and machine learning is required. Cloud-based NLP services are an easy way to train ML models on small sets of data. Unfortunately, to build a truly useful bot, you typically need much larger data sets than these Cloud-based tools are intended to handle. In those cases, more advanced machine learning toolkits are required. General-purpose ML toolkits like TensorFlow and GraphLab are not well suited for building Conversational AI applications.  MindMeld is a better choice is you are looking to build an advanced voice or chat assistant.

Rule-based approaches
^^^^^^^^^^^^^^^^^^^^^
Outline the pros and cons of rule-based bot platforms like microsoft bot framework and bot kit.

Cloud-based NLP services
^^^^^^^^^^^^^^^^^^^^^^^^
Outline the pros and cons of api.ai, wit.ai, amazon lex, microsoft luis.

Machine-learning toolkits
^^^^^^^^^^^^^^^^^^^^^^^^^
Outline the pros and cons of tensorflow, graphlab, nltk, etc.


Unique requirements for production conversational interfaces
------------------------------------------------------------

Building a conversational interface seems simple on the surface, but getting it right is one of the hardest AI challenges solvable today.  The production requirements for conversational interfaces are atypical of mobile or web apps:
they need to have near-perfect accuracy: apps will be effectively unusable until the reach a threshold of 95% accuracy or better
they require large amounts of training data: Small data sets beget trivially simple or brittle functionality. Users are unforgiving when behavior is less than human-like.
they require large-scale machine learning: Large data sets mandate large-scale ML techniques. This is the only approach proven to work in commercial apps.
they require careful management of user expectations: Without a guiding visual UI, users are often at a loss for words. The best use cases mimic a familiar, real-world interaction.
Conversational interfaces are binary. They are either useful or useless. There is rarely any middle ground. 


Advantages of MindMeld Workbench
--------------------------------

MindMeld ensures that you always maintain ownership and control of the training data and models which power your application  
real, production applications require lots of training data, and MindMeld provides the necessary utilities and analytics to manage large training data sets
high-quality, representative training data is the most important thing to ensuring a good experience, and MindMeld provides necessary tools to collect and QA training data via crowdsourcing
MindMeld is the only platform available today which provides a complete question answering and dialogue management system along with advanced natural language parsing capabilities
MindMeld's knowledge-driven learning approach is ideally suited for domains which involve a large product or content catalog
unlike UI-based NLP tools which are often too rigid to accommodate the functionality required in your application, MindMeld's flexible and powerful architecture can accommodate just about any application
