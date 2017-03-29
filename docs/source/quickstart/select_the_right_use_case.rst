Step 1: Select the Right Use Case
=================================

Selecting the right use case is perhaps the most important step in building a conversational application that users will love. For many use cases, a voice or chat conversation can make it simpler to find information or accomplish a task. For many others, a conversational interface can be inconvenient or frustrating. Selecting an unrealistic or incorrect use case will render even the smartest voice or chat assistant dead on arrival.

While there is no magic formula to determine which use case is ideal for a conversational interface, some best practices have begun to emerge for distinguishing the good candidates from the bad. To ensure that your conversational application is practical to build and provides real value to users, it is important to ask the following questions *before* selecting a use case.

===================================================== ===
**Does it resemble a real-world human interaction?**  Conversational interfaces do not come with instruction manuals, and there is little opportunity to teach users about the supported functionality before they take it for a spin. The best use cases mimic an existing, familiar real-world human interaction so that users intuitively know what they can ask and how the service can help. For example, a conversational interface could mimic an interaction with a bank teller, a barista or a customer support rep.

**Does it save users time?**                          Conversational interfaces shine when they save users time. A conversational interface is viewed as an unwelcome impediment when a well-designed GUI would be faster. The most useful conversational experiences often center around a use case where users are looking to accomplish a specific task and know how to articulate it. For example, simply saying 'play my smooth jazz playlist in the kitchen' can be much faster than launching an app and navigating to the equivalent option by touch.

**Is it more convenient for the user?**               Voice interfaces can be particularly useful when users' hands and attention are occupied or no mobile device is within reach. If it makes sense to use your application while driving, biking, walking, exercising, cooking, or sitting on the couch, that application is likely to be an excellent candidate for a conversational interface.

**Does it hit the Goldilocks zone?**                  The best conversational applications fall squarely into the 'Goldilocks zone.' The range of functionality that they offer is both narrow enough to ensure that the machine learning models achieve high accuracy, and broad enough that users find the experience useful for a wide variety of tasks. Apps that are too narrow can be trivial and useless. Apps that are too broad can frustrate users by managing only hit-or-miss accuracy.

**Is it possible to get enough training data?**       Even the best use cases fail when it is not possible or practical to collect enough training data to reflect the full range of envisioned functionality. For ideal use cases, training data can be easily generated from production traffic or crowdsourcing techniques. If training data for your use case can only be sourced from a small number of hard-to-find human experts, it is not likely to be a good candidate for a conversational interface.
===================================================== ===

In this Guide we consider a simple conversational use case about a neighborhood retail store called Kwik-E-Mart. For instance, you could ask this service about store hours: 'What time does the Kwik-E-Mart on Elm street close today?' This rudimentary use case will serve as a reference example to highlight the key steps in building a useful conversational interface.
