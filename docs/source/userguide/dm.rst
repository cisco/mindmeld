.. meta::
    :scope: private

Dialogue Manager
================

In contrast to other parts of the system that are stateless, the Dialogue Manager is stateful and maintains information about each state or step in the dialogue flow. It is therefore able to use historical context from previous conversation turns to move the dialogue along towards the end goal of satisfying the user's intent.


Architecting the dialogue manager correctly is often one of the most challenging software engineering tasks when building a conversational app for a non-trivial use case.