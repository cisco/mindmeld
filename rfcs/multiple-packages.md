# MindMeld Multi Package Refactor

The MindMeld framework is a powerful solution to building natural language applications. Over time more and more functionality has been added, at times leveraging third-party dependencies. There are now situations in which this growing list of dependencies is an obstacle. Here are some examples of use cases where it would be preferable to include only a subset of MindMeld functionality (and therefore of the dependencies which it includes):

- developers of a large application decide they want to deploy their NLP models separately from their dialogue management logic so that they can scale them independently.
- developers of a pre-existing application want to migrate to MindMeld for NLP models but would like to keep their existing dialogue handling system.
- When starting a small application or proof of concept, developers don't want to bother with gathering training data but would prefer to leverage MindMeld's dialogue handling. (e.g. for a Webex Assistant Skill).

This is a proposal to subdivide MindMeld into multiple packages to allow MindMeld projects to easily be deployed as multiple processes, each minimizing thier dependencies and footprint.

## Overview

To support the aforementioned use cases we will separate the `mindmeld` package into multiple packages which can be added as dependencies directly.

- `mindmeld-core` will contain base data structures used by all MindMeld packages
- `mindmeld-nlp` will contain code for building and running natural language processing models
- `mindmeld-dialogue` will contain code for implementing the dialogue logic of MindMeld applications
- `mindmeld-qa` will contain code for implementing MindMeld's question answerer component.
- `mindmeld` will contain the above packages as dependencies, and will continue to work as it does now

## Package Names
There are two options for naming our new mindmeld packages after this refactor: traditional packages or namespace packages.

#### Traditional Packages

- With traditional packages we would give the new packages a `mindmeld_` prefix, e.g. `mindmeld_core` and `mindmeld_nlp`.
- To use a mindmeld package other than the primary package, a user would use it's specific name, i.e. `from mindmeld_nlp import NaturalLanguageProcessor`
- It is possible to implement the change while maintaining the existing `mindmeld` package API.

#### Namespace Packages

- With namespace packages all packages would use the `mindmeld` namespace.
- To use, for example, the nlp package a user would use `from mindmeld.nlp import NaturalLanguageProcessor` -- even if they have not installed the primary `mindmeld` package.
- It is unlikely that this can be implemented while maintaining the existing package API. This would require a major version bump.

## Code Migration Destinations

Here is an early plan for what code will belong in which of the new packages

- MindMeld Core
    - `mindmeld.core`
    - `mindmeld.components._config`
    - `mindmeld.components.request`
    - `mindmeld.components.schema`
    - `mindmeld.converter`
    - `mindmeld.cli`
    - `mindmeld.exceptions`
    - `mindmeld.path`
    - `mindmeld.resource_loader`
    - `mindmeld.server`
    - `mindmeld._util` (blueprint code)
    - a new interface for NaturalLanguageProcessor objects

- MindMeld NLP
    - `mindmeld.auto_annotator`
    - `mindmeld.components.classifier`
    - `mindmeld.components.domain_classifier`
    - `mindmeld.components.entity_recognizer`
    - `mindmeld.components.entity_resolver`
    - `mindmeld.components.intent_classifier`
    - `mindmeld.components.nlp`
    - `mindmeld.components.parser`
    - `mindmeld.components.preprocessor`
    - `mindmeld.gazetteer`
    - `mindmeld.models`
    - `mindmeld.markup`
    - `mindmeld.query_cache`
    - `mindmeld.query_factory`
    - `mindmeld.ser`
    - `mindmeld.stemmers`
    - `mindmeld.system_entity_recognizer`
    - `mindmeld.tokenizer`
- Mindmeld Dialogue
    - `mindmeld.app`
    - `mindmeld.app_manager`
    - `mindmeld.components.client`
    - `mindmeld.components.custom_action`
    - `mindmeld.components.dialogue`
- MindMeld Question Answering
    - `mindmeld.components._elasticsearch_helpers`
    - `mindmeld.components.question_answering`
    - `mindmeld.models` (embedding related code)