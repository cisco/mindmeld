Dealing with Voice Inputs
=========================

MindMeld provides all the functionality required to build a text-based natural language
chat interface. However, you might also want to support voice interactions with your application
or build for a platform where voice is the only input modality. For such applications, you can
leverage a third party
`Automatic Speech Recognition <https://en.wikipedia.org/wiki/Speech_recognition>`_ (ASR) system to
convert the input speech into text. There are multiple third party speech recognition systems available such as those from `Google <https://cloud.google.com/speech-to-text/>`_,
`Hound <https://soundhound.com/houndify>`_,
`Nuance <https://www.nuance.com/mobile/speech-recognition-solutions.html>`_, and
`Microsoft <https://azure.microsoft.com/en-us/services/cognitive-services/speech/?v=18.05>`_.
The converted text transcript can then be processed by your MindMeld application to return an
appropriate text response. Finally, you can send this text response to a third party
`Text to Speech <https://en.wikipedia.org/wiki/Speech_synthesis>`_ (TTS) system to synthesize an
audio response that can be "spoken" back to the user.  Similar to speech recognition, there are multiple third party TTS systems available for use, including but not limited to the services
provided by `Amazon <https://aws.amazon.com/polly/>`_, 
`Google <https://cloud.google.com/text-to-speech/>`__ and
`Microsoft <https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/>`__.


Challenges with speech recognition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It's important to note that speech recognition is not perfect, especially for domain-specific
vocabulary and proper nouns. ASR errors may cause the text input to your MindMeld application to
be something different than what the user intended, which can lead to an unexpected response that
makes the app appear unintelligent. The part of the pipeline that is most susceptible to a drop in
accuracy due to ASR errors is the entity resolution step. This is because domain and intent
classification rely more on sentence structure and context words which generic speech recognition
systems tend to get right. However, domain-specific entities, which are generally less common
words or proper nouns, are likely to be mistranscribed by generic third party speech recognition
systems. These terms are often transcribed to tokens that are phonetically similar but textually
significantly different from what was said.

One way to overcome this problem is by building your own custom domain ASR system if you have
enough audio data available for training. This is a larger task which we won't describe here, but
there are a variety of open source models available as a starting point including
`Mozilla Deep Speech <https://github.com/mozilla/DeepSpeech>`_,
`CMUSphinx <https://cmusphinx.github.io/>`_, and `Kaldi <https://github.com/kaldi-asr/kaldi>`_.
However, given the cost and effort associated with building ASR models from scratch, the most
common scenario is to use an out-of-the-box ASR. In the following sections, we will describe a
couple of techniques you can leverage in MindMeld to maintain a high entity resolution accuracy
despite speech recognition errors.


Phonetic matching in entity resolution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The MindMeld Entity Resolver is optimized for typed inputs by default. It can handle text
variations like typos and misspellings, and it leverages synonym lists to resolve terms that are
semantically similar. In addition to these variations, for voice inputs, there is a large category
of entity variations that are phonetically similar but textually different from the canonical form.
For example, the entity "Arnold Schwarzenegger" may be mistranscribed to "our old shorts hanger".
These two terms sound similar but have little character overlap. To resolve these types of
mistranscriptions, you can enable phonetic matching in the entity resolver by specifying the
``phonetic_match_types`` parameter in your entity resolution config.

The value corresponding to the ``phonetic_match_types`` key is a list of phonetic encoding
techniques to use for entity matching. There are a few common techniques that are used to generate
the phonetic representation of text, of which, one of the most optimized and efficient is the
`Double Metaphone <https://en.wikipedia.org/wiki/Metaphone#Double_Metaphone>`_ algorithm. Double
Metaphone is based on a series of rules optimized for indexing the phonetic representations of
names of various origins. Currently, ``double_metaphone`` is the only supported value for 
``phonetic_match_types``. It can be specified in your application configuration file (config.py)
as follows.

.. code-block:: python

    ENTITY_RESOLVER_CONFIG = {
        'model_type': 'text_relevance',
        'phonetic_match_types': ['double_metaphone']
    }


If the ``phonetic_match_types`` key is specified in the entity resolution config, the resolver
tries to phonetically match extracted entity spans against your entity index in addition to using
the default text-based matching techniques. This can improve the relevance of the ranked results
returned by the entity resolver as illustrated in the following example.

.. note::
    - In order to utilize phonetic matching, you must install the phonetic analysis plugin for
      Elasticsearch and restart the Elasticsearch service. Refer to `Elasticsearch documentation
      <https://www.elastic.co/guide/en/elasticsearch/plugins/current/analysis-phonetic.html>`_
      for instructions. You may also have to delete the default Elasticsearch template by running
      this command in your shell: ``curl -X DELETE "localhost:9200/_template/default"``.

    - After first introducing ``phonetic_match_types`` to the entity resolver config, restart your
      Python shell and rebuild the entity resolver index from scratch by running a clean fit
      (``er.fit(clean=True)``). See the :doc:`Entity Resolver <entity_resolver>` page for
      additional details.

Consider a setting where the user says "I want to eat some Pad Thai and Yellow Curry", and the ASR
transcribes it as "i want to beat some pad thai and mellow Kerrie". Let us see the resolved values
for the span "mellow Kerrie" without using phonetic matching.

.. code-block:: python
   :emphasize-lines: 7-8

   from mindmeld import configure_logs; configure_logs()
   from mindmeld.components.nlp import NaturalLanguageProcessor
   from mindmeld.core import Entity
   nlp = NaturalLanguageProcessor(app_path='food_ordering')
   nlp.build()
   nlp.dump()
   er = nlp.domains['ordering'].intents['build_order'].entities['dish'].entity_resolver
   er.predict(Entity(text='mellow Kerrie', entity_type='dish'))

.. code-block:: console
   :emphasize-lines: 33-36

   {'cname': '60. Crispy Fried Portobello Mushroom',
    'id': 'B01CLQSAVG',
    'score': 21.071535,
    'top_synonym': '60. Crispy Fried Portobello Mushroom'},
   {'cname': '61. Crispy Fried Portobello Mushroom',
    'id': 'B01CLQS9ZI',
    'score': 19.876467,
    'top_synonym': '61. Crispy Fried Portobello Mushroom'},
   {'cname': 'Fried Mozzarella',
    'id': 'B01ENMPJPG',
    'score': 19.337563,
    'top_synonym': 'Fried Mozzarella'},
   {'cname': 'Fried Mozzarella Sticks',
    'id': 'B01N18BT3F',
    'score': 18.817226,
    'top_synonym': 'fried mozzarella'},
   {'cname': 'Twin Scallops Fried Rice',
    'id': 'B01CIKMRB4',
    'score': 18.401768,
    'top_synonym': 'scallops fried rice'},
   {'cname': '36. Spicy Prawn & Crispy Fried Portobello Mushroom',
    'id': 'B01CLQSEYE',
    'score': 18.10899,
    'top_synonym': '36. Spicy Prawn & Crispy Fried Portobello Mushroom'},
   {'cname': 'Hamachi (Yellow Tail)',
    'id': 'B01MRRJDRC',
    'score': 15.289129,
    'top_synonym': 'Hamachi (Yellow Tail)'},
   {'cname': 'Yellow Sea',
    'id': 'B01CPOE9BE',
    'score': 14.5856,
    'top_synonym': 'Yellow Sea'},
   {'cname': 'Yellow Curry',
    'id': 'B01CPOEBC6',
    'score': 14.556676,
    'top_synonym': 'Yellow Curry'},
   {'cname': 'Tuna Melt Sandwich',
    'id': 'B01CH0SPK2',
    'score': 14.51431,
    'top_synonym': 'tuna melt grinder with fries'}]

In the absence of phonetic information, the resolution results do not resemble what the user
originally said. You can see that the top result has character overlaps with the mistranscription
("ello" , "rie" , etc.), but it is clearly not what the user intended ("Yellow Curry"). There is
just enough character overlap to rank "Yellow Curry" in the ninth spot but the remaining results
are unrelated ("Fried Portobello", "Fried Mozzarella", etc.).

Next, let us see the resolved values for "mellow Kerrie" with phonetic matching enabled in the
config.

.. code-block:: python
   :emphasize-lines: 7-9

   # After updating app config and restarting the Python shell
   from mindmeld import configure_logs; configure_logs()
   from mindmeld.components.nlp import NaturalLanguageProcessor
   from mindmeld.core import Entity
   nlp = NaturalLanguageProcessor(app_path='food_ordering')
   nlp.load()
   er = nlp.domains['ordering'].intents['build_order'].entities['dish'].entity_resolver
   er.fit(clean=True)
   er.predict(Entity(text='mellow Kerrie', entity_type='dish'))

.. code-block:: console
   :emphasize-lines: 1-4,13-16,33-40

   {'cname': 'Yellow Curry',
     'id': 'B01CPOEBC6',
     'score': 25.13264,
     'top_synonym': 'Yellow Curry'},
    {'cname': '60. Crispy Fried Portobello Mushroom',
     'id': 'B01CLQSAVG',
     'score': 21.071535,
     'top_synonym': '60. Crispy Fried Portobello Mushroom'},
    {'cname': '61. Crispy Fried Portobello Mushroom',
     'id': 'B01CLQS9ZI',
     'score': 19.876467,
     'top_synonym': '61. Crispy Fried Portobello Mushroom'},
    {'cname': '79. Kao Pad Goong Pong-Ga-Ree Fried Rice',
     'id': 'B01LY4ZA0M',
     'score': 19.338999,
     'top_synonym': 'yellow curry and shrimp fried rice'},
    {'cname': 'Fried Mozzarella',
     'id': 'B01ENMPJPG',
     'score': 19.337563,
     'top_synonym': 'Fried Mozzarella'},
    {'cname': 'Fried Mozzarella Sticks',
     'id': 'B01N18BT3F',
     'score': 18.817226,
     'top_synonym': 'fried mozzarella'},
    {'cname': 'Twin Scallops Fried Rice',
     'id': 'B01CIKMRB4',
     'score': 18.401768,
     'top_synonym': 'scallops fried rice'},
    {'cname': '36. Spicy Prawn & Crispy Fried Portobello Mushroom',
     'id': 'B01CLQSEYE',
     'score': 18.10899,
     'top_synonym': '36. Spicy Prawn & Crispy Fried Portobello Mushroom'},
    {'cname': 'Panang Curry (Over Rice)',
     'id': 'B01DV7324O',
     'score': 17.12096,
     'top_synonym': 'Creamy yellow tofu curry'},
    {'cname': 'Roti with Curry',
     'id': 'B01LX5THED',
     'score': 15.802841,
     'top_synonym': 'roti with yellow curry'}]

These results look more reasonable. The top result exactly matches the user's intended dish,
"Yellow Curry" due to its high phonetic similarity to the extracted entity "mellow Kerrie". Many
other results have also been ranked higher due to phonetic matches against the canonical name or 
the synonym list.

.. _nbest_lists:

Leveraging ASR n-best lists
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Almost all out-of-the-box third party ASR APIs return a ranked list of multiple possible
transcripts, also called an *n-best list*. For example, if the user said "Look for movies
directed by Tarantino", the n-best list of recognition hypotheses may look like the following:

.. code-block:: text

    ['look for movies directed by Terren Tina',
     'look for movies directed by Darren Tina',
     'look for movies directed by Tarantino',
     'look for movies directed by tear and tea no',
     'look for movies directed by Terren teen']

This list of transcripts represents the top guesses by the speech recognition language model given
the phonemes in the audio file. For uncommon terms, the correct transcription may be in a lower
transcript or not in the list at all. However, using the phonetic information from the n-best list
and app-specific context, we can resolve entities to their intended values with a high accuracy.

MindMeld provides the option to pass in the n-best list of ASR transcripts for extracting
multiple candidate entities to improve entity resolution.

To leverage this functionality add an ``NLP_CONFIG`` dictionary to your application configuration
file (``config.py``) as follows.

.. code-block:: python

    NLP_CONFIG = {
        'resolve_entities_using_nbest_transcripts': ['video_content.*']
    }

Specify the domains and intents of interest by including them as an element in the list value
corresponding to the ``resolve_entities_using_nbest_transcripts`` key. The format is
``<domain>.<intent>``, and an asterisk ('*') wildcard denotes *all* intents within the specified
domain.

MindMeld will limit running the n-best enhanced entity resolution to the domains and intents you
have specified. This is an optimization to avoid unnecessary processing of a factor of 
`n` for queries without entities of interest. While the code is parallelized for minimal latency
increase, there will be an increase in memory usage from the domains and intents for which n-best
entity processing is run. You can control the parallel processing behavior using the
:ref:`MM_SUBPROCESS_COUNT <parallel_processing>` enviroment variable.

Also make sure that you have phonetic matching enabled for the entity resolver in your app config.

.. code-block:: python

    ENTITY_RESOLVER_CONFIG = {
        'model_type': 'text_relevance',
        'phonetic_match_types': ['double_metaphone']
    }

Once the app config is updated, you can pass in a list or tuple of strings to :meth:`nlp.process()`
instead of a single string. If the top transcript gets classified as one of the domains or intents
specified for n-best enhanced entity resolution, information from the entire n-best list will be
used for resolving the entity.

Let us see the results of n-best enhanced entity resolution for the above example where the user said "Look for movies directed by Tarantino". Note that we pass the entire the n-best list of ASR transcripts to :meth:`nlp.process`.

.. code-block:: python

   nlp.process(['look for movies directed by Terren Tina',
                'look for movies directed by Darren Tina',
                'look for movies directed by Tarantino',
                'look for movies directed by tear and tea no',
                'look for movies directed by Terren teen'])

.. code:: console
   :emphasize-lines: 7-10,22,30,51

    {
     'domain': 'video_content',
     'entities': [{'role': None,
       'span': {'end': 39, 'start': 28},
       'text': 'Terren Tina',
       'type': 'director',
       'value': [{'cname': 'Quentin Tarantino',
         'id': 'B01CPOEKPY',
         'score': 176.69968,
         'top_synonym': 'Tarantino'},
        {'cname': 'Tina Mabry',
         'id': 'B01G67O8GM',
         'score': 128.46222,
         'top_synonym': 'Tina'},
        {'cname': '51. Darren Aronofsky',
         'id': 'B01LXTA7WA',
         'score': 42.02176,
         'top_synonym': 'Darren'},
         ...
         ]}],
     'intent': 'browse',
     'nbest_aligned_entities': [
        [{'text': 'Terren Tina', 'type': director},
         {'text': 'Darren Tina', 'type': director},
         {'text': 'Tarantino', 'type': director},
         {'text': 'tear and tea non', 'type': director},
         {'text': 'Terren teen', 'type': director}
        ]
      ],
     'nbest_transcripts_entities': [
        [{'role': None,
          'span': {'end': 39, 'start': 28},
          'text': 'Terren Tina',
          'type': 'director'}],
        [{'role': None,
          'span': {'end': 39, 'start': 28},
          'text': 'Darren Tina',
          'type': 'director'}],
        [{'role': None,
          'span': {'end': 37, 'start': 28},
          'text': 'Tarantino',
          'type': 'director'}],
        [{'role': None,
          'span': {'end': 43, 'start': 28},
          'text': 'tear and tea non',
          'type': 'director'}],
        [{'role': None,
          'span': {'end': 39, 'start': 28},
          'text': 'Terren teen',
          'type': 'director'}]],
     'nbest_transcripts_text': [
        'look for movies directed by Terren Tina',
        'look for movies directed by Darren Tina',
        'look for movies directed by Tarantino',
        'look for movies directed by tear and tea no',
        'look for movies directed by Terren teen'],
     'text': 'look for movies directed by Terren Tina'}

You can see that the query was classified as the ``video_content`` domain and the ``browse``
intent. Since all intents in the ``video_content`` domain were specified in the
``NLP_CONFIG`` above, MindMeld ran n-best entity processing for this query. This involves running
entity recognition on all the n-best transcripts and using information from the all of the
extracted entities for entity resolution.

In this example, the n-best transcripts had multiple examples of phonetic matches to
"Tarantino", and one of the hypotheses even had the exact correct transcription of "Tarantino". By
using the entities extracted from all of these transcripts, the entity resolver was able to
correctly get the top entity as "Quentin Tarantino". Without using n-best entity resolution, the
phonetic matching against just the top transcript "Terren Tina" may not be enough to differentiate
between similar names like "Darren Lima". The n-best transcripts often provide additional
phonetic information to improve the accuracy of resolving to the intended entity.

While the built-in entity resolver that leverages phonetic information and n-best transcripts is a
great starting point for dealing with ASR errors, in many cases you can further improve accuracy
by leveraging application-specific context. To enable this, the NLP response includes a few
additional fields that you can you use in the dialogue manager as you see fit:

+------------------------------------+------------------------------------------------------------+
| Key                                | Description                                                |
+====================================+============================================================+
| :data:`nbest_transcripts_text`     | The input list of n-best transcripts.                      |
+------------------------------------+------------------------------------------------------------+
| :data:`nbest_transcripts_entities` | A list of lists, one for each input transcript. Each       |
|                                    | sublist contains a list of extracted entities for that     |
|                                    | transcript.                                                |
|                                    |                                                            |
|                                    | For example, "Terren Tina" is the extracted ``director``   |
|                                    | entity from the first transcript "look for movies          |
|                                    | directed by Terren Tina", "Darren Tina" is the extracted   |
|                                    | ``director`` entity from the second transcript "look for   |
|                                    | movies directed by Darren Tina", and so on.                |
+------------------------------------+------------------------------------------------------------+
| :data:`nbest_aligned_entities`     | A list of lists, one for each detected entity in the input.|
|                                    | Each sublist contains the text spans extracted across all  |
|                                    | the n-best transcripts for that particular entity.         |
|                                    |                                                            |
|                                    | This is useful for queries with multiple entities like     |
|                                    | "Order pad thai and spring rolls please" where both        |
|                                    | "pad thai" and "spring rolls" are ``dish`` entities. In    |
|                                    | that case, the first entry would be a list of all text     |
|                                    | spans for the entity "pad thai" extracted across all the   |
|                                    | n-best transcripts and the second entry would similarly be |
|                                    | a list of all the text spans for "spring rolls".           |
+------------------------------------+------------------------------------------------------------+

For example, you can build an app-specific entity resolver that is called from the dialogue
manager which uses all the n-best entity spans along with phonetic matching to resolve to the
correct term. To derive phonetic representations for your extracted entities, you can leverage the
`double metaphone <https://en.wikipedia.org/wiki/Metaphone#Double_Metaphone>`_ algorithm (used by
the MindMeld entity resolver) or a more advanced machine-learned model like
`grapheme to phoneme <https://github.com/cmusphinx/g2p-seq2seq>`_.

.. note::

    The domain and intent classification models solely use the top transcript to make a prediction.
    The n-best transcripts are only leveraged for entity processing since those are the parts of
    the NLP pipeline most susceptible to errors due to ASR mistranscriptions.

.. note::

    While using phonetic matching and n-best transcripts will improve accuracy for entity
    resolution on voice inputs, these approaches are not perfect. They heavily depend on the
    quality of the ASR transcripts which varies with the vendor used, the background noise of the
    environment, the quality of the recording device, etc. You may want to additionally include
    some application-specific post processing to verify that the resolved entities are reasonable
    for your use case.
