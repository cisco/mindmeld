# Regex feature

A feature based on regex pattern matching that MindMeld's NER models can use to detect custom entities. 

- Developers expect NER models to correctly infer that the query `AVET128-321` is a `pid` entity 
without providing surrounding sentence context to the classifier. This is more relevant in services that
just use MindMeld for NLP while doing their own dialog management. 
- Developers also want granular distinctions between entities that occur in the same context, 
for example, `look up {SKU120|sku}` and `look up {123-ABC-12|pid}`. Currently, we ask developers to defer
these types of distinction to the business logic in the dialogue manager.
- DialogFlow, NLTK and other similar platforms have the regex matching feature. Moreover, we already have an 
orthographic matching feature in MindMeld. This feature would augment that.


## Usage

The developer makes a new file called `regex_matches.txt` for the entity type in the entities folder that needs 
these regex matches like this following:

```text
entities
├── account_type
│   ├── gazetteer.txt
│   ├── mapping.json
│   └── regex_matches.txt
├── autopay_status
│   └── mapping.json
└── credit_amount
    └── mapping.json
```

Inside the `regex_matches.txt` file, the developer lists regex patterns compliant with python syntax that need 
to be matched for this custom entity. 

```text
\w{4}[0-9]{3}\-[0-9]{3}
[0-9]{3}\-[0-9]{3}
```

In the above syntax, the `account_type` entity would match words like `ABCD-123-281` (first pattern) 
and `123-456` (second pattern).

Finally, the developer would have the ability to turn on/off the regex feature through the model config in 
config.py using the key,value `regex-seq: True`. The feature will be off by default.

```python
ENTITY_RECOGNIZER_CONFIG = {
    "features": {
        "bag-of-words-seq": {
            "ngram_lengths_to_start_positions": {
                1: [-2, -1, 0, 1, 2],
                2: [-2, -1, 0, 1],
            }
        },
        "sys-candidates-seq": {"start_positions": [-1, 0, 1]},
        "regex-seq": True
    }
}
```

If one is using the `view_extracted_features` method of the entity classifier to inspect the features of a query,
the following result would be shown:

```python
er.view_extracted_features("look up 123-456")
```

```text
[{...}, {...}, {'regex_match|type:account_type:1', ...}]
```

## Problems with this approach

- The processing time during inference will increase significantly since each input token has to be matched against 
each regex pattern for every entity type.
- The training time will also increase for the similar reason
- The feature could cast a wide net on every token, for example `.*` would match against all tokens. However since the 
classifier would able to learn from this high noise signal, it would reduce it's dependence on it to make a classification.
- There could be overlaps with regex patterns from different entity types, but the classifier would be able to learn from
those patterns.
- This approach locks-in the one-language one-mindmeld application idea since these regex patterns will not translate
across languages.

## Alternative ideas

- Have the regex mapping in `config.py` instead. This idea is a bit confusing since it doesnt localize all the entity 
logic in the same folder location.
- Not have this feature since it could be abused and developers will be confused. Moreover, the performance costs could 
add up to an already slow platform.

