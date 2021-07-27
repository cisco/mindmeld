# Disallowed intents feature

This feature blacklists certain MindMeld intents such that these intents will never be picked during
inference.

## Problem

Purely follow-up intents can get triggered on the first turn:

```
user >> the beatles
bot >> cannot find a restaurant titled "the beatles"
```
The intent classifier miss-classifies such queries to `confirm_restaurant_name` intent instead
of the `unknown` domain.


The "happy path" of such conversations are as follows:

```
user >> find a restaurant
bot >> what restaurant
user >> pappadeux
```

Here is a sample NLP hierarchy of such a followup conversation:

```
.
└── restaurants
    ├── find_restaurants
    ├── confirm_restaurant_name
└── unknown
    ├── unknown
```

In the above case, the dialogue handler is constructed as a dialogflow named `find_restaurants`.

```python
@app.dialogue_flow(domain='restaurants', intent='find_restaurants')
async def find_restaurants(ctx, responder):
  pass

@find_restaurants.handle(intent='confirm_restaurant_name')
async def confirm_restaurant(ctx, responder):
  pass
```

The `confirm_restaurant_name` intent has the following training queries:
```
Burger King
Mcdonalds
Chick-fil-A
...
```

## Proposed solution

For purely followup intents like `confirm_restaurant_name`, which are defined as intents whose dialogue handlers are
only associated with a dialogflow and not as an independent dialogue handler, we mask those intents for all predictions
that are not in the associated dialogflow.

```
user >> the beatles

[allowed_intents = all intents except `confirm_restaurant_name` since it's associated with find_restaurants df]

bot >> I dont know what that means

user >> find restaurants

[allowed_intents = all intents except `confirm_restaurant_name` since it's associated with find_restaurants df]
[find_restaurants df is activated now]

bot >> what restaurant

[allowed_intents = all intents and `confirm_restaurant_name` since it's associated with find_restaurants df]
[find_restaurants df is activated now]

user >> the beatles

bot >> cannot find a restaurant titled "the beatles"
```

This solution will have no visible API changes.

## Important note:

We have not considered biasing the classifiers in this discussion, as compared to complete blacklisting. Hence, we ignore
the cases were the true intent was a follow-up but we classified something else. For example:

```
user >> find a restaurant
bot >> what restaurant
user >> oasis place [intent: unknown, true_intent: confirm_restaurant_name]
bot >> I dont know what that means
```

NLP biasing is complicated since we need to agree on hyperparams to judge the closeness of NLP predictions and when
to prefer a lower ranked NLP component prediction to a higher ranked one.


## API Design

We have to standardize and refactor the existing `allowed_intents` field since its being used for different purposes like
masking roles. Here are some design proposals on that:


```python
nlp.process('I want to move $190 to checkings',
            allow_nlp={'domain.intent', 'domain2', 'domain.intent2.entity'},
            deny_nlp={'domain.intent3', 'domain4', 'domain.intent1.entity'},
        )
```

The design collapses the allow and deny lists, so that the developer can specify arbitrary nlp representations.

```python
nlp.process('I want to move $190 to checkings',
            allow_intents=['domain.intent'],
            allow_entities=['domain.intent.entity'],
            allow_domains=['domain'],
            allow_roles=['domain.intent.entity.roles'],
            deny_intents=[],
            deny_entities=[],
            deny_domains=[],
            deny_roles=[]
        )
```

This design makes it explicit what NLP components are being masked. However, its very verbose.

Other designs:

```python
nlp.process('I want to move $190 to checkings',
              mask_nlp={
                'allow': {
                    'domain': ['domain'],
                    'intents': ['domain.intent'],
                    'entities': ['domain.intent.entity'],
                    'roles': ['domain.intent.entity.roles']
                },
                'deny': {
                    'domain': [],
                    'intents': [],
                    'entities': [],
                    'roles': []
                }
              }
            )
```


```python
nlp.process('I want to move $190 to checkings',
            mask_nlp={
                'intents': {
                  'allowed': ['domain.intent'],
                  'deny': []
                },
                'domains': {
                  'allowed': ['domain'],
                  'deny': []
                },
                'entities': {
                  'allowed': [],
                  'deny': []
                },
                'roles': {
                  'allowed': [],
                  'deny': []
                }
            }
        )
```



#### Implementation details:

To maintain a clear separation of concerns between the dialogue management system and the NLP components,
the dialogue manager will pass context to the NLP components through the `disallowed_intents` field to either
consider a particular NLP component or not.

During application load time, the system determines which NLP components are purely followup intents defined
as follows:
1. An NLP component whose dialogue handlers are all associated with a dialogflow

These NLP components are then never considered for NLP processing unless the dialogflow under which it
is associated is activated. The NLP component will then be removed from the `disallowed_intents` until
the dialogflow is de-scoped. We will also merge any user defined `disallowed_intents` with the DM derived
`disallowed_intents` as follows:

```python
processed_query = self.nlp.process(query_text=text,
                                   allowed_intents=params.allowed_intents,
                                   disallowed_intents=merge(params.disallowed_intents, dm_derived_disallowed_intents),
                                   locale=params.locale,
                                   language=params.language,
                                   time_zone=params.time_zone,
                                   timestamp=params.timestamp,
                                   dynamic_resource=params.dynamic_resource,
                                   verbose=verbose)
```

The API for the `disallowed_intents` is similar to `allowed_intents` with the following pattern: `domain.intent.entity.role`.
The API changes will be included in `nlp.process`.

Due to the static nature of the DM, there will be applications where `disallowed_intents` will be used all the time.
Since both `allowed_intents` and `disallowed_intents` need to use `predict_proba` for their processing, there is a
concern of latency of inference going up. Due to this, we characterized the latency of SKLearn model's `predict` vs
`predict_proba` calls. What we found surprisingly is that for intent classification, `predict_proba` was faster
than `predict`:

```bash
With predict_proba:
50th percentile: 5.55 ms
50th percentile: 6.76 ms
50th percentile: 7.50 ms

With predict:
50th percentile: 8.49 ms
50th percentile: 10.76 ms
50th percentile: 12.84 ms
```

## Alternate design 1:

We could just use the current `allowed_intents` field for blacklisting the follow-up intents, but then
for every dialogue turn, we need to pass in the `allowed_intents` to maintain the blacklist instead of
only specifying the blacklist when followup intents are present.

## Alternate design 2:

Instead of maintaining `allowed_intents` and `disallowed_intents`, we explicitly pass in `allowed_nlp_classes`
which encodes `allowed_intents` and `disallowed_intents` in a map. However, we now have to pass in `allowed_nlp_classes`
at every turn.

## Alternate design 3:

We could "tag" certain intents explicitly as followup intents within the NLP folder as metadata and only consider them
when the dialogflow is activated. However, this tightly couples the DM and NLP components.


## Alternate design 4:

We do not infer what is a followup intent through the handlers but instead give the developer explicit power to
blacklist.
