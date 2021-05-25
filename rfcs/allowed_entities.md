# Allowed entities feature

This feature allows MindMeld's entity classifier to more precisely tag entities in 
queries without context.

## Problem

Supposed for an intent `transfer_money`, there are training queries like this following:
```text
move money from {savings|account_type} to {checkings|account_type}
transfer to {savings|account_type} from {checkings|account_type}
collect money into {savings|account_type}
```

The entity classifier currently uses the following information to make it's classification:
- Contextual words
- Gazetteers
- Whether it has seen the phrase before (in-vocab)

However, in multi-turn dialogues where MindMeld's NLP service is only used (and not its
dialogue management), such a conversation is possible:
```text
User> Transfer money
Bot> From which account
User> Checkings
``` 

For the last query, we want `Checkings` to have an entity type `account_type`, but since 
the contextual signal is missing and since `Checkings` could be used as another entity
in another intent, its currently not possible to reliably classify `Checkings` as 
`account_type` entity.

## Proposed solution

To solve this issue, we propose using the `allowed_intents` key in the `/parse` API to
provide additional context for MindMeld to match to the correct entity type.

The client provides the following payload:
```python
{
    'text': 'Checkings',
    'allowed_intents': ['banking.money_transfer.account_type']
}
```
Here, `domain=banking`, `intent=money_transfer` and `entity_type=account_type`.

MindMeld will do the following logic which is different from the current behavior:
1. Since an additional entity type is provided in the `allowed_intents` field, it will 
first run the intent's entity classifier against the query.
1. If the classifier returns `account_type`, so it matches the allowed_intent's entity type,
MindMeld will return `Checkings` as `account_type` entity, along with all other entities detected.
1. If the classifier doesnt detect any entities, we attempt to exact match the input phrase 
against the allowed_intent's entity type's gazetteer entries and all training phrases annotated
with the entity type. If there is a match, we return `Checkings` as `account_type` entity. If
there is no match, we don't return any entities back to the client.
1. If the classifier detects an entity that is different from the allowed_intent's entity type,
then we return the classifier's predicted entity and not the allowed_intent's entity type.

With the above approach, the client is providing MindMeld a hint to try to match free-form user 
input to an entity type that might traditionally match without more contextual information. MindMeld
uses this hint to try harder to match the user's input against the entity_type, but if it cannot or
it has a better prediction based on contextual information/other signals, it provides it's better response.

### What about user input that doesn't match the gazetteer or training data?

If MindMeld cannot leverage condition 3 in the above matching scheme, more information needs to be provided 
to the system. For example, say the user has a custom nickname called `John's investment account` for his/her account 
type:
```text
User> Transfer money
Bot> From which account
User> John's investment account 
``` 

Here, the query is `John's investment account`, which is a phrase MindMeld has never seen before, so doing this won't help
since it won't match condition 3: 
```python
{
    'text': "John's investment account",
    'allowed_intents': ['banking.money_transfer.account_type']
}
```

However, if the client knew the list of nicknames to the end-user's account, for eg, 
`["John's investment account", "John's retirement account"]`, supplying this information through the 
dynamic resource can help the classifier detect such a query:

```python
{
    'text': "John's investment account",
    'allowed_intents': ['banking.money_transfer.account_type'],
    'dynamic_resource': {
        'gazetteers': {
            'account_type': {
                "John's investment account": 1.0,
                "John's retirement account": 1.0
            }
        }
    }
}
```

In the above payload, since we know to use `money_transfer` intent's entity classifier and apply 
condition 3 to the combined gazetteer which consists of both the static gazetteer entities and the
dynamic gazetteer entries, we will be able to infer that `John's investment account` is an `account_type` entity.

## Alternative idea
Create a special catch-all entity called `sys_any` (modelled after DialogFlow's entity with the same name) that
captures all free-form text. In this method, the training data would be annotated as follows:

```text
move money from {savings|sys.any} to {checkings|sys-any}
transfer to {savings|sys.any} from {checkings|sys-any}
collect money into {savings|sys-any}
```

Then, for a query `Savings` where `allowed_intents=[banking.money_transfer.sys_any]`, we immidiately
match the query to a `sys_any` entity type since this entity matches all freeform text.

The issue with this approach is that it could match queries like `$400` or `12:30pm`, which clearly are 
not account type entities. So it sacrifices too much precision for recall.


## Potential issues with this feature
1. Role classification without context is impossible, so the classifier still has to provide an result for
that even if it doesn't know. Maybe it can set the role to be null, but such behavior should be communicated to the client
so that it doesnt assume a role will always be returned.
2. The behavior for `allowed_intents` at the intent level is inconsistent with the entity level since
if it's set to an intent, that intent is returned, whereas for the entity level, we have the conditions listed above.
3. It is abusing the `allowed_intents` field. However, we can change this name to `allowed_nlp_components` in the future
while being backwards compatible.

