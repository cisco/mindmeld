# Disallow intents feature

This feature blacklists certain MindMeld intents such that these intents will never be picked during 
inference.


## Problem

Currently, the way to disallow certain intents is to allow all the intents in the NLP hierarchy 
through the `allowed_intents` field and remove the ones that need to be disallowed.

As a guiding example, consider the following hierarchy:

.
├── banking
│   └── transfer_money
│       └── train.txt
└── store_info
    ├── exit
    ├── find_nearest_store
    ├── get_store_hours
    ├── get_store_number
    ├── greet
    └── help

If one wants to disallow the "greet" intent, one has to do the following:
`nlp.process("Good product", allowed_intents=['banking', 'store_info.exit', 
'store_info.find_nearest_store', 'store_info.get_store_hours',
'store_info.get_store_number', 'store_info.help'],)`

This is verbose and suboptimal for a developer experience.


## Proposed solution







## Alternate solution
