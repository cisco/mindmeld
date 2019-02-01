import copy


class Request:
    def __init__(self, text, context, history, frame, params, previous_turn_params,
                 domain, intent, entities, dialogue_state=None, directives=(), confidence=None,
                 nbest_transcripts_text=None, nbest_transcripts_entities=None,
                 nbest_aligned_entities=None, name=''):
        self.domain = domain
        self.intent = intent
        self.entities = entities
        self.history = history
        self.text = text
        self.frame = copy.deepcopy(frame)
        self.params = params
        self.previous_turn_params = previous_turn_params
        self.context = context
        self.dialogue_state = dialogue_state
        self.directives = directives
        self.confidence = confidence
        self.nbest_transcripts_text = nbest_transcripts_text
        self.nbest_transcripts_entities = nbest_transcripts_entities
        self.nbest_aligned_entities = nbest_aligned_entities
        self.name = name

    def to_json(self):
        attrs_to_serialize = ['params', 'domain', 'intent', 'entities', 'text', 'context',
                              'directives', 'dialogue_state', 'history', 'frame',
                              'confidence', 'nbest_transcripts_text', 'previous_turn_params',
                              'nbest_transcripts_entities', 'nbest_aligned_entities', 'name']
        serialized_obj = {}
        for attr, value in vars(self).items():
            if attr not in attrs_to_serialize or value is None:
                continue
            serialized_obj[attr] = value
        return serialized_obj
