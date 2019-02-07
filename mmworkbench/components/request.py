import attr
import immutables


@attr.s(frozen=False, kw_only=True)
class Params:
    previous_params = attr.ib(default=None)
    allowed_intents = attr.ib(default=tuple(), converter=tuple)
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    dynamic_resource = attr.ib(default=immutables.Map(),
                               converter=immutables.Map)

    def to_json(self):
        # We have to do this since attr.asdict cannot serialize immutables Map objects
        serialized_obj = {}
        for k, v in vars(self).items():
            if type(v) == immutables._map.Map:
                serialized_obj[k] = dict(v)
            elif type(v) == Params:
                serialized_obj[k] = v.to_json()
            else:
                serialized_obj[k] = v
        return serialized_obj


@attr.s(frozen=True, kw_only=True)
class Request:
    domain = attr.ib()
    intent = attr.ib()
    entities = attr.ib(default=tuple(), converter=tuple)
    history = attr.ib(default=tuple(), converter=tuple)
    text = attr.ib()
    frame = attr.ib(default=immutables.Map(),
                    converter=immutables.Map)
    params = attr.ib(default=Params())
    context = attr.ib(default=immutables.Map(),
                      converter=immutables.Map)
    confidence = attr.ib(default=immutables.Map(),
                         converter=immutables.Map)
    nbest_transcripts_text = attr.ib(default=tuple(), converter=tuple)
    nbest_transcripts_entities = attr.ib(default=tuple(), converter=tuple)
    nbest_aligned_entities = attr.ib(default=tuple(), converter=tuple)

    def to_json(self):
        # Have to do this since attr.asdict cannot serialize immutables Map objects
        serialized_obj = {}
        for k, v in vars(self).items():
            if type(v) == immutables._map.Map:
                serialized_obj[k] = dict(v)
            elif type(v) == Params:
                serialized_obj[k] = v.to_json()
            else:
                serialized_obj[k] = v
        return serialized_obj
