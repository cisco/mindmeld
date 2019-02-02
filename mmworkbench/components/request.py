import attr


@attr.s(frozen=True, kw_only=True)
class Request:
    domain = attr.ib()
    intent = attr.ib()
    entities = attr.ib()
    history = attr.ib()
    text = attr.ib()
    frame = attr.ib()
    params = attr.ib()
    context = attr.ib()
    confidence = attr.ib(default=None)
    nbest_transcripts_text = attr.ib(default=None)
    nbest_transcripts_entities = attr.ib(default=None)
    nbest_aligned_entities = attr.ib(default=None)

    @property
    def dialogue_flow(self):
        return self.context.get('dialogue_flow')


@attr.s(frozen=False, kw_only=True)
class Params:
    previous_params = attr.ib(default={})
    allowed_intents = attr.ib(default=[])
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    dynamic_resource = attr.ib(default={})
