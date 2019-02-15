import attr
import immutables
import logging
from pytz import timezone
from pytz.exceptions import UnknownTimeZoneError

logger = logging.getLogger(__name__)


def _validate_time_zone(param=None):
    """Validates time zone parameters

    Args:
        param (str, optional): The time zone parameter

    Returns:
        str: The passed in time zone
    """
    if not param:
        return None
    if not isinstance(param, str):
        logger.warning("Invalid %r param: %s is not of type %s.", 'time_zone', param, str)
        return None
    try:
        timezone(param)
    except UnknownTimeZoneError:
        logger.warning("Invalid %r param: %s is not a valid time zone.", 'time_zone', param)
        return None
    return param


def _validate_generic(name, ptype):
    def validator(param):
        if not isinstance(param, ptype):
            logger.warning("Invalid %r param: %s is not of type %s.", name, param, ptype)
            param = None
        return param
    return validator


PARAM_VALIDATORS = {
    'allowed_intents': _validate_generic('allowed_intents', list),

    # TODO: use a better validator for this
    'target_dialogue_state': _validate_generic('target_dialogue_state', str),

    'time_zone': _validate_time_zone,
    'timestamp': _validate_generic('timestamp', int),
    'dynamic_resource': _validate_generic('dynamic_resource', dict)
}


@attr.s(frozen=False, kw_only=True)
class Params:
    previous_params = attr.ib(default=None)
    allowed_intents = attr.ib(default=[])
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    dynamic_resource = attr.ib(default={})

    def validate_param(self, name):
        validator = PARAM_VALIDATORS.get(name)
        param = vars(self).get(name)
        if param:
            return validator(param)
        return param

    def dm_params(self, handler_map):
        target_dialogue_state = self.validate_param('target_dialogue_state')
        if target_dialogue_state and target_dialogue_state not in handler_map:
            logger.error("Target dialogue state {} does not match any dialogue state names "
                         "in for the application. Not applying the target dialogue state "
                         "this turn.".format(target_dialogue_state))
            return {'target_dialogue_state': None}
        return {'target_dialogue_state': target_dialogue_state}

    def nlp_params(self):
        return {param: self.validate_param(param)
                for param in ('time_zone', 'timestamp', 'dynamic_resource')}


@attr.s(frozen=True, kw_only=True)
class FrozenParams(Params):
    previous_params = attr.ib(default=None)
    allowed_intents = attr.ib(default=tuple(), converter=tuple)
    target_dialogue_state = attr.ib(default=None)
    time_zone = attr.ib(default=None)
    timestamp = attr.ib(default=0)
    dynamic_resource = attr.ib(default=immutables.Map(),
                               converter=immutables.Map)


@attr.s(frozen=True, kw_only=True)
class Request:
    domain = attr.ib(default=None)
    intent = attr.ib(default=None)
    entities = attr.ib(default=tuple(), converter=tuple)
    history = attr.ib(default=tuple(), converter=tuple)
    text = attr.ib(default=None)
    frame = attr.ib(default=immutables.Map(),
                    converter=immutables.Map)
    params = attr.ib(default=FrozenParams())
    context = attr.ib(default=immutables.Map(),
                      converter=immutables.Map)
    confidences = attr.ib(default=immutables.Map(),
                          converter=immutables.Map)
    nbest_transcripts_text = attr.ib(default=tuple(), converter=tuple)
    nbest_transcripts_entities = attr.ib(default=tuple(), converter=tuple)
    nbest_aligned_entities = attr.ib(default=tuple(), converter=tuple)
