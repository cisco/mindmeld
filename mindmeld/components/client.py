import copy
import json
import logging
import requests

from .dialogue import DirectiveNames

mod_logger = logging.getLogger(__name__)


class ConversationClient:
    """The conversation object is a very basic MindMeld http client.

    It can be useful for testing out dialogue flows in python.

    Example:
        >>> convo = ConversationClient()
        >>> convo.say('Hello')
        ['Hello. I can help you find store hours. How can I help?']
        >>> convo.say('Is the store on elm open?')
        ['The 23 Elm Street Kwik-E-Mart is open from 7:00 to 19:00.']

    Attributes:
        history (list): The sequence of previous turns in the conversation,
            starting with the most recent message.
        context (dict): The user context of the conversation.
        default_params (dict): The default params to use with each turn. These \
            defaults will be overridden by params passed for each turn.
        params (dict): The params returned by the most recent turn.
    """

    _logger = mod_logger.getChild("ConversationClient")

    def __init__(
        self,
        url="http://localhost:7150/parse",
        default_params=None,
    ):
        """
        Args:
            url (str): The url to be used to talk to the query endpoint.
            context (dict, optional): The context to be used in the conversation.
            default_params (dict, optional): The default params to use with each turn. These
                defaults will be overridden by params passed for each turn.
        """
        self.url = url
        self.form = {}
        self.history = []
        self.frame = {}
        if default_params is None:
            default_params = {}
        self.default_params = default_params
        self.params = {}

    def say(self, text, params=None, frame=None, form=None):
        """Send a message in the conversation. The message will be
        processed by the app based on the current state of the conversation and
        returns the extracted messages from the directives.

        Args:
            text (str): The text of a message.
            params (dict): The params to use with this message,
                overriding any defaults which may have been set.
            frame (dict): The frame to be used with this message, overriding any defaults.
            form (dict): A form data dictionary

        Returns:
            (list): A text representation of the dialogue responses.
        """
        response = self.process(text, params=params, frame=frame, form=form)

        # handle directives
        response_texts = [self._follow_directive(a) for a in response["directives"]]
        return response_texts

    def process(self, text, params=None, frame=None, form=None):
        """Send a message in the conversation. The message will be processed by
        the app based on the current state of the conversation and returns
        the response.

        Args:
            text (str): The text of a message.
            params (dict): The params to use with this message, overriding any defaults
                which may have been set.
            frame (dict): The frame to be used with this message, overriding any defaults.
            context (dict): The context to be used with this message, overriding any defaults.
            form (dict): A form data dictionary

        Returns:
            (dict): The dictionary response.
        """
        internal_params = copy.deepcopy(self.params)

        if params:
            for k, v in params.items():
                internal_params[k] = v

        internal_frame = copy.deepcopy(self.frame)

        if frame:
            for k, v in frame.items():
                internal_frame[k] = v

        internal_form = copy.deepcopy(self.form)

        if form:
            for k, v in form.items():
                internal_form[k] = v

        response = requests.post(
            url=self.url,
            json={
                "text": text,
                "history": self.history,
                "form": internal_form,
                "frame": internal_frame,
                "params": internal_params,
            },
        ).json()

        self.history = response["history"]
        self.frame = response["frame"]
        self.params = response["params"]
        return response

    def _follow_directive(self, directive):
        msg = ""
        try:
            directive_name = directive["name"]
            if directive_name in [DirectiveNames.REPLY, DirectiveNames.SPEAK]:
                msg = directive["payload"]["text"]
            elif directive_name == DirectiveNames.SUGGESTIONS:
                suggestions = directive["payload"]
                if not suggestions:
                    raise ValueError
                msg = "Suggestion{}:".format("" if len(suggestions) == 1 else "s")
                texts = []
                for idx, suggestion in enumerate(suggestions):
                    if idx > 0:
                        msg += ", {!r}"
                    else:
                        msg += " {!r}"

                    texts.append(self._generate_suggestion_text(suggestion))
                msg = msg.format(*texts)
            elif directive_name == DirectiveNames.LIST:
                msg = "\n".join(
                    [
                        json.dumps(item, indent=4, sort_keys=True)
                        for item in directive["payload"]
                    ]
                )
            elif directive_name == DirectiveNames.LISTEN:
                msg = "Listening..."
            elif directive_name == DirectiveNames.RESET:
                msg = "Resetting..."
                self.reset()
        except (KeyError, ValueError, AttributeError):
            msg = "Unsupported response: {!r}".format(directive)
            self._logger.warning(msg)

        return msg

    @staticmethod
    def _generate_suggestion_text(suggestion):
        pieces = []
        if "text" in suggestion:
            pieces.append(suggestion["text"])
        if suggestion["type"] != "text":
            pieces.append("({})".format(suggestion["type"]))

        return " ".join(pieces)

    def reset(self):
        """Reset the history, frame and params of the Conversation object."""
        self.history = []
        self.frame = {}
        self.params = {}
        self.form = {}
