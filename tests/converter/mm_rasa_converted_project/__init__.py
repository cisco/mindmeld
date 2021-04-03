from mindmeld import Application
from mindmeld.components.custom_action import CustomAction
from . import custom_features  # noqa: F401


app = Application(__name__)
__all__ = ['app']
action_config = {'url': 'http://localhost:5055/webhook'}


@app.handle(intent='greet')
@app.handle(intent='fullname', has_entities=['name', 'lastname'])
def utter_name(request, responder):
    additional_actions = ['utter_greet']
    prompts = ["Hey there! Tell me your name.", "Howdy. What's your name."]
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='thanks')
def utter_thanks(request, responder):
    prompts = ["My pleasure."]
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='name', has_entity='name')
def utter_greet(request, responder):
    name_s = [e['text'] for e in request.entities if e['type'] == 'name']
    name = name_s[0]
    prompts = ["Nice to you meet you {0}. How can I help?".format(name)]
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='goodbye')
def utter_goodbye(request, responder):
    prompts = ["Talk to you later!"]
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='joke')
def action_joke(request, responder):
    # This is a custom action from rasa
    action = CustomAction(name='action_joke', config=action_config)
    action.invoke(request, responder)
