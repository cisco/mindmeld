from mindmeld import Application
from . import custom_features  # noqa: F401

app = Application(__name__)

__all__ = ['app']


@app.handle(intent='greet')
def utter_name(request, responder):
    prompts = ['Hey there! Tell me your name.']
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='thanks')
def utter_thanks(request, responder):
    prompts = ['My pleasure.']
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='name', has_entity='name')
@app.handle(intent='fullname', has_entities=['name', 'lastname'])
def utter_greet(request, responder):
    name = request.context['name']
    prompts = ['Nice to you meet you {name}. How can I help?']
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='goodbye')
def utter_goodbye(request, responder):
    prompts = ['Talk to you later!']
    responder.reply(prompts)
    responder.listen()


@app.handle(intent='joke')
def action_joke(request, responder):
    # This is a custom action from rasa
    pass
