"""This module contains the MindMeld application"""

from mindmeld import Application

app = Application(__name__)

__all__ = ['app']

@app.handle(intent='default_welcome_intent_usersays_en')
def renameMe0(request, responder):
	replies = ['Hi! How are you doing?', 'Hello! How can I help you?', 'Good day! What can I do for you today?', 'Greetings! How can I assist?']
	responder.reply(replies)

@app.handle(intent='languages_usersays_en')
def renameMe1(request, responder):
	replies = ["Wow! I didn't know you knew $language. How long have you known $language?"]
	responder.reply(replies)

@app.handle(intent='name_usersays_en')
def renameMe2(request, responder):
	replies = ['My name is Dialogflow!']
	responder.reply(replies)

@app.handle(intent='languages_custom_usersays_en')
def renameMe3(request, responder):
	replies = ["I can't believe you've known #languages-followup.language for $duration!"]
	responder.reply(replies)

