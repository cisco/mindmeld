"""This module contains the MindMeld application"""

from mindmeld import Application

app = Application(__name__)

__all__ = ['app']

@app.handle(default=True)
@app.handle(intent='unsupported')
def default(request, responder):
	replies = ["I didn't get that. Can you say it again?", 'I missed what you said. What was that?', 'Sorry, could you say that again?', 'Sorry, can you say that again?', 'Can you say that again?', "Sorry, I didn't get that. Can you rephrase?", 'Sorry, what was that?', 'One more time?', 'What was that?', 'Say that one more time?', "I didn't get that. Can you repeat?", 'I missed that, say that again?']
	responder.reply(replies)

