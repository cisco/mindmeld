# -*- coding: utf-8 -*-
"""This module contains the Kwik-E-Mart workbench demo application"""

from mmworkbench import Application


app = Application(__name__)


@app.handle(intent='greet')
def welcome(context, slots, responder):
    try:
        slots['name'] = context['request']['session']['name']
        prefix = 'Hello, {name}. '
    except KeyError:
        prefix = 'Hello. '
    responder.prompt(prefix + 'I can help you find store hours '
                              'for your local Kwik-E-Mart. How can I help?')


@app.handle(intent='exit')
def say_goodbye(context, slots, responder):
    responder.reply(['Bye', 'Goodbye', 'Have a nice day.'])


@app.handle(intent='get_store_hours')
def send_store_hours(context, slots, responder):
    stores = [e for e in context['entities'] if e['type'] == 'location']
    if stores:
        slots['location'] = stores[0]['value']
        responder.reply('The {location} Kwik-E-Mart opens at {{open_time}} and '
                        'closes at {{close_time}} {{date}}.')
        return
    responder.prompt('Which store would you like to know about?')


@app.handle(intent='get_nearest_store')
def send_nearest_store(context, slots, responder):
    # loc = context['session']['location']
    # stores = qa.get(index='stores', sort='location', current_location=loc)
    # slots['store_name'] = stores[0]['name']
    responder.reply('Your nearest Kwik-E-Mart is located at {{location}}.')


if __name__ == '__main__':
    app.cli()
