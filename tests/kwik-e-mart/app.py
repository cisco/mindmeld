# -*- coding: utf-8 -*-
"""This module contains the Kwik-E-Mart workbench demo application"""

from mmworkbench import Application


app = Application(__name__)

@app.handle(intent='greet')
def welcome(context):
    #print('CONTEXT')
    #print(context)
    return {'replies': 'Hello!'}

@app.handle(intent='exit')
def goodbye(context):
    return {'replies': 'Goodbye!'}
