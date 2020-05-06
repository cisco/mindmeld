# -*- coding: utf-8 -*-
"""This module contains the Kwik-E-Mart MindMeld demo application"""
from mindmeld import Application
from mindmeld.core import FormEntity

from . import custom_features  # noqa: F401

app = Application(__name__)

__all__ = ["app"]


@app.handle(intent="greet")
def welcome(request, responder):
    try:
        responder.slots["name"] = request.context["name"]
        prefix = "Hello, {name}. "
    except KeyError:
        prefix = "Hello. "

    responder.reply(
        prefix + "I can help you find store hours "
        "for your local Kwik-E-Mart. How can I help?"
    )
    responder.listen()


@app.handle(intent="exit")
def say_goodbye(request, responder):
    responder.reply(["Bye", "Goodbye", "Have a nice day."])


@app.handle(intent="help")
def provide_help(request, responder):
    prompts = [
        "I can help you find store hours phone numbers for your local Kwik-E-Mart."
        "For example, you can say 'When does the Elm Street store open?' or "
        "'Where's the nearest store?' or  or 'what's the phone number for Central Plaza?"
    ]
    responder.reply(prompts)
    responder.listen()


@app.handle(intent="find_nearest_store")
def send_nearest_store(request, responder):
    try:
        user_location = request.context["location"]
    except KeyError:
        responder.reply("I'm not sure. You haven't told me where you are!")
        responder.suggest([{"type": "location", "text": "Share your location"}])
        return

    stores = app.question_answerer.get(
        index="stores",
        _sort="location",
        _sort_type="distance",
        _sort_location=user_location,
    )
    target_store = stores[0]
    responder.slots["store_name"] = target_store["store_name"]

    responder.frame["target_store"] = target_store
    responder.reply("Your nearest Kwik-E-Mart is located at {store_name}.")


@app.handle()
def default(request, responder):
    prompts = [
        "Sorry, not sure what you meant there. I can help you find store hours "
        "for your local Kwik-E-Mart."
    ]
    responder.reply(prompts)
    responder.listen()


@app.dialogue_flow(domain="store_info", intent="get_store_hours")
def send_store_hours(request, responder):
    active_store = None
    store_entity = next(
        (e for e in request.entities if e["type"] == "store_name"), None
    )
    if store_entity:
        try:
            stores = app.question_answerer.get(
                index="stores", id=store_entity["value"]["id"]
            )
        except TypeError:
            # failed to resolve entity
            stores = app.question_answerer.get(
                index="stores", store_name=store_entity["text"]
            )
        try:
            active_store = stores[0]
            responder.frame["target_store"] = active_store
        except IndexError:
            # No active store... continue
            pass
    elif "target_store" in request.frame:
        active_store = request.frame["target_store"]

    if active_store:
        responder.slots["store_name"] = active_store["store_name"]
        responder.slots["open_time"] = active_store["open_time"]
        responder.slots["close_time"] = active_store["close_time"]
        responder.reply(
            "The {store_name} Kwik-E-Mart opens at {open_time} and "
            "closes at {close_time}."
        )
        return

    responder.frame["count"] = responder.frame.get("count", 0) + 1

    if responder.frame["count"] <= 3:
        responder.reply("Which store would you like to know about?")
        responder.listen()
    else:
        responder.reply("Sorry I cannot help you. Please try again.")
        responder.exit_flow()


@send_store_hours.handle(default=True)
def default_handler(request, responder):
    responder.frame["count"] = responder.frame.get("count", 0) + 1
    if responder.frame["count"] <= 3:
        responder.reply(
            "Sorry, I did not get you. Which store would you like to know about?"
        )
        responder.listen()
    else:
        responder.reply("Sorry I cannot help you. Please try again.")
        responder.exit_flow()


@send_store_hours.handle(intent="exit", exit_flow=True)
def exit_handler(request, responder):
    responder.reply(["Bye", "Goodbye", "Have a nice day."])


@send_store_hours.handle(intent="get_store_hours")
def send_store_hours_in_flow_handler(request, responder):
    send_store_hours(request, responder)


form_store_phone = {
    'entities': [
        FormEntity(
            entity='store_name',
            responses='Which store would you like to know about?',
            retry_response="Sorry, I did not get you. "
                           "Which store would you like to know about?")],
    'max_retries': 1,
    'exit_msg': "Sorry I cannot help you. Please try again."}


@app.auto_fill(domain="store_info", intent="get_store_number", form=form_store_phone)
def send_store_phone(request, responder):
    active_store = None
    store_entity = next(
        (e for e in request.entities if e["type"] == "store_name"), None
    )
    try:
        stores = app.question_answerer.get(
            index="stores", id=store_entity["value"][0]["id"]
        )
    except TypeError:
        # failed to resolve entity
        stores = app.question_answerer.get(
            index="stores", store_name=store_entity["text"]
        )
    try:
        active_store = stores[0]
    except IndexError:
        # No active store... continue
        pass

    if active_store:
        responder.slots["store_name"] = active_store["store_name"]
        responder.slots["phone_number"] = active_store["phone_number"]
        responder.reply(
            "The {store_name} Kwik-E-Mart can be reached at {phone_number}."
        )
        return
