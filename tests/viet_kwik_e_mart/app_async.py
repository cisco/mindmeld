# -*- coding: utf-8 -*-
"""This module contains the Kwik-E-Mart MindMeld demo application implemented in Vietnamese"""
from mindmeld import Application


app = Application(__name__, async_mode=True, language="vi")


@app.handle(intent="greet")
async def welcome(request, responder):
    try:
        responder.slots["name"] = request.context["name"]
        prefix = "Chào {name}. "
    except KeyError:
        prefix = "Xin chào. "
    responder.reply(prefix)

    statement = "Tôi có thể giúp bạn tìm giờ cửa hàng "\
    "cho Kwik-E-Mart tại địa phương của bạn. "
    responder.reply(statement)

    question = "Tôi có thể giúp gì cho bạn?"
    responder.reply(question)

    responder.listen()


@app.handle(intent="exit")
async def say_goodbye(request, responder):
    responder.reply(["Tạm biệt!", "Chúc bạn một ngày vui vẻ."])


@app.handle(intent="help")
async def provide_help(request, responder):
    prompts = [
        "Tôi có thể giúp bạn tìm giờ cửa hàng cho Kwik-E-Mart tại địa phương của bạn. Ví dụ, bạn có"
        " thể nói 'Cửa hàng gần nhất ở đâu vậy?' hoặc là 'Khi nào cái tiệm ở Elm Street mở vậy?'"
    ]
    responder.reply(prompts)
    responder.listen()


@app.handle(intent="find_nearest_store")
async def send_nearest_store(request, responder):
    try:
        user_location = request.context["location"]
    except KeyError:
        responder.reply("Tôi không chắc. Bạn chưa nói bạn ở đâu!")

        # only translated value of 'text'
        responder.suggest([{"type": "location", "text": "Đưa vị trí của bạn"}])
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
    responder.reply("Cái Kwik-E-Mart gần nhất bạn là ở {store_name}.")


@app.handle()
async def default(request, responder):
    prompts = [
        "Xin lỗi tôi không biết ý bạn là gì. Tôi có thể giúp "
        "bạn tìm giờ cửa hàng cho Kwik-E-Mart tại địa phương của bạn."
    ]
    responder.reply(prompts)
    responder.listen()


@app.dialogue_flow(domain="store_info", intent="get_store_hours")
async def send_store_hours(request, responder):
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
            "Cái Kwik-E-Mart ở {store_name} mở cửa {open_time}h và "
            "đóng cửa {close_time}h."
        )
        return

    responder.frame["count"] = responder.frame.get("count", 0) + 1

    if responder.frame["count"] <= 3:
        responder.reply("Bạn muốn biết về cửa hàng nào?")
        responder.listen()
    else:
        responder.reply("Xin lỗi tôi không thể giúp bạn được. Xin vui lòng thử lại.")
        responder.exit_flow()


@send_store_hours.handle(default=True)
async def default_handler(request, responder):
    responder.frame["count"] += 1
    if responder.frame["count"] <= 3:
        responder.reply("Xin lỗi tôi không có hiểu bạn. Bạn muốn biết về cửa hàng nào?")
        responder.listen()
    else:
        responder.reply("Xin lỗi tôi không thể giúp bạn được. Xin vui lòng thử lại.")
        responder.exit_flow()


@send_store_hours.handle(intent="exit", exit_flow=True)
async def exit_handler(request, responder):
    responder.reply(["Tạm biệt!", "Chúc bạn một ngày vui vẻ."])


@send_store_hours.handle(intent="get_store_hours")
async def send_store_hours_in_flow_handler(request, responder):
    return await send_store_hours(request, responder)
