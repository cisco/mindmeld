Step 10: Deploy Trained Models
==============================

Once your application has been built, MindMeld makes it easy to test locally. In :doc:`Step 4 <04_define_the_dialogue_handlers>`, we created an application container for your dialogue state handler logic. This was in the ``__init__.py`` file in the application root directory. To provide the necessary interface to manage deployment, we now create the ``__main__.py`` file with the following three lines of code.

.. code:: python

  if __name__ == "__main__":
      from . import app
      app.cli()

We can now test our application locally.


If you have not already, you must build your models before the application can be run. To build your models use the ``build`` command:

.. code-block:: console

    cd $HOME
    python -m my_app build

.. code-block:: console

    Building application my_app...complete.

To launch the web service use the ``run`` command:

.. code-block:: console

    cd $HOME
    python -m my_app run

.. code-block:: console

    Numerical parser running, PID 20248
    Loading intent classifier: domain='store_info'
    ...
     * Running on http://0.0.0.0:7150/ (Press CTRL+C to quit)

To test using any REST client (such as Postman or Advanced Rest Client), send `POST` requests to the web service endpoint at ``http://localhost:7150/parse``. Alternately, you can use a :manpage:`curl` command from your terminal as follows:

.. code-block:: console

  curl -X POST -H 'Content-Type: application/json' -d '{"text": "order from firetrail"}' "http://localhost:7150/parse" | jq .

.. note:: The MindMeld flask server is stateless, so in order to perform multi-turn dialogues with the server, copy the response returned from the server in the first turn and use that in the data parameter of the curl request in the next turn, along with the new ``text`` key, value.

.. code-block:: console

   {
     "dialogue_state": "build_order",
     "directives": [
       {
         "name": "reply",
         "payload": {
           "text": "Great, what would you like to order from Firetrail Pizza?"
         },
         "type": "view"
       },
       {
         "name": "listen",
         "type": "action"
       }
     ],
     "frame": {
       "dishes": [],
       "restaurant": {
         "categories": [
           "Beverages",
           "Pizzas",
           "Sides",
           "Popular Dishes"
         ],
         "cuisine_types": [
           "Pizza"
         ],
         "id": "B01CT54GYE",
         "image_url": "https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/FiretrailPizza/logo_232x174._CB295435423_SX600_QL70_.png",
         "menus": [
           {
             "id": "127c097e-2d9d-4880-99ac-f1688909af07",
             "option_groups": [
               {
                 "id": "ToppingsGF",
                 "max_selected": 9,
                 "min_selected": 0,
                 "name": "Add Some Extra Toppings",
                 "options": [
                   {
                     "description": null,
                     "id": "B01D8TDFV0",
                     "name": "Goat Cheese",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TCH3M",
                     "name": "Olives",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TD8VC",
                     "name": "Garlic",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TD4YI",
                     "name": "Sausage",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TD5J2",
                     "name": "Onions",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TDHAY",
                     "name": "Bruno Nippy Peppers",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TDALK",
                     "name": "Pepperoni",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TCT7G",
                     "name": "Roasted Red Peppers",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TCWXC",
                     "name": "Mushrooms",
                     "price": 1
                   }
                 ]
               },
               {
                 "id": "Toppings",
                 "max_selected": 12,
                 "min_selected": 0,
                 "name": "Add Some Extra Toppings",
                 "options": [
                   {
                     "description": null,
                     "id": "B01D8TCVYC",
                     "name": "Roasted Red Peppers",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TDF9M",
                     "name": "Garlic",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TCWM8",
                     "name": "Olives",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TC930",
                     "name": "Basil",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TDBFK",
                     "name": "Goat Cheese",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TCKHU",
                     "name": "Sausage",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TCSNG",
                     "name": "Onions",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TCO2G",
                     "name": "Bruno Nippy Peppers",
                     "price": 1
                   },
                   {
                     "description": null,
                     "id": "B01D8TC8AE",
                     "name": "Pepperoni",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TCJRQ",
                     "name": "Mushrooms",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TC8PE",
                     "name": "Shredded Parmesan",
                     "price": 2
                   },
                   {
                     "description": null,
                     "id": "B01D8TD560",
                     "name": "Shredded Mozzarella",
                     "price": 2
                   }
                 ]
               }
             ],
             "size_groups": [
               {
                 "description": null,
                 "id": "Pizzasize",
                 "name": "Choose Your Pizza Size",
                 "sizes": [
                   {
                     "alt_name": "10\" Pizza",
                     "name": "10\" Pizza"
                   },
                   {
                     "alt_name": "14\" Pizza",
                     "name": "14\" Pizza"
                   }
                 ]
               }
             ]
           }
         ],
         "name": "Firetrail Pizza",
         "num_reviews": 13,
         "price_range": 2,
         "rating": 4.1
       }
     },
     "history": [
       {
         "dialogue_state": "build_order",
         "directives": [
           {
             "name": "reply",
             "payload": {
               "text": "Great, what would you like to order from Firetrail Pizza?"
             },
             "type": "view"
           },
           {
             "name": "listen",
             "type": "action"
           }
         ],
         "frame": {
           "dishes": [],
           "restaurant": {
             "categories": [
               "Beverages",
               "Pizzas",
               "Sides",
               "Popular Dishes"
             ],
             "cuisine_types": [
               "Pizza"
             ],
             "id": "B01CT54GYE",
             "image_url": "https://images-na.ssl-images-amazon.com/images/G/01/ember/restaurants/SanFrancisco/FiretrailPizza/logo_232x174._CB295435423_SX600_QL70_.png",
             "menus": [
               {
                 "id": "127c097e-2d9d-4880-99ac-f1688909af07",
                 "option_groups": [
                   {
                     "id": "ToppingsGF",
                     "max_selected": 9,
                     "min_selected": 0,
                     "name": "Add Some Extra Toppings",
                     "options": [
                       {
                         "description": null,
                         "id": "B01D8TDFV0",
                         "name": "Goat Cheese",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TCH3M",
                         "name": "Olives",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TD8VC",
                         "name": "Garlic",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TD4YI",
                         "name": "Sausage",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TD5J2",
                         "name": "Onions",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TDHAY",
                         "name": "Bruno Nippy Peppers",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TDALK",
                         "name": "Pepperoni",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TCT7G",
                         "name": "Roasted Red Peppers",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TCWXC",
                         "name": "Mushrooms",
                         "price": 1
                       }
                     ]
                   },
                   {
                     "id": "Toppings",
                     "max_selected": 12,
                     "min_selected": 0,
                     "name": "Add Some Extra Toppings",
                     "options": [
                       {
                         "description": null,
                         "id": "B01D8TCVYC",
                         "name": "Roasted Red Peppers",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TDF9M",
                         "name": "Garlic",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TCWM8",
                         "name": "Olives",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TC930",
                         "name": "Basil",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TDBFK",
                         "name": "Goat Cheese",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TCKHU",
                         "name": "Sausage",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TCSNG",
                         "name": "Onions",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TCO2G",
                         "name": "Bruno Nippy Peppers",
                         "price": 1
                       },
                       {
                         "description": null,
                         "id": "B01D8TC8AE",
                         "name": "Pepperoni",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TCJRQ",
                         "name": "Mushrooms",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TC8PE",
                         "name": "Shredded Parmesan",
                         "price": 2
                       },
                       {
                         "description": null,
                         "id": "B01D8TD560",
                         "name": "Shredded Mozzarella",
                         "price": 2
                       }
                     ]
                   }
                 ],
                 "size_groups": [
                   {
                     "description": null,
                     "id": "Pizzasize",
                     "name": "Choose Your Pizza Size",
                     "sizes": [
                       {
                         "alt_name": "10\" Pizza",
                         "name": "10\" Pizza"
                       },
                       {
                         "alt_name": "14\" Pizza",
                         "name": "14\" Pizza"
                       }
                     ]
                   }
                 ]
               }
             ],
             "name": "Firetrail Pizza",
             "num_reviews": 13,
             "price_range": 2,
             "rating": 4.1
           }
         },
         "params": {
           "allowed_intents": [],
           "dynamic_resource": {},
           "target_dialogue_state": null,
           "time_zone": null,
           "timestamp": 0
         },
         "request": {
           "confidences": {},
           "context": {},
           "domain": "ordering",
           "entities": [
             {
               "role": null,
               "span": {
                 "end": 19,
                 "start": 11
               },
               "text": "firetrail",
               "type": "restaurant",
               "value": [
                 {
                   "cname": "Firetrail Pizza",
                   "id": "B01CT54GYE",
                   "score": 27.906038,
                   "top_synonym": "Firetrail"
                 }
               ]
             }
           ],
           "frame": {},
           "history": [],
           "intent": "build_order",
           "nbest_aligned_entities": [],
           "nbest_transcripts_entities": [],
           "nbest_transcripts_text": [],
           "params": {
             "allowed_intents": [],
             "dynamic_resource": {},
             "target_dialogue_state": null,
             "time_zone": null,
             "timestamp": 0
           },
           "text": "order from firetrail"
         },
         "slots": {
           "restaurant_name": "Firetrail Pizza"
         }
       }
     ],
     "params": {
       "allowed_intents": [],
       "dynamic_resource": {},
       "target_dialogue_state": null,
       "time_zone": null,
       "timestamp": 0
     },
     "request": {
       "confidences": {},
       "context": {},
       "domain": "ordering",
       "entities": [
         {
           "role": null,
           "span": {
             "end": 19,
             "start": 11
           },
           "text": "firetrail",
           "type": "restaurant",
           "value": [
             {
               "cname": "Firetrail Pizza",
               "id": "B01CT54GYE",
               "score": 27.906038,
               "top_synonym": "Firetrail"
             }
           ]
         }
       ],
       "frame": {},
       "history": [],
       "intent": "build_order",
       "nbest_aligned_entities": [],
       "nbest_transcripts_entities": [],
       "nbest_transcripts_text": [],
       "params": {
         "allowed_intents": [],
         "dynamic_resource": {},
         "target_dialogue_state": null,
         "time_zone": null,
         "timestamp": 0
       },
       "text": "order from firetrail"
     },
     "request_id": "21473eb8-14c2-438a-9102-d8178104501f",
     "slots": {
       "restaurant_name": "Firetrail Pizza"
     },
     "response_time": 2.535576820373535,
     "version": "2.0"
   }

The web service responds with a JSON data structure containing the application response along with the detailed output for all of the machine learning components of the MindMeld platform.

.. See the :ref:`User Guide <userguide>` for more about the MindMeld request and response interface format.

.. Cloud Deployment
.. ~~~~~~~~~~~~~~~~~~~~~~~~~

.. Coming Soon
