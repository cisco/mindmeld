:orphan:

System Entities
===============

System entities are entities that Workbench automatically detects in a
query. These entities are common across all Workbench applications, so
there is no need to learn them through machine learning models. The
following entities are currently automatically detected by Workbench:

+-----------------+------------------------------------------------------------+
| System Entity   | Examples                                                   |
+=================+============================================================+
| sys_time        | “today” , “Tuesday, Feb 18” , “last week” , “Mother’s      |
|                 | day”                                                       |
+-----------------+------------------------------------------------------------+
| sys_interval    | “tomorrow morning” , “from 9:30 - 11:00 on tuesday” ,      |
|                 | “Friday 13th evening”                                      |
+-----------------+------------------------------------------------------------+
| sys_temperature | “64°F” , “71° Fahrenheit” , “twenty seven celsius”         |
+-----------------+------------------------------------------------------------+
| sys_number      | “fifteen” , “0.62” , “500k” , “66”                         |
+-----------------+------------------------------------------------------------+
| sys_ordinal     | “3rd” , “fourth” , “first”                                 |
+-----------------+------------------------------------------------------------+
| sys_distance    | “10 miles” , “2feet” , “0.2 inches” , “3’’ “5km” ,”12cm”   |
+-----------------+------------------------------------------------------------+
| sys_volume      | “500 ml” , “5liters” , “2 gallons”                         |
+-----------------+------------------------------------------------------------+
| sys_currency    | “forty dollars” , “9 bucks” , “$30”                        |
+-----------------+------------------------------------------------------------+
| sys_email       | “help@cisco.com”                                           |
+-----------------+------------------------------------------------------------+
| sys_url         | “washpo.com/info” , “foo.com/path/path?ext=%23&foo=bla” ,  |
|                 | “localhost”                                                |
+-----------------+------------------------------------------------------------+
| sys_phone-number| “+91 736 124 1231” , “+33 4 76095663” , “(626)-756-4757    |
|                 | ext 900”                                                   |
+-----------------+------------------------------------------------------------+

.. note::

   ``sys_interval`` differs from ``sys_time`` since
   ``sys_interval`` represents entities that have a time interval while
   ``sys_time`` represents entities with a unit time value. For example,
   “tomorrow morning” is a ``sys_interval`` entity since “morning”
   encompasses any time between 4 am and 12 pm, whereas “tomorrow” is a
   ``sys_time`` entity since it represents the unit date (eg. 2017-07-08).

Annotating System Entities
--------------------------

Users have to annotate system entities to let Workbench know which system
entities are needed for the app. By annotating system entities,
the user would train the entity recognition model on a much smaller
training set since these system entities have already been pre-learned.
Here are some examples of annotated system entities from the
home-assistant blueprint application:

.. code-block:: text

    - adjust the temperature to {65|sys_temperature}
    - {in the morning|sys_interval} set the temperature to {72|sys_temperature}
    - change my {6:45|sys_time|old_time} alarm to {7 am|sys_time|new_time}
    - move my {6 am|sys_time|old_time} alarm to {3pm in the afternoon|sys_time|new_time}
    - what's the forecast for {tomorrow afternoon|sys_interval}

Common issues with system entities
----------------------------------

-  **Misclassifying system entities**:

.. code-block:: text

    change my {6:45|sys_interval|old_time} alarm to {7 am|sys_time|new_time}

In the above query, “6:45” is misclassified as a ``sys_interval``
entity since it is a ``sys_time`` entity. During training, the following
warning will be printed to stdout:

.. code-block:: text

    Unable to load query: Unable to resolve system entity of type 'sys_interval' for '6:45'. Entities found for the following types ['sys_time']

.. tip::
   Find the affected query and change the entity label from a ``sys_interval`` to a ``sys_time`` label.

-  **Unsupported system entities**:

.. code-block:: text

    set my alarm {daily|sys_time}

Since the token “daily” is not supported by Workbench as a system entity
at the moment, the following warning will be printed:

.. code-block:: text

    Unable to load query: Unable to resolve system entity of type 'sys_time' for 'daily'.

.. tip::
   Label the token "daily" as a custom entity label, like "time", so the annotation would be as follows: "set my alarm {daily|time}".
   Otherwise, the user can remove the entity label from tokens like "daily" or remove all queries that contain the unsupported tokens like "daily" entirely.

Debugging System Entities
-------------------------

To check which token spans in a query are detected as system entities,
the following function can be invoked:

.. code-block:: python

    >>> from mmWorkbench.ser import parse_numerics
    >>> parse_numerics("tomorrow morning at 9am")
    {'data': [{'dimension': 'number',
       'entity': {'end': 21, 'start': 20, 'text': '9'},
       'likelihood': -0.11895194286136536,
       'operator': 'equals',
       'rule_count': 1,
       'value': [9]},
        .
        .
      {'dimension': 'time',
       'entity': {'end': 23, 'start': 0, 'text': 'tomorrow morning at 9am'},
       'grain': 'hour',
       'likelihood': -23.558523074887038,
       'operator': 'equals',
       'rule_count': 8,
       'value': ['2017-07-08T09:00:00.000-07:00']}],
     'status': '200'}

The dictionary returned by the ``parse_numerics`` function contains all
the token spans detected as system entities. The relevant keys are
``dimension``, which is the unit dimension of the token span and
``value``, the quantity of the dimension.
