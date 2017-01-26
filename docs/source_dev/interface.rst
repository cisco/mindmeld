Workbench Interface Specification
=================================

This section documents the specification for the request and response objects, which contain all of the data structures used by workbench and its REST API. For example: 

.. code-block:: python

	request: {
		query: ""
		payload: {
			target_domain, target_intent, target_state
		}
		client_context: {}
		verbose=true
	}
	response:
		request: {
			query: ""
			payload: {}
			client_context: {}
		}
		domain: {
			target: order
			probs: [ ]
		},
		intent: {
			target: order
			probs: [ ]
		},
		entities: [
			{text, span, value, role}
		],
		parse_tree: {

		}
		results: {

		}
		dialog_state: {

		}
		frame: {
			basket:
			target_store
			target_index
		}
		client_actions: {
			show_reply: {}
			show_buttons: {'text': 'blah', 'payload': payload}
		}
		history: []
	}

This section should describe the purpose and structure of objects such as the parse_tree, the frame, client_actions, history, etc.
