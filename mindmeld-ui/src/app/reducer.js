import * as APP from './constants';


const initialResult = {
  domain: null,
  intent: null,
  entities: [],
  scores: {
    domains: {},
    intents: {},
  },
};
const initialState = {
  query: '',
  error: '',
  result: null,
  conversation: [],
  frame: {},
  history: [],
  params: {},
  webkitSpeechRecognitionEnabled: ('webkitSpeechRecognition' in window),
};

export const reducer = (state = initialState, action) => {
  switch (action.type) {
    case APP.UPDATE_QUERY: {
      return {
        ...state,
        query: action.query,
      };
    }

    case APP.ON_EXECUTE_QUERY_END: {
      let response = {...action.data.response};

      let conversation = [
        ...(state.conversation || []),
        response,
      ];

      return {
        ...state,
        result: {
          ...initialResult,
          ...action.data,
        },
        conversation,
        frame: response.frame,
        history: response.history,
        params: response.params,
      };
    }

    case APP.ON_EXECUTE_QUERY_ERROR: {
      return {
        ...state,
        result: null,
        error: action.error
      };
    }

    case APP.RESET_STATE: {

      return {
        ...state,
        query: '',
        error: '',
        result: null,
        conversation: [],
        frame: {},
        history: [],
        params: {},
      };
    }

    case APP.ON_RECOGNITION_ERROR: {
      return {
        ...state,
        error: action.error
      };
    }

    default:
      return state;
  }
};

export default reducer;
