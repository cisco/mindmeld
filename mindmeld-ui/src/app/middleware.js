import * as actions from './actions';
import * as APP from './constants';

import { query as apiQuery } from './api';
import * as recognition from "../recognition/speech-recognition";
import GoogleAsr from '../recognition/google-asr';


const asr = new GoogleAsr();

const DEFAULT_CONTEXT = {
};

export default store => next => async action => {
  switch (action.type) {
    case APP.EXECUTE_QUERY: {
      const state = store.getState();
      const { query, frame, history, params } = state;
      const body = {
        text: query,
        context: DEFAULT_CONTEXT,
        frame,
        history,
        params,
        verbose: true,
      };

      return await makeQuery(body, store.dispatch);
    }

    case APP.START_RECOGNITION: {
      const key = process.env.REACT_APP_GOOGLE_KEY || null;
      asr.key = key;
      asr.maxTranscripts = process.env.REACT_APP_MAX_ALTERNATIVES || 8;

      try {
        let transcripts = [];
        if (key) {
          console.log("Using Google Cloud ASR.");
          transcripts = await asr.recognize();
        } else {
          console.log("Using webkitSpeechRecognition.");
          transcripts = await recognition.recognize();
        }
        store.dispatch(actions.updateQuery(transcripts[0].transcript));
        store.dispatch(actions.executeQuery());
      } catch (error) {
        store.dispatch(actions.onRecognitionError(error));
      }
      break;
      }

    default: {
      break;
    }
  }

  return next(action);
};

const makeQuery = async (body, dispatch) => {
  try {
    const result = await apiQuery(body);
    return dispatch(actions.onExecuteQueryEnd(result));
  } catch (error) {
    return dispatch(actions.onExecuteQueryError(error));
  }
};
