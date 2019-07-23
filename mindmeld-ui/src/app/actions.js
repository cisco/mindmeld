import * as APP from './constants';

export const updateQuery = (query) => {
  return {
    type: APP.UPDATE_QUERY,
    query,
  };
};

export const executeQuery = () => {
  return {
    type: APP.EXECUTE_QUERY,
  };
};

export const onExecuteQueryEnd = (data) => {
  return {
    type: APP.ON_EXECUTE_QUERY_END,
    data,
  };
};

export const onExecuteQueryError = (error) => {
  return {
    type: APP.ON_EXECUTE_QUERY_ERROR,
    error
  };
};

export const resetState = () => {
  return {
    type: APP.RESET_STATE
  };
};

export const startRecognition = () => {
  return {
    type: APP.START_RECOGNITION,
  };
};

export const onRecognitionError = (error) => {
  return {
    type: APP.ON_RECOGNITION_ERROR,
    error
  };
};