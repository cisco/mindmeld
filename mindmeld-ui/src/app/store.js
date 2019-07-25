import { applyMiddleware, createStore, compose } from 'redux';
import logger from 'redux-logger';
import thunk from 'redux-thunk';

import reducer from './reducer';
import middleware from './middleware';


const middlewares = [ logger, thunk, middleware ];


// export the store
const composeEnhancers = window.__REDUX_DEVTOOLS_EXTENSION_COMPOSE__ || compose; // For the redux extension
export const store = createStore(reducer, composeEnhancers(applyMiddleware(...middlewares)));
