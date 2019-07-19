import 'babel-polyfill';
import 'bootstrap/dist/css/bootstrap.min.css';

import { Provider } from 'react-redux';
import React from 'react';
import ReactDOM from 'react-dom';
import * as serviceWorker from './serviceWorker';

import { store } from './app/store';
import App from './app/container';

import './style.css';


const rootEl = document.getElementById('app');
ReactDOM.render(
  <Provider store={store}>
    <App/>
  </Provider>,
  rootEl
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();

