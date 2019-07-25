import axios from 'axios';

import Recorder from './recorder';


export default class GoogleAsr {

  constructor(config) {
    config = {
      maxTranscripts: 1,
      key: '',
      ...config,
    };
    this._maxTranscripts = Math.max(config.maxTranscripts - 1, 0);
    this._key = config.key;
    this._recognizing = false;

    this._recorder = new Recorder();
    this._recorder.on(Recorder.ON_ERROR, this._onError.bind(this));
    this._recorder.on(Recorder.ON_DATA, this._onDataReady.bind(this));
  }

  start() {
    if (this._recognizing) {
      return;
    }

    this._recorder.start();
    this._recognizing = true;
  }

  stop() {
    this._recognizing = false;
    this._recorder.stop();
  }

  recognize() {
    return new Promise((resolve, reject) => {
      this._recognizeCallback = (transcripts) => {
        this._recognizeCallback = this._recognizeCallbackError = null;
        resolve(transcripts);
      };
      this._recognizeCallbackError = (error) => {
        this._recognizeCallback = this._recognizeCallbackError = null;
        reject(error);
      };

      this.start();
    });
  }

  set maxTranscripts(_maxTranscripts) {
    this._maxTranscripts = _maxTranscripts;
  }
  set key(_key) {
    this._key = _key;
  }

  _onDataReady(data) {
    request(data, {
      maxAlternatives: this._maxTranscripts
    }, this._key).then((result) => {
      let transcripts = result.data.results[0].alternatives.map((alt) => {
        return {
          transcript: alt.transcript,
          confidence: alt.confidence,
        };
      });
      this._recognizeCallback && this._recognizeCallback(transcripts);
    }).catch((error) => {
      this._onError();
      this._recognizeCallbackError && this._recognizeCallbackError(error);
    }).finally(() => {
      this._recognizing = false;
    });
  }

  _onError() {
    this._recognizing = false;
  }
}

export const request = (data, config, key) => {
  const requestData = data.split(',').slice(-1)[0];
  return axios({
    method: 'post',
    url: 'https://speech.googleapis.com/v1/speech:recognize',
    data: {
      audio: {
        content: requestData,
      },
      config: {
        languageCode: 'en-US',
        ...config,
      }
    },
    params: {
      key,
    }
  });
};
