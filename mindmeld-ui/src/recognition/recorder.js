import EventEmitter from 'events';


export default class Recorder extends EventEmitter {
  static ON_START = 'RECORDER_ON_START';
  static ON_STOP = 'RECORDER_ON_STOP';
  static ON_ERROR = 'RECORDER_ON_ERROR';
  static ON_DATA = 'RECORDER_ON_DATA';

  constructor() {
    super();
    this._onAudioProcess = this._onAudioProcess.bind(this);
    this._onSilence = this._onSilence.bind(this);
    this._onDataReady = this._onDataReady.bind(this);

    this._context = new window.AudioContext() || new window.webkitAudioContext();
    this._encoder = new Encoder();
    this._encoder.on(Encoder.ON_DATA, this._onDataReady);
    this._encoder.sampleRate = this._context.sampleRate;

    this._recording = false;
  }

  start() {
    if (this._recording) {
      return;
    }

    this._encoder.reset();
    this._started = Date.now();

    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      this._stream = stream;
      this._context.resume();
      this._source = this._context.createMediaStreamSource(stream);

      this._processor = this._context.createScriptProcessor(1024, 1, 1);
      this._processor.onaudioprocess = this._onAudioProcess;

      this._analyser = this._context.createAnalyser();
      this._analyser.minDecibels = -90;
      this._analyser.maxDecibels = -10;
      this._analyser.smoothingTimeConstant = 0.85;

      this._source.connect(this._analyser);
      this._analyser.connect(this._processor);
      this._processor.connect(this._context.destination);
    }).catch((error) => {
      this._recording = false;
      this.emit(Recorder.ON_ERROR, error);
    });
    this._recording = true;
    this.emit(Recorder.ON_START);
  }

  stop() {
    this._context.suspend();
    this._encoder.exportBuffer(this._context.sampleRate);

    this._processor.disconnect();
    this._source.disconnect();
    this._stream.getAudioTracks().forEach((stream) => {
      stream.stop();
    });
    this._recording = false;
    this.emit(Recorder.ON_STOP);
  }

  _onDataReady(data) {
    this.emit(Recorder.ON_DATA, data);
  }

  _onAudioProcess(event) {
    this._encoder.record(event.inputBuffer.getChannelData(0));
    this._detectSilence();
  }

  _onSilence() {
    this.stop();
  }

  _detectSilence() {
    this._analyser.fftSize = 2048;
    let bufferLength = this._analyser.fftSize,
        dataArray = new Uint8Array(bufferLength),
        amplitude = 0.2,
        time = 1500;

    this._analyser.getByteTimeDomainData(dataArray);

    for (let i = 0; i < bufferLength; i++) {
      // Normalize between -1 and 1.
      let curr_value_time = (dataArray[i] / 128) - 1.0;
      if (curr_value_time > amplitude || curr_value_time < (-1 * amplitude)) {
        this._started = Date.now();
      }
    }

    if (Date.now() - this._started > time) {
      this._onSilence();
    }
  }
}

class Encoder extends EventEmitter{
  static ON_DATA = 'ENCODER_ON_DATA';

  constructor() {
    super();

    this.reset();
    this._sampleRate = 48000;
    this._dataReader = new FileReader();
    this._dataReader.onloadend = this._onDataReady.bind(this);
  }

  record(input) {
    let len = input.length,
        view = new DataView(new ArrayBuffer(len * 2)),
        offset = 0;

    for (let i = 0; i < len; ++i) {
      let x = input[i] * 0x7fff;
      view.setInt16(offset, x < 0 ? Math.max(x, -0x8000) : Math.min(x, 0x7fff), true);
      offset += 2;
    }
    this._buffer.push(view);
    this._length += len;
  }

  reset() {
    this._buffer = [];
    this._length = 0;
  }

  exportBuffer() {
    this._finalize();
    let audioBlob = new Blob(this._buffer, { type: 'audio/wav' });
    this._dataReader.readAsDataURL(audioBlob);
  }

  set sampleRate(sampleRate) {
    this._sampleRate = sampleRate;
  }

  _onDataReady() {
    this.emit(Encoder.ON_DATA, this._dataReader.result);
  }

  _writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  _finalize() {
    let dataSize = this._length * 2,
        view = new DataView(new ArrayBuffer(44));
    this._writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    this._writeString(view, 8, 'WAVE');
    this._writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, this._sampleRate, true);
    view.setUint32(28, this._sampleRate * 4, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    this._writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);
    this._buffer.unshift(view);
  }
}
