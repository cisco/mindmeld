export const recognize = () => {
  return new Promise((resolve, reject) => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 8;
    recognition.start();

    recognition.onresult = (event) => {
      const transcripts = [];

      for (let i = 0; i < event.results[0].length; ++i) {
        transcripts.push(
          {
            transcript: event.results[0][i].transcript,
            confidence: event.results[0][i].confidence,
          }
        );
      }
      resolve(transcripts);
    };
    recognition.onerror = (error) => {
      reject(error);
    };
  });
};
