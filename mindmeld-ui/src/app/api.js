import axios from 'axios';
import get from 'lodash.get';

const URI = process.env.REACT_APP_MINDMELD_URL || 'http://127.0.0.1:7150/parse';

export const query = async (body) => {
  if (typeof body === 'string') {
    body = {
      text: body
    };
  }

  const result = await axios({
    method: 'post',
    baseURL: URI,
    headers: {
      'Content-Type': 'application/json',
    },
    data: body
  });

  const { data } = result,
        directives = get(data, 'directives', []),
        replies = directives
                    .filter((item) => item.name === 'reply')
                    .map((item) => get(item, 'payload.text')),
        suggestions = get(directives.find((item) => item.name === 'suggestions'), 'payload', [])
                      .map((item) => get(item, 'text')),
        entities = get(data.request, 'entities', [])
                    .map((entity) => {
                      return {
                        text: entity.text,
                        type: entity.type,
                        score: entity.score,
                        span: entity.span,
                        role: entity.role,
                        value: (entity.value || []),
                        children: (entity.children || []),
                      };
                    });

  const kbObjects = [];
  entities.forEach((entity) => {
    entity['value'].forEach((value) => {
      kbObjects.push(value);
    });
  });

  data.request.confidences.entities.forEach((entity, index) => {
    entities[index]['score'] = entity[Object.keys(entity)[0]];
  });

  return {
    domain: data.request.domain,
    intent: data.request.intent,
    entities,
    replies,
    suggestions,
    response: {
      ...data
    },
    scores: data.request.confidences,
    kbObjects,
  };
};
