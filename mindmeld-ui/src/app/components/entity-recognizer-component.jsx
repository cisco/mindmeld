import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Container, Row, Col } from 'reactstrap';
import get from 'lodash.get';

import Tag from './tag-component';
import QueryInput from './query-input-component';
import Entity from './entity-component';


class EntityRecognizer extends Component {

  static propTypes = {
    result: PropTypes.object
  };

  render() {
    const entities = get(this.props, 'result.entities');
    return (
      <Container className="section entity-recognizer">
        <Row>
          <Col>
            <p>
              The entity recognizer identifies the important words and phrases contained in each user request. It relies
              on machine learning sequence models trained on large numbers of user requests. The entity recognizer
              prediction is based on the user query, user context, dialogue history as well as the output of the intent
              classifier.
            </p>
          </Col>
        </Row>
        <Row className="mb-3">
          <Col>
            <QueryInput/>
          </Col>
        </Row>
        <Row className="mb-3">
          <Col>
            <Tag body="user content"/>
            <Tag body="dialogue history"/>
            <Tag body="target domain"/>
            <Tag body="target intent"/>
          </Col>
        </Row>
        <Row>
          <Col>
            <p>
              The output of the entity recognizer consists of the important words and phrases identified in the user
              request along with their associated entity types and prediction confidence.
            </p>
          </Col>
        </Row>
        { entities &&
          <Row>
            <Col>
              {
                entities.map((entity, index) => {
                  const score = String(entity.score || '').substr(0, 6);
                  let role = '';
                  if (('role' in entity) && (entity.role)) {
                    role = '|' + entity.role;
                  }
                  return <Entity name={entity.text} type={entity.type + role} confidence={score} key={index}/>;
                })
              }
            </Col>
          </Row>
        }
      </Container>
    );
  }
}

const mapStateToProps = (state) => {
  const { result } = state;

  return {
    result
  };
};

export default connect(mapStateToProps)(EntityRecognizer);
