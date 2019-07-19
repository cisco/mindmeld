import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Container, Row, Col } from 'reactstrap';

import Tag from './tag-component';
import QueryInput from './query-input-component';


const getLabelConf = (toTransform) => {
  if (!toTransform) {
    return [];
  }
  const sorted = Object.keys(toTransform).map((item) => {
    return {
      label: item,
      confidence: toTransform[item] * 100,
    };
  }).sort((a, b) => {
    return b.confidence - a.confidence;
  });

  const max = sorted[0].confidence;

  return sorted.map((item) => {
    return {
      ...item,
      confidence: item.confidence / max * 100,
    };
  });
};

class FillBar extends Component {

  static propTypes = {
    label: PropTypes.string,
    fill: PropTypes.number,
  };

  render() {
    let { fill, label } = this.props;
    fill = Math.ceil(Math.floor(fill || 0, 0), 100);

    return (
      <div className="fill-bar">
        <div className="label">{label}</div>
        <div className="fill-container">
          <div className="fill" style={{'width': `${fill}%`}}/>
        </div>
      </div>
    );
  }
}

class IntentClassifier extends Component {

  static propTypes = {
    result: PropTypes.object,
  };

  renderResult() {
    if (!this.props.result) {
      return;
    }

    const { domain, intent, scores } = this.props.result;
    let { domains, intents } = (scores || {});
    domains = getLabelConf(domains);
    intents = getLabelConf(intents);

    return <div>
      <Row className="mb-3">
        <Col>
          <Tag label="target domain" body={ domain }/>
        </Col>
        <Col>
          <Tag label="target intent" body={ intent }/>
        </Col>
      </Row>
      <Row>
        <Col>
          {
            domains.map((domain, index) => {
              return <FillBar label={domain.label} fill={domain.confidence} key={index}/>;
            })
          }
        </Col>
        <Col>
          {
            intents.map((intent, index) => {
              return <FillBar label={intent.label} fill={intent.confidence} key={index}/>;
            })
          }
        </Col>
      </Row>
    </div>;
  }

  render() {
    return (
      <Container className="section intent-classifier">
        <Row>
          <Col>
            <p>
              The intent classifier determines what the user request is most likely trying to accomplish. It relies on a
              collection of machine learning models trained on thousands or millions of examples to predict an intent
              for each request. The input signals for the intent classifier include the user query, the user context and
              the dialogue history.
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
          </Col>
        </Row>
        <Row>
          <Col>
            <p>
              The trained machine learning models output the target domain, the target intent as well as likelihood
              probabilities for the other domains and intents supported in the application.
            </p>
          </Col>
        </Row>
        { this.renderResult() }
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

export default connect(mapStateToProps)(IntentClassifier);
