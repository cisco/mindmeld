import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { Container, Row, Col } from 'reactstrap';
import { connect } from 'react-redux';
import get from 'lodash.get';

import Tag from './tag-component';
import QueryInput from './query-input-component';
import KnowledgebaseItem from './knowledgebase-item-component';


class QuestionAnswerer extends Component {

  static propTypes = {
    result: PropTypes.object
  };

  render() {
    const result = get(this.props, 'result', {});
    const items = get(result, 'kbObjects', []);
    return (
      <Container className="section question-answerer">
        <Row>
          <Col>
            <p>
              The question answerer finds the best answer candidates to satisfy each user request. It can rely on a knowledge graph, containing comprehensive catalog or product data for example, in order to check the validity of each candidate response as well as provide relevant recommendations and suggestions.
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
            <Tag body="user content" />
            <Tag body="dialogue history" />
            <Tag body="target domain" />
            <Tag body="target intent" />
            <Tag body="resolved entities" />
            <Tag body="resolved dependencies" />
          </Col>
        </Row>
        <Row>
          <Col>
            <p>
              The output of the question answerer consists of the results returned from applying the semantic parser output to the knowledge graph. This can include validation confirmations for high-confidence answer candidates as well as error notifications for invalid or ill-formed requests. The question answerer can also return response recommendations for answer candidates determined likely to be relevant to the user request.
            </p>
          </Col>
        </Row>
        { items &&
          <Row>
            <Col>
              {
                items.map((item, index) => {
                  return <KnowledgebaseItem label={'Item ' + (index + 1) + ':'} item={item} key={index}/>;
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

export default connect(mapStateToProps)(QuestionAnswerer);
