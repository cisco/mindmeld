import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { Container, Row, Col } from 'reactstrap';
import { connect } from 'react-redux';

import Tag from './tag-component';
import QueryInput from './query-input-component';
import EntityGroups from './entity-groups-component';


class SemanticParser extends Component {

  static propTypes = {
    result: PropTypes.object
  };

  render() {
    const { result } = this.props;
    const { entities } = (result || { entities: [] });
    return (
      <Container className="section semantic-parser">
        <Row>
          <Col>
            <p>
              The semantic parser determines the relationships and dependencies between the identified entities in order to understand the meaning of the user request. It relies on supervised machine learning models trained on thousands or millions of examples which illustrate the relationships between parsed entities.
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
          </Col>
        </Row>
        <Row>
          <Col>
            <p>
              The output of the semantic parser is a data structure or logical form which defines the associations and interdependencies between the parsed entities in the user request. This output can be used to construct a knowledge-base query to retrieve candidate answers, to invoke a function to perform a specific task, or to assemble an order basket for submission to a point-of-sale system.
            </p>
          </Col>
        </Row>
        { entities.length &&
          <Row>
            <Col>
              <EntityGroups entities={ entities }/>
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

export default connect(mapStateToProps)(SemanticParser);
