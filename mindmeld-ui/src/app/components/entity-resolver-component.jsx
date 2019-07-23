import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Container, Row, Col } from 'reactstrap';
import get from 'lodash.get';

import Tag from './tag-component';
import QueryInput from './query-input-component';
import ResolvedEntity from './resolved-entity-component';


class EntityResolver extends Component {

  static propTypes = {
    result: PropTypes.object
  };

  render() {
    const entities = get(this.props, 'result.entities');
    return (
      <Container className="section entity-resolver">
        <Row>
          <Col>
            <p>
              The entity resolver maps each entity to a unique and unambiguous concept, such as a product with a
              specific ID or an attribute with a specific SKU number. Entity resolution is performed using machine
              learning models trained on thousands or millions of mapping-pair examples.
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
              The output of entity resolver consists of the specific concept, product or attribute which uniquely maps
              to each entity. For example, each entity might map to a specific product in a catalog or a SKU for a
              point-of-sale system.
            </p>
          </Col>
        </Row>
        { entities &&
          <Row>
            <Col>
              {
                entities.map((entity, index) => {
                  return <ResolvedEntity entity={entity} key={index}/>;
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

export default connect(mapStateToProps)(EntityResolver);

