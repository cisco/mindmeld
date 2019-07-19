import React, { Component } from 'react';
import { Container, Row, Col } from 'reactstrap';


class SemanticParser extends Component {

  render() {
    return (
      <Container className="section default">
        <Row>
          <Col xs={2} />
          <Col xs={8}>
            Click each of the steps above to learn how Deep-Domain Conversational AI can understand natural language and power a voice or chat conversational interface.
          </Col>
          <Col xs={2}/>
        </Row>
        <Row>
          <Col xs={2} className="mt-5">
            The input can be any one of the many trillions of request permutations a user might ask a barista.
          </Col>
          <Col xs={8} />
          <Col xs={2} className="mt-5">
            The output is a natural language conversation which resembles a typical human interaction.
          </Col>
        </Row>
      </Container>
    );
  }
}

export default SemanticParser;
