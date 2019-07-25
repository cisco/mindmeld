import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { Container, Row, Col } from 'reactstrap';
import { connect } from 'react-redux';

import * as actions from '../actions';
import Query from './query-component';

const QUERIES = {
  'Food Ordering': [
    'What do you have for dinner?',
    'Order pizza',
    'Order pizza with pepperoni and a salad',
    'What kind of fish dishes are there?',
    'Submit the order',
  ],
  'Video Discovery': [
    'Show me movies with Robert Downey Jr',
    'Gladiator',
    'What sci fi shows are available to watch?',
    'James Cameron movies',
    'Show me comedies'
  ],
  'Kwik E Mart': [
    'Hello!',
    'Where is my nearest Kwik-E-Mart store?',
    'Where can I find donuts?',
    'Is 181st street open 24 hours?',
    'Can i get a squishee at 181st at 3 am today?'
  ],
  'Home Assistant': [
    'Set the living room to 75 degrees',
    'I need the lights to be on in the whole house',
    'Lock the bedroom doors',
    'Open garage door',
    'Set alarm for 6 am'
  ]
};

class Queries extends Component {

  static propTypes = {
    query: PropTypes.string,
    updateQuery: PropTypes.func,
    executeQuery: PropTypes.func,
  };

  constructor(props) {
    super(props);

    this.updateQuery = this.updateQuery.bind(this);
  }

  updateQuery(query) {
    this.props.updateQuery(query);
    this.props.executeQuery();
  }

  render() {
    const { query } = this.props;

    return (
      <Container fluid className="queries-container">
        <Row>
          <Col className="pt-3 pb-3">
            { Object.keys(QUERIES).map((blueprint, blueprintIndex) => {
              return (
                <div className={'sub-header text-center'} key={ `blueprint-${ blueprintIndex }` }>
                  { blueprint }
                  { QUERIES[blueprint].map((stringQuery, index) => {
                    return <Query key={ `query-${ index }` }
                                  active={ stringQuery.toLowerCase() === query.toLowerCase() }
                                  stringQuery={ stringQuery }
                                  handler={ this.updateQuery } />
                  })}
                </div>
              )
            })}
          </Col>
        </Row>
      </Container>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    query: state.query
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    updateQuery: (query) => dispatch(actions.updateQuery(query)),
    executeQuery: () => dispatch(actions.executeQuery()),
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(Queries);

