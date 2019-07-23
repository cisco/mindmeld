import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { connect } from 'react-redux';

import * as actions from '../actions';


class QueryInput extends Component {

  static propTypes = {
    query: PropTypes.string,
    startRecognition: PropTypes.func,
    updateQuery: PropTypes.func,
    executeQuery: PropTypes.func,
  };

  constructor(props) {
    super(props);

    this.handleOnChange = this.handleOnChange.bind(this);
    this.handleOnSubmit = this.handleOnSubmit.bind(this);
  }

  handleOnChange(event) {
    this.props.updateQuery(event.target.value);
  }

  handleOnSubmit(event) {
    this.props.executeQuery();
    event.preventDefault();
  }

  render() {
    const { query, startRecognition } = this.props;

    return (
      <span className="query-input">
        <form onSubmit={this.handleOnSubmit}>
          <input type="text" className="text" value={query} onChange={this.handleOnChange} />
        </form>
        <span className="mic" onClick={() => startRecognition()} />
      </span>
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
    startRecognition: () => dispatch(actions.startRecognition()),
    executeQuery: () => dispatch(actions.executeQuery()),
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(QueryInput);

