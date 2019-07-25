import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { Button } from 'reactstrap';
import { connect } from 'react-redux';

import * as actions from '../actions';


class ClearContextButton extends Component {

  static propTypes = {
    conversation: PropTypes.array,
    resetState: PropTypes.func,
  };

  constructor(props) {
    super(props);

    this.handleClearContext = this.handleClearContext.bind(this);
  }

  handleClearContext() {
    this.props.resetState();
  }

  render () {
    const { conversation } = this.props || [];
    return (
      <div className="clear-context">
        {conversation.length > 0 &&
          <Button onClick={this.handleClearContext}>Clear context</Button>
        }
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  return {
    conversation: state.conversation,
  };
};

const mapDispatchToProps = (dispatch) => {
  return {
    resetState: () => dispatch(actions.resetState()),
  };
};

export default connect(mapStateToProps, mapDispatchToProps)(ClearContextButton);
