import PropTypes from 'prop-types';
import React, { Component } from 'react';
import classnames from 'classnames';


class Query extends Component {

  static propTypes = {
    active: PropTypes.bool,
    stringQuery: PropTypes.string,
    handler: PropTypes.func,
  };

  render () {
    const { active, stringQuery, handler } = this.props;
    const classNames = classnames(
      'query',
      {'active': active}
    );
    return (
      <p className={ classNames } onClick={ () => handler(stringQuery) }>
        { stringQuery }
      </p>
    )
  }
}

export default Query;
