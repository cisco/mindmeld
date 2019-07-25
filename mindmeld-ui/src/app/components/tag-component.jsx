import PropTypes from 'prop-types';
import React, { Component } from 'react';
import classNames from 'classnames';


class Tag extends Component {

  static propTypes = {
    label: PropTypes.string,
    body: PropTypes.string,
    uncapped: PropTypes.bool,
    disabled: PropTypes.bool,
    full: PropTypes.bool,
  };

  render () {
    const { label, body, uncapped, disabled, full } = this.props;
    const tagClasses = classNames('tag', {
      disabled
    });
    const bodyClasses = classNames('body', {
      uncapped, full
    });
    return (
      <div className={tagClasses}>
        { label && <div className="label">{label}:</div> }
        <div className={bodyClasses}>{body}</div>
      </div>
    );
  }
}

export default Tag;
