import PropTypes from 'prop-types';
import React, { Component } from 'react';

import Tag from './tag-component';


class Entity extends Component {

  static propTypes = {
    name: PropTypes.string,
    type: PropTypes.string,
    confidence: PropTypes.string,
  };

  render () {
    const { name, type, confidence } = this.props;

    return (
      <div className="entity">
        <div className="name-container">
          <div className="line"/>
          <div className="name">
            { name }
          </div>
        </div>
        <Tag body={type} />
        { confidence &&
          <div className="confidence">
            confidence: {confidence}
          </div>
        }
      </div>
    );
  }
}

export default Entity;
