import PropTypes from 'prop-types';
import React, { Component } from 'react';

import Entity from './entity-component';
import ResolvedEntityTag from './resolved-entity-tag-component';

class ResolvedEntity extends Component {

  static propTypes = {
    entity: PropTypes.object,
  };

  render () {
    const { entity } = this.props;

    let role = '';
    if (entity['role']) {
      role = '|' + entity.role;
    }

    return <div className="resolved-entity">
      <div className="entity-container">
        <div className="line"/>
        <div className="arrowhead"/>
        <Entity name={entity.text} type={entity.type + role} />
      </div>
      <ResolvedEntityTag entity={entity}/>
    </div>;
  }
}

export default ResolvedEntity;
