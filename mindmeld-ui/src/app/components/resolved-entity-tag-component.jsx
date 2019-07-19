import PropTypes from 'prop-types';
import React, { Component } from 'react';

import Tag from "./tag-component";


const typeMap = {
  'sys_number': 'value',
  'sys_time': 'value',
  'meeting_room': 'cname',
  'person_name': 'cname',
  'job_title': 'cname',
};

class ResolvedEntityTag extends Component {

  static propTypes = {
    entity: PropTypes.object,
  };

  render () {
    const { entity } = this.props;

    const mapped = typeMap[entity.type];
    let value = entity.value;

    if (Array.isArray(value)) {
      value = value[0];
    }

    let label;
    if (mapped) {
      const displayValue = value[mapped];
      let id = value.id;

      if (id) {
        id = String(id).substr(0, 8);
        label = `ID: ${id}, ${displayValue}`;
      } else {
        label = `Value: ${displayValue}`;
      }
    } else {
      if ('cname' in value) {
        const displayValue = value['cname'];
        label = `Value: ${displayValue}`;
      }
    }
    if(label) {
      return <Tag body={label} uncapped/>;
    }
    return <Tag body="Unresolved" uncapped disabled/>;
  }
}

export default ResolvedEntityTag;
