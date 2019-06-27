import PropTypes from 'prop-types';
import React, { Component } from 'react';
import Tag from "./tag-component";

class KnowledgebaseItem extends Component {

  static propTypes = {
    label: PropTypes.string,
    item: PropTypes.object,
  };

  render() {
    const { label, item } = this.props;
    return (
      <div className="tree">
        <div className="label">{ label }</div>
        <ul>
          <li>
            <Tag body={'ID: ' + item.id} uncapped full />
            <ul>
              {
                Object.keys(item).map(function(key, index) {
                  if (key === 'id' || key === 'displayName') {
                    return null;
                  }
                  return (
                    <li key={index}>
                      <Tag body={key + ': ' + item[key]} uncapped full key={index}/>
                    </li>
                  );
                })
              }
            </ul>
          </li>
        </ul>
      </div>
    );
  }
}

export default KnowledgebaseItem;
