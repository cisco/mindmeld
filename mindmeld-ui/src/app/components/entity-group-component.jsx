import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { Col } from 'reactstrap';

import ResolvedEntityTag from './resolved-entity-tag-component';


class EntityGroup extends Component {

  static propTypes = {
    entity: PropTypes.object,
    label: PropTypes.string,
  };

  render() {
    const { entity , label } = this.props;
    return (
      <Col xs={6} className="group entity-group">
        <div className="tree">
          <div className="label">{label}</div>
          <ul>
            <li>
              <ResolvedEntityTag entity={entity}/>
              { !!entity.children.length &&
                <ul>
                  {
                    entity.children.map((child, childIndex) => {
                      return (
                        <li key={childIndex}>
                          <ResolvedEntityTag entity={child} key={childIndex}/>
                        </li>
                      );
                    })
                  }
                </ul>
              }
            </li>
          </ul>
        </div>
      </Col>
    );
  }
}

export default EntityGroup;
