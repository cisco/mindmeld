import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { Container, Row, Col } from 'reactstrap';

import EntityGroup from './entity-group-component';


class EntityGroups extends Component {

  static propTypes = {
    entities: PropTypes.array
  };

  // Remove entities that are duplicated. That is, if they appear as a child
  // and also independently.
  filterEntities(entities) {
    const cleanEntities = entities.slice(0);
    entities.forEach((entity) => {
      if ('children' in entity) {
        entity['children'].forEach((child) => {
          const spanStart = child.span.start;
          const spanEnd = child.span.end;
          let removeIndex = -1;
          cleanEntities.forEach((cleanEntity, index) => {
            if ((cleanEntity.span.start === spanStart) && (cleanEntity.span.end === spanEnd)) {
              removeIndex = index;
            }
          });
          cleanEntities.splice(removeIndex, 1);
        });
      }
    });

    return cleanEntities;
  }

  render() {
    const { entities } = this.props;
    const cleanEntities = this.filterEntities(entities);
    return (
      <div className="semantic-structure">
        <Container>
          <Row>
            <Col className="group">
              <div className="tree">
                <div className="label">Groups</div>
                <Container>
                  <Row>
                  {
                    cleanEntities.map((entity, index) => {
                      return <EntityGroup entity={entity} key={index} label={'group ' + (index + 1)} />;
                    })
                  }
                  </Row>
                </Container>
              </div>
            </Col>
          </Row>
        </Container>
      </div>
    );
  }
}

export default EntityGroups;
