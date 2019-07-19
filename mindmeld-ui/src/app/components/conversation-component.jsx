import PropTypes from 'prop-types';
import React, { Component } from 'react';
import { connect } from 'react-redux';
import { Container, Row, Col } from 'reactstrap';
import get from 'lodash.get';
import classnames from 'classnames';

import * as actions from '../actions';
import logo from '../../assets/images/logo_orange.png';


class Side extends Component {
  static propTypes = {
    user: PropTypes.bool,
    hideLogo: PropTypes.bool,
    text: PropTypes.string,
  };

  render() {
    const { user, hideLogo, text } = this.props;

    const classNames = classnames(
      'bubble',
      'mb-3',
      { user: user }
    );

    const colProps = {
      size: 8,
      offset: user ? 4 : 0
    };

    return (
      <Container fluid>
        <Row>
          { !user && (
            <Col className="logo">
              {!hideLogo && <img alt="" src={logo} width={24} height={24} />}
              {hideLogo && <div className='logo-space'/>}
            </Col>
          )}
          <Col className={classNames} xs={colProps}>
            {text}
          </Col>
        </Row>
      </Container>
    );
  }
}

class List extends Component {
  static propTypes = {
    list: PropTypes.object,

    onItemClick: PropTypes.func,
  };

  render() {
    const { list, onItemClick } = this.props;

    const itemsPerPage = 4;
    const currentPage = 0;
    let displayItems = [];
    for (let i = 0; i < itemsPerPage; i++) {
      const adjustedIndex = i + itemsPerPage*currentPage;
      if (adjustedIndex < list.payload.length) {
        displayItems.push(list.payload[adjustedIndex]);
      }
    }

    return <div className="list mb-3">
      {
        displayItems.map((item, index) => {
          return <div className="item" key={index} onClick={() => onItemClick(`${item.title}`)}>
            <div className="text">{item.title}</div>
          </div>;
        })
      }
    </div>;
  }
}

class Suggestions extends Component {
  static propTypes = {
    data: PropTypes.array,

    onItemClick: PropTypes.func,
  };

  render() {
    const { data, onItemClick } = this.props;
    return <div className="suggestions mb-3">
      {
        data.map((item, index) => {
          return <div className="bubble mr-2 mb-2" key={index} onClick={() => onItemClick(item)}>{item}</div>;
        })
      }
    </div>;
  }
}

class Turn extends Component {
  static propTypes = {
    data: PropTypes.object,
    updateQuery: PropTypes.func,
    executeQuery: PropTypes.func,
  };

  constructor(props) {
    super(props);

    this.onFollowUp = this.onFollowUp.bind(this);
  }

  onFollowUp(text) {
    const { updateQuery, executeQuery } = this.props;

    updateQuery(text);
    executeQuery();
  }

  render() {
    const { data } = this.props,
          directives = get(data, 'directives', []),
          requestText = get(data, 'request.text'),
          replies = directives
              .filter((item) => item.name === 'reply')
              .map((item) => get(item, 'payload.text')),
          list = directives.find((item) => item.name === 'list'),
          suggestions = get(directives.find((item) => item.name === 'suggestions'), 'payload', []).map((item) => get(item, 'text'));

    return (
      <Row>
        <Col>
          { requestText && <Side user text={requestText} /> }
          { !!replies.length && (
            <div>
              {
                replies.map((reply, index) => {
                  if (index === 0) {
                      return <Side key={index} text={reply}/>;
                  } else {
                      return <Side key={index} text={reply} hideLogo={true}/>;
                  }
                })
              }
            </div>
          )}
          { list && <List list={list} onItemClick={this.onFollowUp} /> }
          { !!suggestions.length && <Suggestions data={suggestions} onItemClick={this.onFollowUp} /> }
        </Col>
      </Row>
    );

  }
}

const ConnectedTurn = connect(null, (dispatch) => {
  return {
    updateQuery: (query) => dispatch(actions.updateQuery(query)),
    executeQuery: () => dispatch(actions.executeQuery()),
  };
})(Turn);

class Conversation extends Component {

  static propTypes = {
    conversation: PropTypes.array,
  };

  constructor(props) {
    super(props);

    this.conversationRef = React.createRef();
  }

  componentDidUpdate() {
    const node = this.conversationRef.current;
    if (node.scrollTo) {
      node.scrollTo({
        top: node.scrollHeight,
        behavior: 'smooth'
      });
    } else {
      node.scrollTop = node.scrollHeight;
    }
  }

  render() {
    const { conversation } = this.props;

    return (
      <div className="container-fluid conversation pt-3" ref={this.conversationRef}>
        {
          conversation && conversation.map((item, index) => {
            return <ConnectedTurn data={item} key={index} />;
          })
        }
      </div>
    );
  }
}

const mapStateToProps = (state) => {
  const conversation = state.conversation;

  return {
    conversation
  };
};

export default connect(mapStateToProps)(Conversation);
