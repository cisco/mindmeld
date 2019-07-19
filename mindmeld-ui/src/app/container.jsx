import React from 'react';
import { hot } from 'react-hot-loader';
import { Route, HashRouter, Switch, withRouter } from 'react-router-dom';
import { connect } from 'react-redux';
import PropTypes from 'prop-types';
import {
  Row, Col,
  Nav, NavItem, NavLink,
} from 'reactstrap';

import Queries from './components/queries-component';
import Default from './components/default-component';
import DialogueManager from './components/dialogue-manager-component';
import EntityRecognizer from './components/entity-recognizer-component';
import EntityResolver from './components/entity-resolver-component';
import IntentClassifier from './components/intent-classifier-component';
import QuestionAnswerer from './components/question-answerer-component';
import SemanticParser from './components/semantic-parser-component';
import Conversation from './components/conversation-component';
import ClearContextButton from './components/clear-context-button-component';

import arrow from '../assets/images/arrow.png';



const RoutedContainer = withRouter(connect()(class Container extends React.Component {

  static propTypes = {
    location: PropTypes.object.isRequired,
  };

  path = (route) => {
    return this.isActive('/' + route) ? [(process.env.BASENAME || ''), '#', ''].join('/') : [(process.env.BASENAME || ''), '#', route].join('/');
  };

  isActive(path) {
    return this.props.location && this.props.location.pathname === path;
  }

  render() {
    return (
      <div className="main container-fluid">
        <Row><Col><h1 className="mt-4 mb-4 text-center">What is Deep-Domain Conversational AI?</h1></Col></Row>
        <Row className="sub-header text-center mb-3">
          <Col>Recent AI breakthroughs are making versatile and helpful voice and chat assistants possible for the first time.</Col>
          <Col>Using a voice or chat assistant is simple, but building one is among the hardest AI challenges that exist today.</Col>
          <Col>To illustrate, consider an AI assistant that can understand anything you might ask a food ordering assistant.</Col>
          <Col>To achieve human-like intelligence for use cases like this, Deep-Domain Conversational AI is required.</Col>
        </Row>
        <Row className="main-content">
          <Col xs={2}><Queries/></Col>
          <Col xs={7}>
            <Nav tabs>
              <NavItem className="arrow"><img alt="" src={arrow} /></NavItem>
              <NavItem>
                <NavLink href={this.path('intent-classifier')} active={this.isActive('/intent-classifier')}>Intent<br/>Classifier</NavLink>
              </NavItem>
              <NavItem className="arrow"><img alt="" src={arrow} /></NavItem>
              <NavItem>
                <NavLink href={this.path('entity-recognizer')} active={this.isActive('/entity-recognizer')}>Entity<br/>Recognizer</NavLink>
              </NavItem>
              <NavItem className="arrow"><img alt="" src={arrow} /></NavItem>
              <NavItem>
                <NavLink href={this.path('entity-resolver')} active={this.isActive('/entity-resolver')}>Entity<br/>Resolver</NavLink>
              </NavItem>
              <NavItem className="arrow"><img alt="" src={arrow} /></NavItem>
              <NavItem>
                <NavLink href={this.path('semantic-parser')} active={this.isActive('/semantic-parser')}>Semantic<br/>Parser</NavLink>
              </NavItem>
              <NavItem className="arrow"><img alt="" src={arrow} /></NavItem>
              <NavItem>
                <NavLink href={this.path('question-answerer')} active={this.isActive('/question-answerer')}>Question<br/>Answerer</NavLink>
              </NavItem>
              <NavItem className="arrow"><img alt="" src={arrow} /></NavItem>
              <NavItem>
                <NavLink href={this.path('dialogue-manager')} active={this.isActive('/dialogue-manager')}>Dialogue<br/>Manager</NavLink>
              </NavItem>
              <NavItem className="arrow"><img  alt="" src={arrow} /></NavItem>
            </Nav>
            <div className="content-container container-fluid">
              <Row className="content">
                <Col>
                  <Switch>
                    <Route exact path='/' component={Default}/>
                    <Route exact path='/intent-classifier' component={IntentClassifier}/>
                    <Route exact path='/entity-recognizer' component={EntityRecognizer}/>
                    <Route exact path='/entity-resolver' component={EntityResolver}/>
                    <Route exact path='/semantic-parser' component={SemanticParser}/>
                    <Route exact path='/question-answerer' component={QuestionAnswerer}/>
                    <Route exact path='/dialogue-manager' component={DialogueManager}/>
                  </Switch>
                </Col>
              </Row>
            </div>
          </Col>
          <Col xs={3} className="conversation-container">
            <ClearContextButton/>
            <Conversation />
          </Col>
        </Row>
      </div>
    );
  }
}));

class ContainerComponent extends React.Component {

  static propTypes = { };

  render() {
    return (
      <HashRouter basename={process.env.BASENAME || '/'}>
        <RoutedContainer/>
      </HashRouter>
    );
  }
}

export default hot(module)(ContainerComponent);
