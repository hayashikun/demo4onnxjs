import React from 'react';
import {HashRouter, Redirect, Route, Switch} from "react-router-dom";
import MainPage from "./components/MainPage";
import MnistPage from "./components/MnistPage";

export const App: React.FC = () => (
    <HashRouter basename={process.env.PUBLIC_URL}>
        <Switch>
            <Route exact path="/" component={MainPage}/>
            <Route exact path="/mnist" component={MnistPage}/>
            <Redirect from="*" to="/"/>
        </Switch>
    </HashRouter>
);

export default App;
