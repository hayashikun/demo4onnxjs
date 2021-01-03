import React from 'react';
import {BrowserRouter, Redirect, Route, Switch} from "react-router-dom";
import MainPage from "./components/MainPage";
import MnistPage from "./components/MnistPage";

export const Routes: React.FC = () => (
    <BrowserRouter basename={process.env.PUBLIC_URL}>
        <Switch>
            <Route exact path="/" component={MainPage}/>
            <Route exact path="/mnist" component={MnistPage}/>
            <Redirect from="*" to="/"/>
        </Switch>
    </BrowserRouter>
);

export default Routes;
