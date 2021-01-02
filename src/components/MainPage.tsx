import React from "react";
import { Link } from "react-router-dom";

const MainPage: React.FC = () => {
    return (
        <div>
            <h1>Demo for onnxjs</h1>
            <ul>
                <li><Link to="/mnist">MNIST</Link></li>
            </ul>
        </div>
    );
};

export default MainPage;
