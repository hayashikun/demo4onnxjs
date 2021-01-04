import React from "react";
import { Link } from "react-router-dom";

const MainPage: React.FC = () => {
    return (
        <div>
            <h1>Demo for onnxjs</h1>
            <ul>
                <li><Link to="/mnist">MNIST</Link></li>
            </ul>

            <ul>
                <li><a href="https://github.com/hayashikun/demo4onnxjs">hayashikun/demo4onnxjs</a></li>
                <li>
                    <a href="https://github.com/microsoft/onnxjs">microsoft/onnxjs</a>
                    <ul>
                        <li><a href="https://github.com/microsoft/onnxjs/blob/master/docs/operators.md">operators.md</a></li>
                    </ul>
                </li>
                <li>
                    <a href="https://github.com/onnx/onnx">onnx/onnx</a>
                    <ul>
                        <li><a href="https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md">Operators.md</a></li>
                    </ul>
                </li>
            </ul>
        </div>
    );
};

export default MainPage;
