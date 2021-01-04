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
                <li><Link to="https://github.com/hayashikun/demo4onnxjs">hayashikun/demo4onnxjs</Link></li>
                <li>
                    <Link to="https://github.com/microsoft/onnxjs">microsoft/onnxjs</Link>
                    <ul>
                        <li><Link to="https://github.com/microsoft/onnxjs/blob/master/docs/operators.md">operators.md</Link></li>
                    </ul>
                </li>
                <li>
                    <Link to="https://github.com/onnx/onnx">onnx/onnx</Link>
                    <ul>
                        <li><Link to="https://github.com/onnx/onnx/blob/rel-1.2.3/docs/Operators.md">Operators.md</Link></li>
                    </ul>
                </li>
            </ul>
        </div>
    );
};

export default MainPage;
