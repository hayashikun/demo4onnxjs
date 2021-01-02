import React, {useState} from "react";
import {InferenceSession} from "onnxjs"


const MnistPage: React.FC = () => {
    const [modelLoaded, setModelLoaded] = useState(false);
    const [backend, setBackend] = useState("cpu");
    const [version, setVersion] = useState(9);
    let session = new InferenceSession({backendHint: "cpu"})

    const loadModel = async () => {
        setModelLoaded(false);
        session = new InferenceSession({backendHint: backend});
        try {
            await session.loadModel(`/models/mnist_1_v${version}.onnx`);
            setModelLoaded(true);
        } catch (e) {
            console.log(e)
        }
    }

    return (
        <div>
            <h1>MNIST</h1>

            <div>
                Backend:
                <select value={backend}
                        onChange={(e) => {
                            setBackend(e.target.value);
                            setModelLoaded(false);
                        }}>
                    <option value="cpu">cpu</option>
                    <option value="webgl">webgl</option>
                    <option value="wasm">wasm</option>
                </select>
            </div>

            <div>
                opset version:
                <select value={version}
                        onChange={(e) => {
                            setVersion(parseInt(e.target.value));
                            setModelLoaded(false);
                        }}>
                    <option value="9">9</option>
                    <option value="10">10</option>
                    <option value="11">11</option>
                    <option value="12">12</option>
                </select>
            </div>

            <div>
                <button onClick={loadModel}>Load</button>
                {modelLoaded && <span>({backend}, {version})</span>}
            </div>

        </div>
    );
};

export default MnistPage;
