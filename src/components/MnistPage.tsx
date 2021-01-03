import React, {useRef, useState} from "react";
import {InferenceSession, Tensor} from "onnxjs"
import DrawableCanvas from "./DrawableCanvas";


const MnistPage: React.FC = () => {
    const canvasRef: React.MutableRefObject<DrawableCanvas | null> = useRef(null);
    const [sessionStatus, setSessionStatus] = useState("");
    const [backend, setBackend] = useState("cpu");
    const [version, setVersion] = useState(9);
    const [model, setModel] = useState("Net1");
    const [inferenceSession, setInferenceSession]: [InferenceSession | null, any] = useState(null)
    const [probabilities, setProbabilities] = useState(new Float32Array(10))

    const loadModel = async () => {
        setSessionStatus("");
        let session = new InferenceSession({backendHint: backend});
        try {
            await session.loadModel(`/models/mnist_${model}_v${version}.onnx`);
            setSessionStatus(`mnist_${model}_v${version} (${backend})`);
            setInferenceSession(session);
        } catch (e) {
            setSessionStatus("Error: " + e);
        }
    }

    const evaluate = async () => {
        let canvas = canvasRef.current;
        if (canvas == null || inferenceSession == null) {
            return;
        }
        const data = canvas.getImageData();
        if (data == null) {
            return;
        }
        const tensor = new Tensor(data, "float32", [1, 1, 28, 28]);
        const result = await inferenceSession!.run([tensor]);
        setProbabilities(result.values().next().value.data);
    }

    const clearCanvas = () => {
        let canvas = canvasRef.current;
        if (canvas == null) {
            return;
        }
        canvas.clearCanvas();
    }

    return (
        <div style={{margin: "24px"}}>
            <h1>MNIST</h1>
            <div>
                Model:
                <select value={model}
                        onChange={(e) => {
                            setModel(e.target.value);
                            setSessionStatus("");
                        }}>
                    <option value="Net1">Net1</option>
                    <option value="Net2">Net2</option>
                </select>
            </div>

            <div>
                Backend:
                <select value={backend}
                        onChange={(e) => {
                            setBackend(e.target.value);
                            setSessionStatus("");
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
                            setSessionStatus("");
                        }}>
                    <option value="9">9</option>
                    <option value="10">10</option>
                    <option value="11">11</option>
                    <option value="12">12</option>
                </select>
            </div>

            <div>
                <button onClick={loadModel}>Load</button>
                <span>{sessionStatus}</span>
            </div>

            <DrawableCanvas
                ref={canvasRef}
                style={{border: "1px solid black", margin: "8px"}}
                displayHeight={300}
                displayWidth={300}
                dataHeight={28}
                dataWidth={28}
                lineWidth={15}
            />
            <button onClick={evaluate}>Eval</button>
            <button onClick={clearCanvas}>Clear</button>

            <ul>{
                (() => {
                    const ans = Array.from({length: 10}, (v, k) => probabilities[k]);
                    const idx = ans.indexOf(Math.max(...ans))
                    return ans.map((a, i) =>
                        <li key={i} style={{color: i === idx ? "red" : "black"}}>{i}: {a}</li>
                    )
                })()
            }</ul>
        </div>
    );
}

export default MnistPage;
