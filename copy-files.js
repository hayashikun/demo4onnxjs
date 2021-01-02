const fs = require("fs");
const path = require("path");


// copy ONNX.js WebAssembly files to /public
const onnxFrom = path.join(__dirname, 'node_modules', 'onnxjs', 'dist');
const onnxTo = path.join(__dirname, 'public');
fs.copyFileSync(path.join(onnxFrom, 'onnx-wasm.wasm'), path.join(onnxTo, 'onnx-wasm.wasm'));
fs.copyFileSync(path.join(onnxFrom, 'onnx-worker.js'), path.join(onnxTo, 'onnx-worker.js'));

// copy model files to /public/models
const modelsFrom = path.join(__dirname, 'models', 'data');
const modelsTo = path.join(__dirname, 'public', 'models');
fs.readdir(modelsFrom, (err, files) => {
    if (err) throw err;
    files.filter(f => {
        return fs.statSync(path.join(modelsFrom, f)).isFile() && /.*\.onnx$/.test(f)
    }).forEach(f => {
        fs.copyFileSync(path.join(modelsFrom, f), path.join(modelsTo, f));
    })
})
