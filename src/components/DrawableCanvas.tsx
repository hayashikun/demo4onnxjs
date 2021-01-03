import React, {ComponentProps} from "react";

interface DrawableCanvasProps extends ComponentProps<any> {
    dataHeight: number,
    dataWidth: number,
    displayHeight: number,
    displayWidth: number,
    lineWidth?: number
}

export default class DrawableCanvas extends React.Component<DrawableCanvasProps> {
    state = {
        drawing: false
    }
    hideCanvas: HTMLCanvasElement | null = null;
    displayCanvas: HTMLCanvasElement | null = null;

    startDrawing = (x: number, y: number) => {
        this.setState({drawing: true});
        const ctx = this.displayCanvas?.getContext("2d");
        if (ctx != null) {
            ctx.beginPath();
            if (this.props.lineWidth) {
                ctx.lineWidth = this.props.lineWidth;
            }
            ctx.moveTo(x, y);
        }
    }
    endDrawing = () => {
        this.setState({drawing: false});
        const ctx = this.displayCanvas?.getContext("2d");
        if (ctx != null) {
            ctx.closePath();
        }
    }
    draw = (x: number, y: number) => {
        if (!this.state.drawing) {
            return;
        }
        const ctx = this.displayCanvas?.getContext("2d");
        if (ctx != null) {
            ctx.lineTo(x, y);
            ctx.stroke();
        }
    }

    getImageData(): Float32Array | null {
        const dCtx = this.displayCanvas?.getContext("2d");
        const hCtx = this.hideCanvas?.getContext("2d");
        if (dCtx == null || hCtx == null) {
            return null;
        }
        hCtx.save();
        hCtx.scale(
            hCtx.canvas.width / dCtx.canvas.width,
            hCtx.canvas.height / dCtx.canvas.height
        );
        hCtx.clearRect(0, 0, hCtx.canvas.width, dCtx.canvas.height);
        hCtx.drawImage(dCtx.canvas, 0, 0);
        hCtx.restore();
        const imageData = hCtx.getImageData(0, 0, this.props.dataWidth, this.props.dataHeight);
        const data = new Float32Array(this.props.dataWidth * this.props.dataHeight);
        for (let i = 0; i < data.length; ++i) {
            data[i] = imageData.data[i * 4 + 3] / 255;
        }
        return data;
    }

    clearCanvas() {
        const dCtx = this.displayCanvas?.getContext("2d");
        const hCtx = this.hideCanvas?.getContext("2d");
        if (this.displayCanvas == null || dCtx == null || this.hideCanvas == null || hCtx == null) {
            return null;
        }
        dCtx.clearRect(0, 0, dCtx.canvas.width, dCtx.canvas.height);
        hCtx.clearRect(0, 0, hCtx.canvas.height, hCtx.canvas.height);
    }

    render() {
        return (
            <div>
                <canvas
                    ref={e => this.hideCanvas = e}
                    width={this.props.dataWidth + "px"}
                    height={this.props.dataHeight + "px"}
                    style={{display: "none"}}
                />
                <canvas
                    ref={e => this.displayCanvas = e}
                    width={this.props.displayWidth + "px"}
                    height={this.props.displayHeight + "px"}
                    id="display-canvas"
                    onMouseDown={e => this.startDrawing(e.nativeEvent.offsetX, e.nativeEvent.offsetY)}
                    onMouseUp={() => this.endDrawing()}
                    onMouseLeave={() => this.endDrawing()}
                    onMouseMove={e => this.draw(e.nativeEvent.offsetX, e.nativeEvent.offsetY)}
                    style={this.props.style}
                />
            </div>
        );
    }
}
