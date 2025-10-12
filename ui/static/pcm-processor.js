// 48k → 16k downsample with smoothing filter and Int16 output
class PCMProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._carry = new Float32Array(0);
        this._ratio = 3; // 48k -> 16k
        this._window = 9; // 9-sample moving average (~1.9 ms window)
    }

    process(inputs) {
        const input = inputs[0][0];
        if (!input) return true;

        // prepend leftover
        let samples = this._carry.length
            ? Float32Array.from([...this._carry, ...input])
            : input;

        // simple low-pass filter before decimation
        const filtered = new Float32Array(samples.length);
        const half = Math.floor(this._window / 2);
        for (let i = 0; i < samples.length; i++) {
            let sum = 0;
            let count = 0;
            for (let k = -half; k <= half; k++) {
                const idx = i + k;
                if (idx >= 0 && idx < samples.length) {
                    sum += samples[idx];
                    count++;
                }
            }
            filtered[i] = sum / count;
        }

        // downsample
        const len = Math.floor(filtered.length / this._ratio);
        const out = new Int16Array(len);
        for (let i = 0, j = 0; j < len; i += this._ratio, j++) {
            const s = Math.max(-1, Math.min(1, filtered[i]));
            out[j] = s < 0 ? s * 0x8000 : s * 0x7FFF;
        }

        // keep tail for next process call
        const consumed = len * this._ratio;
        this._carry = samples.slice(consumed);

        this.port.postMessage(out.buffer, [out.buffer]);
        return true;
    }
}
registerProcessor("pcm-processor", PCMProcessor);
