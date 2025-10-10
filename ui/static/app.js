// let ws;
let audioContext;
let processorNode;

const micSelect = document.getElementById("micSelect");
let currentStream;

// List devices
async function listMics() {
    const devices = await navigator.mediaDevices.enumerateDevices();
    micSelect.innerHTML = "";
    devices
        .filter(d => d.kind === "audioinput")
        .forEach(d => {
            const option = document.createElement("option");
            option.value = d.deviceId;
            option.text = d.label || `Microphone ${micSelect.length + 1}`;
            micSelect.appendChild(option);
        });
}

// Start stream from selected mic
async function startMic(deviceId) {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
    }

    const constraints = {
        audio: { deviceId: deviceId ? { exact: deviceId } : undefined }
    };

    currentStream = await navigator.mediaDevices.getUserMedia(constraints);

    // connect currentStream to your ScriptProcessorNode / AudioWorklet pipeline
    setupAudioStream(currentStream);
}

// Change mic on selection
micSelect.onchange = () => {
    console.log("Switching mic to", micSelect.value);
    startMic(micSelect.value);
};

// Initialize
navigator.mediaDevices.getUserMedia({ audio: true }).then(() => {
    listMics().then(() => {
        if (micSelect.options.length > 0) {
            startMic(micSelect.value);
        }
    });
});

function connect() {
    ws = new WebSocket(`ws://${window.location.hostname}:8080/ws`);

    ws.onopen = () => {
        log("✅ WebSocket connected, starting mic…");
        startMic();  // only start microphone after connection ready
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        if (typeof msg === "string" && msg.startsWith("[Wake] confidence=")) {
            const val = parseFloat(msg.split("=")[1]);
            log(`Wake confidence: ${(val * 100).toFixed(1)}%`);
        }
        if (msg.type === "orion_reply") {
            log("Orion ▶ " + msg.text);

            if (msg.audio) {
                playAudio(msg.audio,ws);
            }
        } else if (msg.type === "user_text") {
            log("You ▶ " + msg.text);
        } else if (msg.type === "state") {
            document.getElementById("status").textContent = "Status: " + msg.state;
        }
    };

    ws.onclose = () => {
        log("⚠️ WebSocket closed");
    };
}


function startMic() {
    try {
        navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(stream);

            processorNode = audioContext.createScriptProcessor(4096, 1, 1);
            source.connect(processorNode);
            processorNode.connect(audioContext.destination);

            processorNode.onaudioprocess = (event) => {
                if (!ws || ws.readyState !== WebSocket.OPEN) return; // 🔑 guard

                const inputData = event.inputBuffer.getChannelData(0);

                // --- 🔇 Compute RMS level ---
                let rms = 0;
                for (let i = 0; i < inputData.length; i++) rms += inputData[i] * inputData[i];
                rms = Math.sqrt(rms / inputData.length);

                // --- ⚙️ Noise gate threshold ---
                // start around 0.02, tune between 0.015–0.03 depending on mic noise
                const threshold = 0.02;
                if (rms < threshold) return; // skip sending silent frames

                // --- 🎚️ Optional gain boost for weak microphones ---
                const gain = 1.2; // try 1.0–1.5
                const amplified = new Float32Array(inputData.length);
                for (let i = 0; i < inputData.length; i++) {
                    amplified[i] = Math.max(-1, Math.min(1, inputData[i] * gain));
                }

                // --- 🔄 Convert to Int16 + base64 encode ---
                const int16Data = new Int16Array(amplified.length);
                for (let i = 0; i < amplified.length; i++) {
                    int16Data[i] = amplified[i] * 0x7fff;
                }

                const audio_b64 = toBase64(int16Data.buffer);

                ws.send(
                    JSON.stringify({
                        type: "user_audio",
                        audio: audio_b64,
                    })
                );
            };
        });
    } catch (err) {
        log("mic error " + err);
    }
}


function toBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunkSize = 0x8000;
    for (let i = 0; i < bytes.length; i += chunkSize) {
        binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
    }
    return btoa(binary);
}

function sendMessage() {
    const input = document.getElementById("message");
    const text = input.value.trim();
    if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;

    ws.send(JSON.stringify({
        type: "user_text",
        text: text
    }));

    log("You ▶ " + text);
    input.value = "";
}

function playAudio1(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.play().catch(err => console.error("Playback failed:", err));
}

function playAudio(base64, ws) {
    try {
        const binary = atob(base64);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        document.getElementById("status").textContent = "Status: speaking";
        log("🔊 Playing response...");
        // --- SAFETY NET ---
        const watchdog = setTimeout(() => {
            log("Client watchdog: forcing listening reset");
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(
                    JSON.stringify({
                        type: "playback_finished",
                        watchdog: true,
                    })
                );
            }
        }, 30000); // 30s fallback

        // --- NORMAL END ---
        audio.onended = () => {
            clearTimeout(watchdog);
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(
                    JSON.stringify({
                        type: "playback_finished",
                        data: {},
                    })
                );
            }
        };

        // --- PLAY START ---
        audio.onplay = () => clearTimeout(watchdog);

        // --- START PLAYBACK ---
        audio.play().catch((err) => {
            console.error("Playback failed:", err);
            clearTimeout(watchdog);
        });
    } catch (err) {
        console.error("playAudio decode error:", err);
    }
}

function log(msg) {
    const logEl = document.getElementById("log");
    if (logEl) {
        const line = document.createElement("div");
        line.textContent = msg;
        logEl.appendChild(line);
        logEl.scrollTop = logEl.scrollHeight;
    }
}

// Start everything
connect();

// Allow pressing Enter in the input
document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("message");
    if (input) {
        input.addEventListener("keypress", (e) => {
            if (e.key === "Enter") sendMessage();
        });
    }
});
