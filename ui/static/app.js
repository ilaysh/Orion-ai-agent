// Orion WebSocket UI Controller
let socket;
let audioCtx;
let pcmNode;
let isStreaming = false;
let watchdogTimer = null;

// --- Initialize WebSocket connection ---
function connect() {
    socket = new WebSocket("ws://" + window.location.host + "/ws");

    socket.onopen = () => {
        console.log("[WebSocket] Connected ✅");
        startWatchdog();
    };

    socket.onmessage = async (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === "state") {
            handleState(msg.state);
        } else if (msg.type === "orion_reply") {
            await handleOrionReply(msg);
        }
    };

    socket.onclose = () => {
        console.warn("[WebSocket] Disconnected. Retrying...");
        stopWatchdog();
        stopAudioStream();
        setTimeout(connect, 2000);
    };
}

// --- Handle Orion state updates ---
function handleState(state) {
    document.getElementById("status").textContent = state.toUpperCase();
}

// --- Handle Orion reply (with optional audio) ---
async function handleOrionReply(msg) {
    if (msg.audio) {
        const audioData = Uint8Array.from(atob(msg.audio), (c) => c.charCodeAt(0));
        const blob = new Blob([audioData], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => {
            socket.send(JSON.stringify({ type: "playback_finished" }));
        };
        audio.play();
    } else if (msg.text) {
        console.log("[Orion Reply] " + msg.text);
    }
}

// --- Watchdog: restarts stream if idle too long ---
function startWatchdog() {
    if (watchdogTimer) clearInterval(watchdogTimer);
    watchdogTimer = setInterval(() => {
        if (!isStreaming) {
            console.warn("[Watchdog] Stream idle — restarting");
            startAudioStream();
        }
    }, 10000);
}
function stopWatchdog() {
    if (watchdogTimer) clearInterval(watchdogTimer);
    watchdogTimer = null;
}

// --- Audio Worklet Recorder ---
async function startAudioStream() {
    try {
        if (isStreaming) return;
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const source = audioCtx.createMediaStreamSource(stream);

        // Load our custom processor
        await audioCtx.audioWorklet.addModule("/static/pcm-processor.js");
        cmNode = new AudioWorkletNode(audioCtx, "pcm-processor");
        cmNode.port.onmessage = (event) => {
            const buf = event.data; // ArrayBuffer of Int16
            if (socket && socket.readyState === WebSocket.OPEN) {
                const bytes = new Uint8Array(buf);
                let binary = "";
                const chunkSize = 0x8000;
                for (let i = 0; i < bytes.length; i += chunkSize) {
                    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunkSize));
                }
                const b64 = btoa(binary);
                socket.send(JSON.stringify({ type: "user_audio", audio: b64 }));
            }
        };

        source.connect(cmNode).connect(audioCtx.destination);
        isStreaming = true;
        console.log("[Audio] 🎙️ Streaming started");
    } catch (err) {
        console.error("[Audio] ❌ Failed to start audio stream:", err);
    }
}

function stopAudioStream() {
    if (!isStreaming || !audioCtx) return;
    audioCtx.close().catch(() => { });
    pcmNode = null;
    audioCtx = null;
    isStreaming = false;
    console.log("[Audio] ⏹️ Stopped");
}

// --- Handle text input submission (optional UI) ---
function sendText() {
    const input = document.getElementById("textInput");
    if (!input) return;
    const text = input.value.trim();
    if (text && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "user_text", text }));
        input.value = "";
    }
}

// --- Start everything on page load ---
// --- Auto-connect WebSocket on load ---
window.addEventListener("load", () => {
    connect(); // connect right away to receive states
    const powerBtn = document.getElementById("powerBtn");
    if (powerBtn) {
        powerBtn.addEventListener("click", toggleMic);
    }

    const diagBtn = document.getElementById("btnDiag");
    if (diagBtn) diagBtn.addEventListener("click", recordDiagnostic);
});

// --- Toggle microphone streaming ---
async function toggleMic() {
    const btn = document.getElementById("powerBtn");
    if (!isStreaming) {
        btn.textContent = "🛑 Stop Listening";
        await startAudioStream(); // now browser allows AudioContext start (user gesture)
    } else {
        btn.textContent = "🎙️ Start Listening";
        stopAudioStream();
    }
}


// --- Diagnostic capture: same format as user_audio stream ---
async function recordDiagnostic() {
    const btn = document.getElementById("btnDiag");
    btn.disabled = true;
    btn.textContent = "Recording (10 s)...";

    try {
        const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const src = ctx.createMediaStreamSource(stream);

        // Load the same PCM processor
        await ctx.audioWorklet.addModule("/static/pcm-processor.js");
        const node = new AudioWorkletNode(ctx, "pcm-processor");
        const allChunks = [];

        node.port.onmessage = (e) => {
            // e.data is Float32Array — same as user_audio
            allChunks.push(e.data);
        };

        src.connect(node);
        // connect to destination so worklet runs
        node.connect(ctx.destination);

        // Stop after 10 s
        setTimeout(async () => {
            stream.getTracks().forEach(t => t.stop());
            node.disconnect();
            src.disconnect();
            ctx.close();

            // Concatenate Float32 chunks into one array
            const totalLen = allChunks.reduce((a, b) => a + b.length, 0);
            const joined = new Float32Array(totalLen);
            let offset = 0;
            for (const c of allChunks) {
                joined.set(c, offset);
                offset += c.length;
            }

            // Encode EXACTLY like user_audio
            const bytes = new Uint8Array(joined.buffer);
            let binary = "";
            const chunkSize = 0x8000;
            for (let i = 0; i < bytes.length; i += chunkSize) {
                const sub = bytes.subarray(i, i + chunkSize);
                binary += String.fromCharCode.apply(null, sub);
            }
            const b64 = btoa(binary);

            socket.send(JSON.stringify({
                type: "diagnostic_audio",
                audio: b64,
                sampleRate: 48000
            }));

            btn.textContent = "Sent ✅ (check logs)";
            setTimeout(() => {
                btn.textContent = "🎧 Record 10 s Diagnostic";
                btn.disabled = false;
            }, 3000);
        }, 10000);

    } catch (err) {
        console.error("Diagnostic record error:", err);
        btn.textContent = "Error ❌";
        btn.disabled = false;
    }
}

window.addEventListener("load", () => {
    const diagBtn = document.getElementById("btnDiag");
    if (diagBtn) diagBtn.addEventListener("click", recordDiagnostic);
});
