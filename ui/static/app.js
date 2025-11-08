let socket, audioCtx, pcmNode, isStreaming = false;

function connect() {
    const orb = document.getElementById("orb");
    orb.className = "orb connecting";
    document.getElementById("status").textContent = "connecting...";

    const proto = location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${proto}://${location.host}/ws`);

    socket.onopen = () => {
        orb.className = "orb idle";
        document.getElementById("status").textContent = "idle";
        console.log("[WS] ✅ Connected");
    };

    socket.onmessage = async (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "state") {
            orb.className = "orb " + msg.state;
            document.getElementById("status").textContent = msg.state;
        }
        if (msg.type === "orion_reply") {
            if (msg.text) addLog("Orion: " + msg.text);
            if (msg.audio) playAudio(msg.audio);
        }
    };

    socket.onclose = () => {
        orb.className = "orb connecting";
        document.getElementById("status").textContent = "reconnecting...";
        setTimeout(connect, 2000);
    };
}


function addLog(t) {
    const log = document.getElementById("log");
    const p = document.createElement("div");
    p.textContent = t;
    log.appendChild(p);
}

async function playAudio(b64) {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.onended = () => socket.send(JSON.stringify({ type: "playback_finished" }));
    await audio.play();
}

function sendMessage() {
    const input = document.getElementById("textInput");
    if (!input || socket.readyState !== WebSocket.OPEN) return;
    const text = input.value.trim();
    if (text) {
        socket.send(JSON.stringify({ type: "user_text", text }));
        addLog("You: " + text);
        input.value = "";
    }
}

async function startAudioStream() {
    if (isStreaming) return;
    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const src = audioCtx.createMediaStreamSource(stream);
    await audioCtx.audioWorklet.addModule("/static/pcm-processor.js");
    pcmNode = new AudioWorkletNode(audioCtx, "pcm-processor");
    const mute = audioCtx.createGain(); mute.gain.value = 0;
    pcmNode.port.onmessage = (e) => {
        const buf = new Uint8Array(e.data);
        const binary = String.fromCharCode(...buf);
        socket.send(JSON.stringify({ type: "user_audio", audio: btoa(binary) }));
    };
    src.connect(pcmNode).connect(mute).connect(audioCtx.destination);
    isStreaming = true;
    document.getElementById("powerBtn").textContent = "🛑 Stop";
}

function stopAudioStream() {
    if (!isStreaming) return;
    audioCtx.close();
    isStreaming = false;
    document.getElementById("powerBtn").textContent = "🎤 Speak";
}

window.addEventListener("load", () => {
    connect();
    document.getElementById("powerBtn").addEventListener("click", () => {
        isStreaming ? stopAudioStream() : startAudioStream();
    });
});

// --- Allow Enter to send, Shift+Enter for newline ---
window.addEventListener("keydown", (e) => {
    const input = document.getElementById("textInput");
    if (!input) return;

    // Only handle when typing in the text box
    if (document.activeElement === input) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
        // Shift+Enter → natural newline (do nothing special)
    }
});

