let socket, audioCtx, pcmNode, isStreaming = false;
let audioEnabled = false;

// Initialize audio context on first user interaction
// --- THE AUTOPLAY UNLOCKER ---
function enableAudio() {
    if (!audioEnabled) {
        // Play a microscopic, silent base64 WAV file to securely unlock the HTML5 Audio engine
        const unlocker = new Audio("data:audio/wav;base64,UklGRigAAABXQVZFZm10IBIAAAABAAEARKwAAIhYAQACABAAAABkYXRhAgAAAAEA");

        unlocker.play().then(() => {
            audioEnabled = true;
            console.log("[Audio] 🔓 Browser Autoplay Restrictions Unlocked!");

            // Remove listeners once successfully unlocked to save memory
            document.removeEventListener("click", enableAudio);
            document.removeEventListener("keydown", enableAudio);
            document.removeEventListener("touchstart", enableAudio);
        }).catch(err => {
            console.warn("[Audio] Waiting for stronger user gesture to unlock audio...");
        });
    }
}

// Bind to ALL forms of initial interaction, not just clicks
document.addEventListener("click", enableAudio);
document.addEventListener("keydown", enableAudio);
document.addEventListener("touchstart", enableAudio);


// --- UPDATE SEND MESSAGE ---
function sendMessage() {
    enableAudio(); // Force an unlock attempt the moment you hit send

    const input = document.getElementById("textInput");
    if (!input || socket.readyState !== WebSocket.OPEN) return;
    const text = input.value.trim();
    if (text) {
        socket.send(JSON.stringify({ type: "user_text", text }));
        addLog("You: " + text);
        input.value = "";
    }
}

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
        if (msg.type === "speak") {
            console.log("[WS] 🔊 Received TTS audio");
            if (msg.audio) {
                if (!audioEnabled) {
                    console.warn("[Audio] ⚠️ Autoplay blocked. Click anywhere to enable audio.");
                    // Show user prompt to enable audio
                    addLog("🔇 Click anywhere to enable audio playback");
                } else {
                    playAudio(msg.audio);
                }
            }
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
async function playPreemptiveAudio(b64) {
    if (currentAudioNode) {
        console.log("[Audio] 🛑 Interrupting current playback for new audio.");
        currentAudioNode.pause();
        currentAudioNode.currentTime = 0;
        currentAudioNode.onended = null;
        currentAudioNode = null;
    }

    try {
        console.log("[Audio] ▶️ Playing new audio payload.");
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);

        currentAudioNode = new Audio(url);

        currentAudioNode.onended = () => {
            URL.revokeObjectURL(url);
            currentAudioNode = null;
            socket.send(JSON.stringify({ type: "playback_finished" }));
        };

        await currentAudioNode.play();

    } catch (err) {
        console.error("[Audio] ❌ Playback failed:", err);
        if (err.name === "NotAllowedError") {
            addLog("🔇 Audio blocked - click anywhere to enable");
        }

        // --- THE FAIL-SAFE UNLOCKER ---
        // If it fails to play, release the backend lock immediately so it doesn't freeze
        socket.send(JSON.stringify({ type: "playback_finished" }));
    }
}

async function playAudio(b64) {
    console.log("[Audio] ▶️ Playing response audio");
    try {
        const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
        const blob = new Blob([bytes], { type: "audio/wav" });
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => socket.send(JSON.stringify({ type: "playback_finished" }));
        await audio.play();
    } catch (err) {
        console.error("[Audio] ❌ Playback failed:", err);
        if (err.name === "NotAllowedError") {
            addLog("🔇 Audio blocked - click anywhere to enable");
        }

        // --- THE FAIL-SAFE UNLOCKER ---
        socket.send(JSON.stringify({ type: "playback_finished" }));
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

    // Enable audio on any click (satisfies browser autoplay policy)
    document.addEventListener("click", enableAudio, { once: true });

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

