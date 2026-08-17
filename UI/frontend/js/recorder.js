let mediaRecorder;
let audioChunks = [];
let audioBlob = null;
let audioFilename = 'recording.webm';
let recordingTimer;
let startTime;
let audioContext;
let analyser;
let dataArray;
let animationId;

// OGG/Opus first: the backend can send it to the speech-to-text API as-is,
// while WebM needs an ffmpeg transcode on the server.
const PREFERRED_TYPES = [
    { mime: 'audio/ogg;codecs=opus', ext: 'ogg' },
    { mime: 'audio/ogg', ext: 'ogg' },
    { mime: 'audio/webm;codecs=opus', ext: 'webm' },
    { mime: 'audio/webm', ext: 'webm' },
];

function pickRecordingType() {
    if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) {
        return { mime: '', ext: 'webm' };
    }
    return PREFERRED_TYPES.find(t => MediaRecorder.isTypeSupported(t.mime)) || { mime: '', ext: 'webm' };
}

function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

export function initRecorder() {
    const recordBtn = document.getElementById('record-btn');
    const playBtn = document.getElementById('play-btn');
    const rerecordBtn = document.getElementById('rerecord-btn');

    recordBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            stopRecording();
        } else {
            await startRecording();
        }
    });

    playBtn.addEventListener('click', () => {
        if (!audioBlob) return;
        const url = URL.createObjectURL(audioBlob);
        const audio = new Audio(url);
        audio.addEventListener('ended', () => URL.revokeObjectURL(url));
        audio.play();
    });

    rerecordBtn.addEventListener('click', resetRecorder);
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const type = pickRecordingType();

        mediaRecorder = type.mime ? new MediaRecorder(stream, { mimeType: type.mime }) : new MediaRecorder(stream);
        audioChunks = [];

        setupVisualizer(stream);

        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });

        mediaRecorder.addEventListener('stop', () => {
            const blobType = mediaRecorder.mimeType || type.mime || 'audio/webm';
            audioBlob = new Blob(audioChunks, { type: blobType.split(';')[0] });
            audioFilename = `recording.${type.ext}`;

            stream.getTracks().forEach(track => track.stop());
            if (audioContext) audioContext.close();
            cancelAnimationFrame(animationId);

            const recordBtn = document.getElementById('record-btn');
            recordBtn.classList.remove('recording');
            recordBtn.style.display = 'none';
            document.getElementById('audio-controls').classList.remove('hidden');

            const duration = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('record-status').textContent = `Recording saved (${formatTime(duration)})`;
        });

        mediaRecorder.start();
        startTime = Date.now();
        document.getElementById('record-btn').classList.add('recording');

        recordingTimer = setInterval(() => {
            const duration = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('record-status').textContent = `Recording… (${formatTime(duration)})`;
            if (duration >= 120) stopRecording();
        }, 1000);

    } catch (err) {
        console.error('Error accessing microphone:', err);
        document.getElementById('record-status').textContent =
            'Could not access the microphone. Check your browser permissions.';
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        clearInterval(recordingTimer);

        const canvas = document.getElementById('waveform-canvas');
        canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
    }
}

function setupVisualizer(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    audioContext.createMediaStreamSource(stream).connect(analyser);

    analyser.fftSize = 256;
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');

    // Canvas cannot resolve var(--…), so the theme colours are read here.
    const surface = cssVar('--color-surface', '#FFFFFF');
    const accent = cssVar('--color-accent', '#E8503A');

    function draw() {
        animationId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArray);

        ctx.fillStyle = surface;
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / dataArray.length) * 2.5;
        let x = 0;

        ctx.fillStyle = accent;
        for (let i = 0; i < dataArray.length; i++) {
            const barHeight = dataArray[i] / 2;
            ctx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
            x += barWidth + 1;
        }
    }
    draw();
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
}

export function getAudioBlob() {
    return audioBlob;
}

export function getAudioFilename() {
    return audioFilename;
}

export function resetRecorder() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    }
    audioBlob = null;
    audioChunks = [];
    clearInterval(recordingTimer);

    const recordBtn = document.getElementById('record-btn');
    recordBtn.style.display = 'flex';
    recordBtn.classList.remove('recording');
    document.getElementById('audio-controls').classList.add('hidden');
    document.getElementById('record-status').textContent = 'Click to start recording';

    const canvas = document.getElementById('waveform-canvas');
    canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
}
