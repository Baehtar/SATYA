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

// Uploaded file (audio or video) — mutually exclusive with the mic recorder.
let uploadedFile = null;

// OGG/Opus first: the backend can send it to the speech-to-text API as-is,
// while WebM needs an ffmpeg transcode on the server.
const PREFERRED_TYPES = [
    { mime: 'audio/ogg;codecs=opus', ext: 'ogg' },
    { mime: 'audio/ogg', ext: 'ogg' },
    { mime: 'audio/webm;codecs=opus', ext: 'webm' },
    { mime: 'audio/webm', ext: 'webm' },
];

// MIME types accepted for direct audio upload
const ACCEPTED_AUDIO_TYPES = [
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave', 'audio/ogg',
    'audio/flac', 'audio/aac', 'audio/mp4', 'audio/x-m4a', 'audio/m4a',
    'audio/aiff', 'audio/x-aiff',
];
// MIME types accepted for video upload (audio will be extracted server-side)
const ACCEPTED_VIDEO_TYPES = [
    'video/mp4', 'video/quicktime', 'video/webm', 'video/x-matroska',
    'video/x-msvideo', 'video/mpeg', 'video/3gpp',
];
const ACCEPTED_AUDIO_EXTS = ['.mp3','.wav','.ogg','.flac','.aac','.m4a','.aiff','.opus'];
const ACCEPTED_VIDEO_EXTS = ['.mp4','.mov','.webm','.mkv','.avi','.mpeg','.mpg','.3gp'];

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

function reportError(message) {
    document.dispatchEvent(new CustomEvent('satya:error', { detail: message }));
}

// ── File type helpers ─────────────────────────────────────────────────────────

function _isAudioType(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    return ACCEPTED_AUDIO_TYPES.includes(file.type) || ACCEPTED_AUDIO_EXTS.includes(ext);
}

function _isVideoType(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    return ACCEPTED_VIDEO_TYPES.includes(file.type) || ACCEPTED_VIDEO_EXTS.includes(ext);
}

function _formatBytes(bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ── Recorder (microphone) ─────────────────────────────────────────────────────

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
    // If a file was uploaded, clear it first — mic takes priority.
    if (uploadedFile) resetUploadedFile();

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

// ── Audio/Video file upload ───────────────────────────────────────────────────

export function initAudioUpload() {
    const dropzone = document.getElementById('audio-dropzone');
    const fileInput = document.getElementById('audio-file-input');
    const removeBtn = document.getElementById('audio-remove-btn');

    if (!dropzone || !fileInput) return; // graceful: elements may not exist yet

    // Drag and drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });
    ['dragenter', 'dragover'].forEach(evt => {
        dropzone.addEventListener(evt, () => dropzone.classList.add('drag-active'), false);
    });
    ['dragleave', 'drop'].forEach(evt => {
        dropzone.addEventListener(evt, () => dropzone.classList.remove('drag-active'), false);
    });
    dropzone.addEventListener('drop', e => {
        const file = e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) _handleAudioFile(file);
    }, false);

    // Click to browse
    dropzone.addEventListener('click', e => {
        if (removeBtn && (e.target === removeBtn || removeBtn.contains(e.target))) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) _handleAudioFile(this.files[0]);
    });

    if (removeBtn) {
        removeBtn.addEventListener('click', e => {
            e.stopPropagation();
            resetUploadedFile();
        });
    }
}

function _handleAudioFile(file) {
    const isAudio = _isAudioType(file);
    const isVideo = _isVideoType(file);

    if (!isAudio && !isVideo) {
        reportError('Unsupported file type. Upload an audio file (MP3, WAV, OGG, FLAC, AAC, M4A) or a video file (MP4, MOV, WebM, MKV).');
        return;
    }

    const maxBytes = isVideo ? 50 * 1024 * 1024 : 25 * 1024 * 1024;
    const limitLabel = isVideo ? '50 MB' : '25 MB';
    if (file.size > maxBytes) {
        reportError(`That file is larger than ${limitLabel}. Try a smaller one.`);
        return;
    }

    // If mic has a recording, clear it — file upload takes priority.
    if (audioBlob) resetRecorder();

    uploadedFile = file;
    _showFilePreview(file, isVideo);
}

function _showFilePreview(file, isVideo) {
    const dropzoneContent = document.getElementById('audio-dropzone-content');
    const previewEl = document.getElementById('audio-file-preview');
    const nameEl = document.getElementById('audio-file-name');
    const metaEl = document.getElementById('audio-file-meta');

    if (!previewEl) return;

    if (nameEl) nameEl.textContent = file.name;
    if (metaEl) {
        const typeLabel = isVideo ? '🎬 Video (audio will be extracted)' : '🎵 Audio';
        metaEl.textContent = `${typeLabel} · ${_formatBytes(file.size)}`;
    }

    if (dropzoneContent) dropzoneContent.classList.add('hidden');
    previewEl.classList.remove('hidden');
}

export function getUploadedFile() {
    return uploadedFile;
}

export function resetUploadedFile() {
    uploadedFile = null;
    const fileInput = document.getElementById('audio-file-input');
    if (fileInput) fileInput.value = '';

    const dropzoneContent = document.getElementById('audio-dropzone-content');
    const previewEl = document.getElementById('audio-file-preview');

    if (dropzoneContent) dropzoneContent.classList.remove('hidden');
    if (previewEl) previewEl.classList.add('hidden');
}
