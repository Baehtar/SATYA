let mediaRecorder;
let audioChunks = [];
let audioBlob = null;
let recordingTimer;
let startTime;
let audioContext;
let analyser;
let dataArray;
let animationId;

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
        if (audioBlob) {
            const audio = new Audio(URL.createObjectURL(audioBlob));
            audio.play();
        }
    });

    rerecordBtn.addEventListener('click', resetRecorder);
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        setupVisualizer(stream);

        mediaRecorder.addEventListener('dataavailable', event => {
            audioChunks.push(event.data);
        });

        mediaRecorder.addEventListener('stop', () => {
            audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            
            // Stop tracks
            stream.getTracks().forEach(track => track.stop());
            if (audioContext) audioContext.close();
            cancelAnimationFrame(animationId);

            document.getElementById('record-btn').classList.remove('recording');
            document.getElementById('record-btn').style.display = 'none';
            document.getElementById('audio-controls').classList.remove('hidden');
            
            const duration = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('record-status').textContent = `Recording saved (${formatTime(duration)})`;
        });

        mediaRecorder.start();
        startTime = Date.now();
        document.getElementById('record-btn').classList.add('recording');
        
        recordingTimer = setInterval(() => {
            const duration = Math.floor((Date.now() - startTime) / 1000);
            document.getElementById('record-status').textContent = `Recording... (${formatTime(duration)})`;
            
            if (duration >= 120) {
                stopRecording();
            }
        }, 1000);

    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Could not access microphone. Please check permissions.');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        clearInterval(recordingTimer);
        
        const canvas = document.getElementById('waveform-canvas');
        const canvasCtx = canvas.getContext('2d');
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    }
}

function setupVisualizer(stream) {
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);

    const canvas = document.getElementById('waveform-canvas');
    const canvasCtx = canvas.getContext('2d');

    function draw() {
        animationId = requestAnimationFrame(draw);

        analyser.getByteFrequencyData(dataArray);

        canvasCtx.fillStyle = 'var(--color-surface)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

        const barWidth = (canvas.width / bufferLength) * 2.5;
        let barHeight;
        let x = 0;

        for (let i = 0; i < bufferLength; i++) {
            barHeight = dataArray[i] / 2;

            canvasCtx.fillStyle = 'var(--color-accent)';
            canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);

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

export function resetRecorder() {
    audioBlob = null;
    audioChunks = [];
    clearInterval(recordingTimer);
    
    document.getElementById('record-btn').style.display = 'flex';
    document.getElementById('record-btn').classList.remove('recording');
    document.getElementById('audio-controls').classList.add('hidden');
    document.getElementById('record-status').textContent = 'Click to start recording';
    
    const canvas = document.getElementById('waveform-canvas');
    const canvasCtx = canvas.getContext('2d');
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
}
