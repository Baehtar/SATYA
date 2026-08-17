import { initUpload, getImageFile, resetUpload, getCaption } from './upload.js';
import { initRecorder, getAudioBlob, getAudioFilename, resetRecorder } from './recorder.js';
import { submitCheck, streamProgress } from './api.js';
import { renderVerdict, resetVerdict, renderProgress, updateProgressStep, hideVerdict, showVerdict } from './verdict.js';

// Must match UI_STEPS in UI/src/server.py.
const STEPS = [
    { id: 'upload', label: 'Uploading' },
    { id: 'analyze', label: 'Analysing content' },
    { id: 'search', label: 'Cross-referencing sources' },
    { id: 'verdict', label: 'Generating verdict' },
];

function init() {
    initUpload();
    initRecorder();

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const submitBtn = document.getElementById('submit-check');
    const textInput = document.getElementById('text-input');
    const progressSection = document.getElementById('progress-section');
    const checkAgainBtn = document.getElementById('check-again-btn');
    const errorBanner = document.getElementById('error-banner');

    let currentTab = 'image';
    let cancelStream = null;

    function showError(message) {
        errorBanner.textContent = `⚠️ ${message}`;
        errorBanner.classList.remove('hidden');
    }

    function clearError() {
        errorBanner.textContent = '';
        errorBanner.classList.add('hidden');
    }

    // Errors raised by the upload/recorder modules.
    document.addEventListener('satya:error', (e) => showError(e.detail));

    // ── Tabs ─────────────────────────────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const target = btn.getAttribute('aria-controls').replace('tab-', '');

            tabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');

            tabContents.forEach(content => {
                content.classList.add('hidden');
                content.classList.remove('active');
            });

            const targetContent = document.getElementById(`tab-${target}`);
            if (targetContent) {
                targetContent.classList.remove('hidden');
                requestAnimationFrame(() => targetContent.classList.add('active'));
            }

            currentTab = target;
            clearError();
        });
    });

    // ── Submit ───────────────────────────────────────────────────────────────
    submitBtn.addEventListener('click', () => {
        clearError();
        const formData = new FormData();

        if (currentTab === 'image') {
            const file = getImageFile();
            const caption = getCaption();
            if (!file && !caption) {
                showError('Add an image, or a caption to check on its own.');
                return;
            }
            if (file) formData.append('image', file);
            if (caption) formData.append('text', caption);
        } else if (currentTab === 'text') {
            const text = textInput.value.trim();
            if (!text) {
                showError('Paste the message you want checked.');
                return;
            }
            formData.append('text', text);
        } else if (currentTab === 'voice') {
            const audio = getAudioBlob();
            if (!audio) {
                showError('Record a voice note first.');
                return;
            }
            formData.append('audio', audio, getAudioFilename());
        }

        startCheck(formData);
    });

    checkAgainBtn.addEventListener('click', resetAll);

    async function startCheck(formData) {
        submitBtn.disabled = true;
        submitBtn.textContent = 'Checking…';

        progressSection.classList.remove('hidden');
        progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        hideVerdict();
        resetVerdict();
        renderProgress(STEPS);

        try {
            updateProgressStep('upload', 'running', 'Uploading…');
            const { id } = await submitCheck(formData);
            updateProgressStep('upload', 'completed', 'Uploaded');

            cancelStream = streamProgress(
                id,
                ({ step, status, message }) => updateProgressStep(step, status, message),
                (card) => {
                    renderVerdict(card);
                    showVerdict();
                    resetCheckingState({ keepProgress: true });
                },
                (message) => {
                    updateProgressStep('verdict', 'error', 'Check failed');
                    showError(message);
                    resetCheckingState({ keepProgress: true });
                }
            );
        } catch (error) {
            updateProgressStep('upload', 'error', 'Upload failed');
            showError(error.message);
            resetCheckingState({ keepProgress: true });
        }
    }

    function resetCheckingState({ keepProgress = false } = {}) {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Check This →';
        if (!keepProgress) progressSection.classList.add('hidden');
    }

    function resetAll() {
        if (cancelStream) {
            cancelStream();
            cancelStream = null;
        }
        resetCheckingState();
        resetUpload();
        resetRecorder();
        textInput.value = '';
        clearError();
        hideVerdict();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
