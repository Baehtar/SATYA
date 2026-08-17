import { initUpload, getImageFile, resetUpload, getCaption } from './upload.js';
import { initRecorder, initAudioUpload, getAudioBlob, getAudioFilename, getUploadedFile, resetRecorder, resetUploadedFile } from './recorder.js';
import { submitCheck, streamProgress } from './api.js';
import { renderVerdict, resetVerdict, renderProgress, updateProgressStep, hideVerdict, showVerdict } from './verdict.js';

// Must match UI_STEPS / VALID_MODES in UI/src/server.py.
const STEPS_BY_MODE = {
    fake_news: [
        { id: 'upload', label: 'Uploading' },
        { id: 'analyze', label: 'Analysing content' },
        { id: 'search', label: 'Cross-referencing sources' },
        { id: 'verdict', label: 'Generating verdict' },
    ],
    // No OCR and no source search in this mode, so the stepper doesn't pretend.
    ai_image: [
        { id: 'upload', label: 'Uploading' },
        { id: 'analyze', label: 'Inspecting the image' },
        { id: 'verdict', label: 'Generating verdict' },
    ],
};

function init() {
    initUpload();
    initRecorder();
    initAudioUpload();

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const submitBtn = document.getElementById('submit-check');
    const textInput = document.getElementById('text-input');
    const progressSection = document.getElementById('progress-section');
    const checkAgainBtn = document.getElementById('check-again-btn');
    const errorBanner = document.getElementById('error-banner');
    const modeBtns = document.querySelectorAll('.mode-btn');
    const tabsRow = document.querySelector('.tabs');
    const captionInput = document.getElementById('image-caption');
    const aiImageHint = document.getElementById('ai-image-hint');

    let currentTab = 'image';
    let currentMode = 'fake_news';
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

    // ── Mode: fake news vs AI-generated image ────────────────────────────────
    modeBtns.forEach(btn => {
        btn.addEventListener('click', () => applyMode(btn.dataset.mode));
    });

    function applyMode(mode) {
        currentMode = mode;

        modeBtns.forEach(b => {
            const active = b.dataset.mode === mode;
            b.classList.toggle('active', active);
            b.setAttribute('aria-pressed', String(active));
        });

        const imageOnly = mode === 'ai_image';
        // AI-image mode takes an image and nothing else — hide what it ignores
        // instead of collecting input the backend will drop.
        tabsRow.classList.toggle('hidden', imageOnly);
        captionInput.classList.toggle('hidden', imageOnly);
        aiImageHint.classList.toggle('hidden', !imageOnly);
        submitBtn.textContent = imageOnly ? 'Check This Image →' : 'Check This →';

        if (imageOnly) selectTab('image');
        clearError();
    }

    function selectTab(target) {
        tabBtns.forEach(b => {
            const active = b.getAttribute('aria-controls') === `tab-${target}`;
            b.classList.toggle('active', active);
            b.setAttribute('aria-selected', String(active));
        });
        tabContents.forEach(content => {
            const active = content.id === `tab-${target}`;
            content.classList.toggle('hidden', !active);
            content.classList.toggle('active', active);
        });
        currentTab = target;
    }

    // ── Tabs ─────────────────────────────────────────────────────────────────
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            selectTab(btn.getAttribute('aria-controls').replace('tab-', ''));
            clearError();
        });
    });

    // ── Submit ───────────────────────────────────────────────────────────────
    submitBtn.addEventListener('click', () => {
        clearError();
        const formData = new FormData();
        formData.append('mode', currentMode);

        if (currentMode === 'ai_image') {
            const file = getImageFile();
            if (!file) {
                showError('Upload the image you want tested for AI generation.');
                return;
            }
            formData.append('image', file);
            startCheck(formData);
            return;
        }

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
            const uploadedFile = getUploadedFile();

            if (uploadedFile) {
                // Uploaded audio or video file — send on the right field.
                const isVideo = uploadedFile.type.startsWith('video/');
                if (isVideo) {
                    formData.append('video', uploadedFile, uploadedFile.name);
                } else {
                    formData.append('audio', uploadedFile, uploadedFile.name);
                }
            } else if (audio) {
                formData.append('audio', audio, getAudioFilename());
            } else {
                showError('Record a voice note or upload an audio / video file first.');
                return;
            }
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
        renderProgress(STEPS_BY_MODE[currentMode]);

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
        submitBtn.textContent = currentMode === 'ai_image' ? 'Check This Image →' : 'Check This →';
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
        resetUploadedFile();
        textInput.value = '';
        applyMode('fake_news');
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
