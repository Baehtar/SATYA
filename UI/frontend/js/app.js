import { initUpload, getImageFile, resetUpload, getCaption } from './upload.js';
import { initRecorder, getAudioBlob, resetRecorder } from './recorder.js';
import { submitCheck, streamProgress } from './api.js';
import { renderVerdict, resetVerdict, renderProgress, updateProgressStep, hideVerdict, showVerdict } from './verdict.js';

function init() {
    // Initialize modules
    initUpload();
    initRecorder();

    // DOM Elements
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    const submitBtn = document.getElementById('submit-check');
    const textInput = document.getElementById('text-input');
    const progressSection = document.getElementById('progress-section');
    const checkAgainBtn = document.getElementById('check-again-btn');

    let currentTab = 'image';

    // Tab Switching
    tabBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const target = btn.getAttribute('aria-controls').replace('tab-', '');
            
            // Update buttons
            tabBtns.forEach(b => {
                b.classList.remove('active');
                b.setAttribute('aria-selected', 'false');
            });
            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');

            // Update content
            tabContents.forEach(content => {
                content.classList.add('hidden');
                content.classList.remove('active');
            });
            
            const targetContent = document.getElementById(`tab-${target}`);
            if (targetContent) {
                targetContent.classList.remove('hidden');
                requestAnimationFrame(() => {
                    targetContent.classList.add('active');
                });
            }

            currentTab = target;
        });
    });

    // Form Submission
    submitBtn.addEventListener('click', async () => {
        let formData = new FormData();
        let hasData = false;

        if (currentTab === 'image') {
            const file = getImageFile();
            const caption = getCaption();
            if (file) {
                formData.append('image', file);
            }
            if (caption) {
                formData.append('text', caption);
            }
            if (file || caption) {
                hasData = true;
            }
        } else if (currentTab === 'text') {
            const text = textInput.value.trim();
            if (text) {
                formData.append('text', text);
                hasData = true;
            }
        } else if (currentTab === 'voice') {
            const audio = getAudioBlob();
            if (audio) {
                formData.append('audio', audio, 'recording.webm');
                hasData = true;
            }
        }

        if (!hasData) {
            alert('Please provide an image, message text, or voice recording to check.');
            return;
        }

        startCheck(formData);
    });

    checkAgainBtn.addEventListener('click', resetAll);

    async function startCheck(formData) {
        // UI State -> checking
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span style="display:inline-block;animation:spin 1s linear infinite;">⏳</span> Checking...';
        
        progressSection.classList.remove('hidden');
        progressSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        hideVerdict();
        resetVerdict();

        // Initialize empty steps
        renderProgress([
            { id: 'upload', label: 'Uploading data' },
            { id: 'analyze', label: 'Analyzing content' },
            { id: 'search', label: 'Cross-referencing sources' },
            { id: 'verdict', label: 'Generating verdict' }
        ]);

        try {
            updateProgressStep('upload', 'running', 'Uploading data...');
            const { id } = await submitCheck(formData);
            updateProgressStep('upload', 'done', 'Upload complete');

            // Start SSE stream
            streamProgress(id, 
                (progressEvent) => {
                    // onProgress
                    updateProgressStep(progressEvent.step, progressEvent.status, progressEvent.message);
                },
                (verdictData) => {
                    // onVerdict
                    renderVerdict(verdictData);
                    showVerdict();
                    submitBtn.disabled = false;
                    submitBtn.textContent = 'Check This \u2192';
                },
                (errorMsg) => {
                    // onError
                    alert('Error: ' + errorMsg);
                    resetCheckingState();
                }
            );

        } catch (error) {
            alert('Failed to start check: ' + error.message);
            resetCheckingState();
        }
    }

    function resetCheckingState() {
        submitBtn.disabled = false;
        submitBtn.textContent = 'Check This \u2192';
        progressSection.classList.add('hidden');
    }

    function resetAll() {
        resetCheckingState();
        resetUpload();
        resetRecorder();
        textInput.value = '';
        hideVerdict();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
