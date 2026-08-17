let currentFile = null;

export function initUpload() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('image-input');
    const removeBtn = document.getElementById('remove-image');

    // Drag and Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.add('drag-active'), false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => dropzone.classList.remove('drag-active'), false);
    });

    dropzone.addEventListener('drop', handleDrop, false);
    
    // Click to upload
    dropzone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    // Clipboard paste
    document.addEventListener('paste', (e) => {
        const activeTabBtn = document.querySelector('.tab-btn.active');
        if (activeTabBtn && activeTabBtn.id === 'btn-image') {
            const items = (e.clipboardData || e.originalEvent.clipboardData).items;
            for (let item of items) {
                if (item.type.indexOf('image') === 0) {
                    const blob = item.getAsFile();
                    handleFile(blob);
                    break;
                }
            }
        }
    });

    // Remove image
    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files && files[0]) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
        alert('Invalid file type. Please upload a JPG, PNG, or WEBP image.');
        return;
    }

    if (file.size > 10 * 1024 * 1024) {
        alert('File is too large. Maximum size is 10MB.');
        return;
    }

    currentFile = file;

    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('image-preview').src = e.target.result;
        document.getElementById('dropzone-content').classList.add('hidden');
        document.getElementById('preview-container').classList.remove('hidden');
    }
    reader.readAsDataURL(file);
}

export function getImageFile() {
    return currentFile;
}

export function getCaption() {
    return document.getElementById('image-caption').value.trim();
}

export function resetUpload() {
    currentFile = null;
    document.getElementById('image-input').value = '';
    document.getElementById('image-caption').value = '';
    document.getElementById('image-preview').src = '';
    document.getElementById('preview-container').classList.add('hidden');
    document.getElementById('dropzone-content').classList.remove('hidden');
}
