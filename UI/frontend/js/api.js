export async function submitCheck(formData) {
    const response = await fetch('/api/check', {
        method: 'POST',
        body: formData
    });

    let data = null;
    try {
        data = await response.json();
    } catch (err) {
        data = null;
    }

    if (!response.ok) {
        // The server explains refusals (too large, wrong type, nothing sent).
        throw new Error((data && data.error) || `Server error (${response.status})`);
    }

    return data;
}

export function streamProgress(checkId, onProgress, onVerdict, onError) {
    const eventSource = new EventSource(`/api/check/${checkId}/stream`);

    let settled = false;
    // The whole check is budgeted server-side; this only catches a dead socket.
    let timeout = setTimeout(() => finish(() => onError('The check timed out. Please try again.')), 90000);

    function finish(callback) {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        eventSource.close();
        if (callback) callback();
    }

    eventSource.addEventListener('progress', (e) => {
        try {
            onProgress(JSON.parse(e.data));
        } catch (err) {
            console.error('Bad progress payload:', err);
        }
    });

    eventSource.addEventListener('verdict', (e) => {
        let card;
        try {
            card = JSON.parse(e.data);
        } catch (err) {
            finish(() => onError('The verdict could not be read.'));
            return;
        }
        finish(() => onVerdict(card));
    });

    // Named server-side failure — distinct from the transport-level 'error'
    // event that EventSource fires on any disconnect, including a clean one.
    eventSource.addEventListener('failed', (e) => {
        let message = 'The check could not be completed.';
        try {
            message = JSON.parse(e.data).error || message;
        } catch (err) { /* keep the default */ }
        finish(() => onError(message));
    });

    eventSource.addEventListener('done', () => finish());

    eventSource.addEventListener('error', () => {
        // Fires when the stream closes after 'done' too — only surface it if
        // we never received a verdict.
        if (eventSource.readyState === EventSource.CLOSED) {
            finish(() => onError('Lost connection to the server.'));
        }
    });

    return () => finish();
}
