export async function submitCheck(formData) {
    try {
        const response = await fetch('/api/check', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error submitting check:', error);
        throw error;
    }
}

export function streamProgress(checkId, onProgress, onVerdict, onError) {
    const eventSource = new EventSource(`/api/check/${checkId}/stream`);

    let timeout = setTimeout(() => {
        eventSource.close();
        onError('Connection timed out');
    }, 60000); // 60s timeout

    eventSource.addEventListener('progress', (e) => {
        try {
            const data = JSON.parse(e.data);
            onProgress(data);
        } catch (err) {
            console.error('Error parsing progress:', err);
        }
    });

    eventSource.addEventListener('verdict', (e) => {
        clearTimeout(timeout);
        try {
            const data = JSON.parse(e.data);
            onVerdict(data);
            eventSource.close();
        } catch (err) {
            console.error('Error parsing verdict:', err);
            onError('Failed to parse verdict');
        }
    });

    eventSource.addEventListener('error', (e) => {
        clearTimeout(timeout);
        let msg = 'Stream error';
        try {
            if (e.data) {
                const data = JSON.parse(e.data);
                msg = data.error || msg;
            }
        } catch (err) {}
        onError(msg);
        eventSource.close();
    });

    eventSource.addEventListener('done', () => {
        clearTimeout(timeout);
        eventSource.close();
    });
}
