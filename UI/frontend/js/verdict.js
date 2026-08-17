export function renderProgress(steps) {
    const stepper = document.getElementById('stepper');
    stepper.innerHTML = '';

    steps.forEach((step, index) => {
        const stepEl = document.createElement('div');
        stepEl.className = 'step pending';
        stepEl.id = `step-${step.id}`;
        stepEl.style.animationDelay = `${index * 100}ms`;
        
        stepEl.innerHTML = `
            <div class="step-icon"></div>
            <div class="step-label">${step.label}</div>
        `;
        stepper.appendChild(stepEl);
    });
}

export function updateProgressStep(stepId, status, message) {
    // Map backend step IDs to frontend step IDs
    const stepMap = {
        'image_analysis': 'analyze',
        'text_analysis': 'analyze',
        'audio_analysis': 'analyze',
        'reverse_search': 'search',
        'fact_check': 'search',
        'generating_verdict': 'verdict',
    };
    const mappedId = stepMap[stepId] || stepId;
    
    // Map backend statuses to CSS classes
    const statusMap = {
        'started': 'running',
        'running': 'running',
        'completed': 'done',
        'done': 'done',
        'error': 'error',
        'pending': 'pending',
    };
    const mappedStatus = statusMap[status] || status;

    const stepEl = document.getElementById(`step-${mappedId}`);
    if (!stepEl) return;

    stepEl.className = `step ${mappedStatus}`;
    
    const iconEl = stepEl.querySelector('.step-icon');
    if (mappedStatus === 'done') {
        iconEl.innerHTML = '✓';
    } else if (mappedStatus === 'error') {
        iconEl.innerHTML = '✕';
    } else {
        iconEl.innerHTML = '';
    }

    if (message) {
        stepEl.querySelector('.step-label').textContent = message;
    }
}

export function hideVerdict() {
    const section = document.getElementById('verdict-section');
    section.classList.add('hidden');
    const card = section.querySelector('.verdict-card');
    card.classList.remove('show');
}

export function showVerdict() {
    const section = document.getElementById('verdict-section');
    section.classList.remove('hidden');
    
    // Trigger animation after render
    requestAnimationFrame(() => {
        const card = section.querySelector('.verdict-card');
        card.classList.add('show');
        
        // Scroll to verdict smoothly
        setTimeout(() => {
            section.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    });
}

export function resetVerdict() {
    const fill = document.getElementById('confidence-fill');
    fill.style.width = '0%';
    fill.className = 'confidence-fill';
    
    document.getElementById('image-flags').innerHTML = '';
    document.getElementById('image-flags').classList.add('hidden');
    
    document.getElementById('sources-list').innerHTML = '';
}

export function renderVerdict(data) {
    const { verdict, confidence, confidence_level, explanation_en, explanation_hi, sources, image_flags, disclaimer } = data;
    
    // Verdict Badge — map backend enum values to display
    const badge = document.getElementById('verdict-badge');
    badge.className = 'verdict-badge'; // reset
    
    let emoji = '🟡';
    let label = 'UNVERIFIABLE';
    let statusClass = 'unverifiable';
    
    if (verdict === 'likely_true') {
        emoji = '🟢';
        label = 'LIKELY TRUE';
        statusClass = 'true';
    } else if (verdict === 'likely_false') {
        emoji = '🔴';
        label = 'LIKELY FALSE';
        statusClass = 'false';
    }
    
    badge.classList.add(statusClass);
    badge.innerHTML = `${emoji} ${label}`;
    
    // Confidence Bar — backend sends 0.0-1.0, convert to percentage
    const confidencePct = Math.round(confidence * 100);
    document.getElementById('confidence-label').textContent = `Confidence: ${confidencePct}%`;
    const fill = document.getElementById('confidence-fill');
    fill.className = `confidence-fill ${statusClass}`;
    
    // Delay width animation slightly for better effect
    setTimeout(() => {
        fill.style.width = `${confidencePct}%`;
    }, 100);
    
    // Explanations
    document.getElementById('explanation-en').textContent = explanation_en;
    document.getElementById('explanation-hi').textContent = explanation_hi;
    
    // Sources — backend sends FactCheckMatch objects with source_name, source_url
    const sourcesList = document.getElementById('sources-list');
    sourcesList.innerHTML = '';
    if (sources && sources.length > 0) {
        sources.forEach(source => {
            const a = document.createElement('a');
            a.href = source.source_url;
            a.className = 'source-pill';
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.textContent = source.source_name;
            sourcesList.appendChild(a);
        });
    } else {
        sourcesList.innerHTML = '<span style="color: var(--color-text-muted); font-size: 13px;">No direct sources found.</span>';
    }
    
    // Image Flags
    const flagsContainer = document.getElementById('image-flags');
    flagsContainer.innerHTML = '';
    if (image_flags && image_flags.length > 0) {
        flagsContainer.classList.remove('hidden');
        image_flags.forEach(flag => {
            const span = document.createElement('span');
            span.className = 'flag-tag';
            // Map technical flags to user-friendly labels
            const flagLabels = {
                'AI_GENERATED': 'AI-Generated Image Detected',
                'MANIPULATED': 'Image Manipulation Detected',
                'RECYCLED': 'Recycled / Old Image',
            };
            span.innerHTML = `⚠️ ${flagLabels[flag] || flag}`;
            flagsContainer.appendChild(span);
        });
    } else {
        flagsContainer.classList.add('hidden');
    }
    
    // Disclaimer
    document.getElementById('disclaimer-text').textContent = disclaimer || 'This assessment is generated by AI and may contain errors. Always verify critical information from official sources.';
}
