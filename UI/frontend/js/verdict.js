// Verdict slugs come from UI/src/adapter.py.
const VERDICT_DISPLAY = {
    likely_true:        { emoji: '🟢', label: 'Likely True',        cls: 'true' },
    authentic_image:    { emoji: '🟢', label: 'Not AI-Generated',    cls: 'true' },
    likely_false:       { emoji: '🔴', label: 'Likely False',       cls: 'false' },
    ai_generated:       { emoji: '🤖', label: 'AI Generated',       cls: 'ai' },
    manipulated:        { emoji: '✂️', label: 'Manipulated',        cls: 'ai' },
    misleading_context: { emoji: '🟠', label: 'Misleading Context', cls: 'ai' },
    unverifiable:       { emoji: '🟡', label: 'Unverifiable',       cls: 'unverifiable' },
};

const FLAG_LABELS = {
    AI_GENERATED: 'AI-generated image detected',
    MANIPULATED: 'Image manipulation detected',
    RECYCLED: 'Recycled / old image',
};

// Backend statuses → CSS classes on .step
const STATUS_CLASS = {
    pending: 'pending',
    running: 'running',
    started: 'running',
    completed: 'done',
    done: 'done',
    skipped: 'skipped',
    error: 'error',
};

export function renderProgress(steps) {
    const stepper = document.getElementById('stepper');
    stepper.innerHTML = '';

    steps.forEach((step, index) => {
        const stepEl = document.createElement('div');
        stepEl.className = 'step pending';
        stepEl.id = `step-${step.id}`;
        stepEl.style.animationDelay = `${index * 100}ms`;
        stepEl.dataset.label = step.label;

        const icon = document.createElement('div');
        icon.className = 'step-icon';
        const label = document.createElement('div');
        label.className = 'step-label';
        label.textContent = step.label;

        stepEl.append(icon, label);
        stepper.appendChild(stepEl);
    });
}

export function updateProgressStep(stepId, status, message) {
    const stepEl = document.getElementById(`step-${stepId}`);
    if (!stepEl) return;

    const statusClass = STATUS_CLASS[status] || 'running';
    stepEl.className = `step ${statusClass}`;

    const iconEl = stepEl.querySelector('.step-icon');
    if (statusClass === 'done') {
        iconEl.textContent = '✓';
    } else if (statusClass === 'error') {
        iconEl.textContent = '✕';
    } else if (statusClass === 'skipped') {
        iconEl.textContent = '–';
    } else {
        iconEl.textContent = '';
    }

    // Keep the step's own name when a stage reports no message of its own.
    stepEl.querySelector('.step-label').textContent = message || stepEl.dataset.label;
}

export function hideVerdict() {
    const section = document.getElementById('verdict-section');
    section.classList.add('hidden');
    section.querySelector('.verdict-card').classList.remove('show');
}

export function showVerdict() {
    const section = document.getElementById('verdict-section');
    section.classList.remove('hidden');

    requestAnimationFrame(() => {
        section.querySelector('.verdict-card').classList.add('show');
        setTimeout(() => {
            section.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 100);
    });
}

export function resetVerdict() {
    const fill = document.getElementById('confidence-fill');
    fill.style.width = '0%';
    fill.className = 'confidence-fill';

    const flags = document.getElementById('image-flags');
    flags.innerHTML = '';
    flags.classList.add('hidden');

    document.getElementById('claim-box').classList.add('hidden');
    document.getElementById('sources-section').classList.remove('hidden');
    document.getElementById('sources-list').innerHTML = '';
    document.getElementById('verdict-meta').textContent = '';
}

export function renderVerdict(data) {
    const display = VERDICT_DISPLAY[data.verdict] || VERDICT_DISPLAY.unverifiable;

    // ── Badge ────────────────────────────────────────────────────────────────
    const badge = document.getElementById('verdict-badge');
    badge.className = `verdict-badge ${display.cls}`;
    badge.textContent = `${display.emoji} ${display.label}`;

    // ── Confidence (backend sends 0.0–1.0) ───────────────────────────────────
    const pct = Math.round((data.confidence || 0) * 100);
    const level = data.confidence_level ? ` (${data.confidence_level.toLowerCase()})` : '';
    document.getElementById('confidence-label').textContent = `Confidence: ${pct}%${level}`;

    const fill = document.getElementById('confidence-fill');
    fill.className = `confidence-fill ${display.cls}`;
    setTimeout(() => { fill.style.width = `${pct}%`; }, 100);

    // ── What was actually checked (claim / OCR text / transcript) ────────────
    const claimBox = document.getElementById('claim-box');
    if (data.claim) {
        document.getElementById('claim-label').textContent = data.claim_label || 'Claim checked';
        document.getElementById('claim-text').textContent = data.claim;
        claimBox.classList.remove('hidden');
    } else {
        claimBox.classList.add('hidden');
    }

    // ── Explanations ─────────────────────────────────────────────────────────
    document.getElementById('explanation-en').textContent = data.explanation_en || '';
    document.getElementById('explanation-hi').textContent = data.explanation_hi || '';

    // ── Sources ──────────────────────────────────────────────────────────────
    // A dedicated AI-image check has no claim and no sources — hide the section
    // rather than showing an empty "no fact-checks matched" note.
    const aiImageMode = (data.meta || {}).mode === 'ai_image';
    document.getElementById('sources-section').classList.toggle('hidden', aiImageMode);

    const sourcesList = document.getElementById('sources-list');
    sourcesList.innerHTML = '';
    if (data.sources && data.sources.length > 0) {
        data.sources.forEach(source => {
            const a = document.createElement('a');
            a.href = source.source_url;
            a.className = 'source-pill';
            a.target = '_blank';
            a.rel = 'noopener noreferrer';
            a.textContent = source.source_name;
            if (source.snippet) a.title = source.snippet;
            sourcesList.appendChild(a);
        });
    } else {
        const empty = document.createElement('span');
        empty.className = 'sources-empty';
        empty.textContent = 'No fact-check or news coverage matched this claim.';
        sourcesList.appendChild(empty);
    }

    // ── Image flags ──────────────────────────────────────────────────────────
    const flagsContainer = document.getElementById('image-flags');
    flagsContainer.innerHTML = '';
    if (aiImageMode) {
        flagsContainer.classList.add('hidden');   // the verdict already says it
    } else if (data.image_flags && data.image_flags.length > 0) {
        flagsContainer.classList.remove('hidden');
        data.image_flags.forEach(flag => {
            const span = document.createElement('span');
            span.className = 'flag-tag';
            span.textContent = `⚠️ ${FLAG_LABELS[flag] || flag}`;
            flagsContainer.appendChild(span);
        });
    } else {
        flagsContainer.classList.add('hidden');
    }

    // ── Footer ───────────────────────────────────────────────────────────────
    document.getElementById('disclaimer-text').textContent = data.disclaimer ||
        'This assessment is generated by AI and may contain errors. Always verify critical information from official sources.';

    const meta = data.meta || {};
    const bits = [];
    if (meta.latency_ms) bits.push(`checked in ${(meta.latency_ms / 1000).toFixed(1)}s`);
    if (meta.language) bits.push(meta.language.toLowerCase().replace('_', ' '));
    if (meta.type === 'image' || meta.type === 'mixed') {
        bits.push(`AI-image score ${Math.round((meta.image_ai_score || 0) * 100)}%`);
    }
    bits.push(meta.mode === 'ai_image' ? 'AI-image detection' : 'fake-news check');
    document.getElementById('verdict-meta').textContent = bits.join(' · ');
}
