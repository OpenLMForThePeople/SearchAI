// --- GLOBAL STATE ---
let currentPath = "";
let activeSession = null;
let currentVideoIndex = 0;

// Creation Wizard State
let wizardStep = 1;
let wizardCriteria = [];

// --- LAYER 1: EXPLORER ---
async function loadExplorer(path = "") {
    try {
        currentPath = path;
        const response = await fetch(`/models?path=${encodeURIComponent(path)}`);
        const items = await response.json();
        
        const container = document.getElementById('explorer-container');
        container.innerHTML = ''; 

        if (currentPath !== "") {
            const backBtn = document.createElement('button');
            backBtn.className = 'explorer-btn';
            
            const backIcon = document.createElement('span');
            backIcon.className = 'material-icons';
            backIcon.textContent = 'arrow_back';
            
            const backLabel = document.createElement('span');
            backLabel.textContent = ' .. (Back)';
            
            backBtn.appendChild(backIcon);
            backBtn.appendChild(backLabel);
            
            backBtn.onclick = () => {
                const parts = currentPath.split(/[/\\]/).filter(Boolean);
                parts.pop(); 
                loadExplorer(parts.join('/'));
            };
            
            container.appendChild(backBtn);
        }
        
        items.forEach(item => {
            const btn = document.createElement('button');
            btn.className = `explorer-btn ${item.type}-btn`; 

            const icon = document.createElement('span');
            icon.className = 'material-icons';
            icon.textContent = item.type === 'folder' ? 'folder' : 'memory';
            
            const label = document.createElement('span');
            label.textContent = ` ${item.name}`;

            btn.appendChild(icon);
            btn.appendChild(label);
            
            btn.onclick = () => {
                if (item.type === 'folder') {
                    loadExplorer(item.relative_path);
                } else if (item.type === 'model') {
                    let fullModelPath = item.relative_path.replace(/\\/g, '/');
                    if (item.name.includes('.')) {
                        let parts = fullModelPath.split('/');
                        parts.pop(); 
                        fullModelPath = parts.join('/');
                    }
                    startTrainingSession(fullModelPath);
                }
            };
            
            container.appendChild(btn);
        });
    } catch (error) {
        console.error("Failed to load models folder:", error);
        document.getElementById('explorer-container').innerHTML = '<p>Error loading directory.</p>';
    }
}

function createFolder() {
    document.getElementById('folder-name').value = '';
    document.getElementById('folder-layer').style.display = 'flex';

    setTimeout(() => {
        document.getElementById('folder-name').focus();
    }, 0);
}

function closeFolderWizard() {
    document.getElementById('folder-layer').style.display = 'none';
}

async function submitFolder() {
    const input = document.getElementById('folder-name');
    const folderName = input.value.trim();

    if (!folderName) {
        input.focus();
        return;
    }

    try {
        const response = await fetch('/api/mkdir', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                name: folderName
            })
        });

        const res = await response.json();

        if (res.status === 'success') {
            closeFolderWizard();
            loadExplorer(currentPath);
        } else {
            // Keep the error INSIDE the SearchAI window
            input.focus();
            input.setCustomValidity(res.message || 'Failed to create folder.');
            input.reportValidity();

            setTimeout(() => {
                input.setCustomValidity('');
            }, 2500);
        }
    } catch (err) {
        console.error('Failed to create folder:', err);

        input.focus();
        input.setCustomValidity('Failed to connect to SearchAI server.');
        input.reportValidity();

        setTimeout(() => {
            input.setCustomValidity('');
        }, 2500);
    }
}
// --- LAYER 2C: CREATION WIZARD ---
function startCreationWizard() {
    wizardStep = 1;
    wizardCriteria = [];
    document.getElementById('model-name').value = '';
    document.getElementById('model-query').value = '';
    document.getElementById('criteria-input').value = '';
    document.getElementById('criteria-list').innerHTML = '';
    document.getElementById('creation-layer').style.display = 'flex';
    updateWizardUI();
}

function closeWizard() {
    document.getElementById('creation-layer').style.display = 'none';
}

function updateWizardUI() {
    for (let i = 1; i <= 5; i++) {
        const stepEl = document.getElementById(`step-${i}`);
        if (stepEl) {
            stepEl.classList.toggle('active', i === wizardStep);
        }
    }

    const prevBtn = document.getElementById('prev-btn');
    const nextBtn = document.getElementById('next-btn');
    const navBox = document.getElementById('wizard-nav');

    if (prevBtn) prevBtn.style.display = wizardStep === 1 ? 'none' : 'inline-flex';
    if (nextBtn) nextBtn.style.display = wizardStep === 5 ? 'none' : 'inline-flex';
    if (navBox) navBox.style.display = wizardStep === 5 ? 'none' : 'flex';

    if (wizardStep === 4) {
        document.getElementById('model-output-size').value = Math.max(1, wizardCriteria.length);
    }

    if (wizardStep === 5) {
        const name = document.getElementById('model-name').value;
        const engine = document.getElementById('model-engine').value;
        const query = document.getElementById('model-query').value;
        const hidden = document.getElementById('model-hidden-layers').value;
        
        document.getElementById('confirm-summary').innerHTML = `
            <strong>Name:</strong> ${name}<br>
            <strong>Engine:</strong> ${engine.toUpperCase()}<br>
            <strong>Query:</strong> ${query}<br>
            <strong>Criteria Count:</strong> ${wizardCriteria.length}<br>
            <strong>Hidden Layers:</strong> [ ${hidden} ]
        `;
    }
}

function nextStep() {
    if (wizardStep === 1 && !document.getElementById('model-name').value.trim()) {
        alert('Please enter a model name.');
        return;
    }
    if (wizardStep === 2 && !document.getElementById('model-query').value.trim()) {
        alert('Please enter a query.');
        return;
    }
    if (wizardStep === 3 && wizardCriteria.length === 0) {
        alert('Please add at least one criterion.');
        return;
    }

    if (wizardStep < 5) {
        wizardStep++;
        updateWizardUI();
    }
}

function prevStep() {
    if (wizardStep > 1) {
        wizardStep--;
        updateWizardUI();
    }
}

// Add Criteria Event Listener
document.addEventListener('DOMContentLoaded', () => {
    const critInput = document.getElementById('criteria-input');
    if (critInput) {
        critInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                const val = critInput.value.trim();
                if (val) {
                    wizardCriteria.push(val);
                    critInput.value = '';
                    renderCriteriaList();
                }
            }
        });
    }
});

function renderCriteriaList() {
    const list = document.getElementById('criteria-list');
    list.innerHTML = '';
    wizardCriteria.forEach((crit, idx) => {
        const li = document.createElement('li');
        li.textContent = `${idx + 1}. ${crit}`;
        list.appendChild(li);
    });
}

async function submitModel() {
    const hiddenStr = document.getElementById('model-hidden-layers').value.trim();
    const hiddenLayers = hiddenStr.split(/\s+/).map(Number).filter(n => !isNaN(n) && n > 0);

    const payload = {
        model_name: document.getElementById('model-name').value.trim(),
        engine: document.getElementById('model-engine').value,
        query: document.getElementById('model-query').value.trim(),
        criteria: wizardCriteria,
        input_size: parseInt(document.getElementById('model-input-size').value, 10),
        hidden_layers: hiddenLayers
    };

    try {
        const response = await fetch('/api/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const res = await response.json();
        if (res.status === 'success') {
            alert(`Model '${payload.model_name}' built successfully!`);
            closeWizard();
            loadExplorer(currentPath);
        } else {
            alert(`Error: ${res.message}`);
        }
    } catch (err) {
        console.error('Failed to create model:', err);
        alert('Failed to connect to backend.');
    }
}

// --- LAYER 2A: TRAINING SESSION WIZARD ---
async function startTrainingSession(modelPath) {
    console.log("[debug] Initializing training for path:", modelPath);
    
    const trainLayer = document.getElementById('training-layer');
    const trainTitle = document.getElementById('train-model-title');
    const trainLoading = document.getElementById('train-loading');
    const trainContent = document.getElementById('train-content');
    const trainReview = document.getElementById('train-review');

    const displayName = modelPath.split('/').pop();
    if (trainTitle) trainTitle.textContent = `Training: ${displayName}`;
    
    if (trainLayer) trainLayer.style.display = 'flex';
    if (trainLoading) trainLoading.style.display = 'block';
    if (trainContent) trainContent.style.display = 'none';
    if (trainReview) trainReview.style.display = 'none';

    try {
        const response = await fetch('/api/train/init', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelPath })
        });
        
        const res = await response.json();
        
        if (res.status === "success") {
            activeSession = res.data;
            currentVideoIndex = 0;
            
            if (trainLoading) trainLoading.style.display = 'none';
            if (trainContent) trainContent.style.display = 'block';
            
            renderCurrentVideo();
        } else {
            alert(`Error initializing session:\n\n${res.message}`);
            closeTrainingWizard();
        }
    } catch (err) {
        console.error("Communication error:", err);
        alert("Failed to communicate with SearchAI server.");
        closeTrainingWizard();
    }
}

function renderCurrentVideo() {
    if (!activeSession || !activeSession.items.length) return;

    const item = activeSession.items[currentVideoIndex];
    const total = activeSession.items.length;

    document.getElementById('video-counter').textContent = `Video ${currentVideoIndex + 1} of ${total}`;
    document.getElementById('video-title').textContent = item.video.title || "No Title";
    document.getElementById('video-channel').textContent = item.video.channel ? `Channel: ${item.video.channel}` : '';
    document.getElementById('video-url').href = item.video.url || '#';

    // Toggle Memory Auto-Fill Badge
    const memoryBadge = document.getElementById('memory-badge');
    if (memoryBadge) {
        // If scores are populated from history memory, display the badge
        if (item.is_memory) {
            memoryBadge.style.display = 'inline-block';
        } else {
            memoryBadge.style.display = 'none';
        }
    }

    // Build Criteria Scoring Inputs
    const container = document.getElementById('criteria-scores-container');
    container.innerHTML = '';

    activeSession.criteria.forEach((criterion, idx) => {
        const row = document.createElement('div');
        row.className = 'criterion-row';

        const label = document.createElement('span');
        label.className = 'criterion-label';
        label.textContent = criterion;

        const pred = document.createElement('span');
        pred.className = 'criterion-prediction';
        const pVal = item.predictions && item.predictions[idx] !== undefined ? item.predictions[idx].toFixed(2) : '0.50';
        pred.textContent = `Pred: ${pVal}`;

        const input = document.createElement('input');
        input.type = 'number';
        input.step = '0.1';
        input.min = '0.0';
        input.max = '1.0';
        input.className = 'criterion-score-input';
        
        // Load existing score (from memory or session)
        input.value = item.scores[idx] !== undefined ? item.scores[idx] : 0.5;

        input.onchange = (e) => {
            let val = parseFloat(e.target.value);
            if (isNaN(val)) val = 0.5;
            val = Math.max(0.0, Math.min(1.0, val));
            e.target.value = val;
            activeSession.items[currentVideoIndex].scores[idx] = val;
        };

        row.appendChild(label);
        row.appendChild(pred);
        row.appendChild(input);
        container.appendChild(row);
    });

    // Nav button states
    document.getElementById('prev-video-btn').style.display = currentVideoIndex === 0 ? 'none' : 'inline-flex';
    document.getElementById('next-video-btn').textContent = currentVideoIndex === total - 1 ? 'Review & Complete' : 'Next Video';
}

function nextVideo() {
    if (currentVideoIndex < activeSession.items.length - 1) {
        currentVideoIndex++;
        renderCurrentVideo();
    } else {
        document.getElementById('train-content').style.display = 'none';
        document.getElementById('train-review').style.display = 'block';
        document.getElementById('review-summary-text').innerHTML = 
            `Reviewed <strong>${activeSession.items.length}</strong> videos for <strong>${activeSession.model_name}</strong>. Ready to update agent weights?`;
    }
}

function prevVideo() {
    if (currentVideoIndex > 0) {
        currentVideoIndex--;
        renderCurrentVideo();
    }
}

function closeTrainingWizard() {
    document.getElementById('training-layer').style.display = 'none';
    activeSession = null;
    currentVideoIndex = 0;
}

// --- LAYER 3A: COMPLETE TRAINING ---
async function submitTrainingSession() {
    if (!activeSession) return;

    try {
        const payload = {
            model_name: activeSession.model_name,
            training_data: activeSession.items
        };

        const response = await fetch('/api/train/complete', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const res = await response.json();

        if (res.status === "success") {
            alert(`Training complete!\nRound: ${res.round_count}\nTrained Videos: ${res.trained_count}\nAvg Loss: ${res.average_loss.toFixed(6)}`);
            closeTrainingWizard();
            loadExplorer(currentPath);
        } else {
            alert(`Training failed: ${res.message}`);
        }
    } catch (err) {
        console.error("Error submitting training:", err);
        alert("Failed to send training payload to server.");
    }
}

// --- INITIALIZATION ---
window.onload = () => {
    loadExplorer();
};