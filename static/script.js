// --- STATE MANAGEMENT ---
const state = {
  domain: 'telecom'
};

// --- DOM ELEMENTS ---
const elements = {
  domainInput: document.getElementById('domain'),
  batchDomainInput: document.getElementById('batchDomain'),
  inputFields: document.getElementById('inputFields'),
  tabs: document.querySelectorAll('.tab-btn'),
  resetBtn: document.getElementById('resetBtn'),
  
  // Mode Switchers
  singleBtn: document.getElementById('singleModeBtn'),
  batchBtn: document.getElementById('batchModeBtn'),
  singleForm: document.getElementById('predictForm'),
  batchForm: document.getElementById('batchForm'),
  
  // Template Button
  templateBtn: document.getElementById('templateBtn')
};

// --- CORE FUNCTIONS ---

// 1. Switch Tabs
function switchDomain(newDomain) {
  state.domain = newDomain;
  if(elements.domainInput) elements.domainInput.value = newDomain;
  if(elements.batchDomainInput) elements.batchDomainInput.value = newDomain;
  
  // Update Template Download Link
  if(elements.templateBtn) {
    elements.templateBtn.href = `/api/template/${newDomain}`;
  }

  // Update Visuals
  elements.tabs.forEach(tab => {
    if (tab.dataset.domain === newDomain) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Fetch New Fields (only needed for single mode)
  if (elements.singleForm && elements.singleForm.style.display !== 'none') {
    renderFields();
  }
}

// 2. Fetch & Render Fields
async function renderFields() {
  const container = elements.inputFields;
  if (!container) return;

  container.innerHTML = Array(6).fill('<div class="loading-skeleton"></div>').join('');

  try {
    // NEW: Single API Call
    const res = await fetch(`/api/config/${state.domain}`);
    const data = await res.json();
    
    // data structure is now: { options: {...}, numeric: [...] }
    const options = data.options || {};
    const numeric = data.numeric || [];

    container.innerHTML = ''; 

    let delay = 0;
    const createField = (html) => {
      const div = document.createElement('div');
      div.className = 'form-group';
      div.style.animationDelay = `${delay}ms`;
      div.innerHTML = html;
      container.appendChild(div);
      delay += 30;
    };

    // Render Categorical (Dropdowns)
    Object.keys(options).forEach(feature => {
      let opts = options[feature].map(val => `<option value="${val}">${val}</option>`).join('');
      let html = `
        <label>${feature.replace(/_/g, ' ')}</label>
        <select name="${feature}" class="input-control" required>
          ${opts}
        </select>
      `;
      createField(html);
    });

    // Render Numeric (Inputs)
    numeric.forEach(feature => {
      let html = `
        <label>${feature.replace(/_/g, ' ')}</label>
        <input type="number" step="any" name="${feature}" class="input-control" placeholder="0.00" required>
      `;
      createField(html);
    });

  } catch (err) {
    console.error(err);
    container.innerHTML = `<div style="color:var(--danger)">Failed to load fields. Please ensure artifacts are generated.</div>`;
  }
}

// --- INITIALIZATION ---

document.addEventListener('DOMContentLoaded', () => {
  if(!elements.domainInput) return; 

  // Tab Listeners
  elements.tabs.forEach(tab => {
    tab.addEventListener('click', () => switchDomain(tab.dataset.domain));
  });

  // Reset Button
  if (elements.resetBtn) {
    elements.resetBtn.addEventListener('click', renderFields);
  }

  // Mode Switching Logic
  if (elements.singleBtn && elements.batchBtn) {
    elements.singleBtn.addEventListener('click', () => {
        elements.singleForm.style.display = 'block';
        elements.batchForm.style.display = 'none';
        
        elements.singleBtn.style.color = 'var(--accent-primary)';
        elements.singleBtn.style.borderBottom = '2px solid var(--accent-primary)';
        
        elements.batchBtn.style.color = 'var(--text-muted)';
        elements.batchBtn.style.borderBottom = 'none';
        
        renderFields(); 
    });

    elements.batchBtn.addEventListener('click', () => {
        elements.singleForm.style.display = 'none';
        elements.batchForm.style.display = 'block';
        
        elements.batchBtn.style.color = 'var(--accent-primary)';
        elements.batchBtn.style.borderBottom = '2px solid var(--accent-primary)';
        
        elements.singleBtn.style.color = 'var(--text-muted)';
        elements.singleBtn.style.borderBottom = 'none';
    });
  }

  // Initial Load
  const currentDomain = elements.domainInput.value || 'telecom';
  switchDomain(currentDomain);
});