// Error Detection Tool - Review Interface

class ReviewInterface {
  constructor() {
    this.layers = [];
    this.filteredLayers = [];
    this.currentLayer = null;
    this.filters = {
      status: 'all',
      severity: 'all'
    };
  }

  async init() {
    await this.loadLayers();
    this.setupEventListeners();
    this.renderLayers();
  }

  async loadLayers() {
    try {
      // Load all layers (correct, incorrect, and unsure)
      const [correctResponse, incorrectResponse, unsureResponse] = await Promise.all([
        Utils.fetchJSON('/api/layers_by_status/correct'),
        Utils.fetchJSON('/api/layers_by_status/incorrect'),
        Utils.fetchJSON('/api/layers_by_status/unsure')
      ]);

      this.layers = [...correctResponse, ...incorrectResponse, ...unsureResponse];
      this.filteredLayers = [...this.layers];
    } catch (error) {
      console.error('Error loading layers:', error);
      console.log('Error loading layers for review', 'error');
    }
  }


  setupEventListeners() {
    // Filter controls
    document.getElementById('status-filter')?.addEventListener('change', (e) => {
      this.filters.status = e.target.value;
    });


    document.getElementById('apply-filters')?.addEventListener('click', () => {
      this.applyFilters();
    });


    // Navigation buttons
    document.getElementById('back-to-detection')?.addEventListener('click', () => {
      window.location.href = '/detection';
    });

    document.getElementById('proceed-to-export')?.addEventListener('click', () => {
      this.proceedToExport();
    });

    // Modal interactions
    this.setupModalEventListeners();
  }

  setupModalEventListeners() {
    const layerDetailModal = document.getElementById('layer-detail-modal');

    // Tab switching
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        this.switchTab(e.target.dataset.tab);
      });
    });

    // Close modals
    document.querySelectorAll('.modal-close').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const modal = e.target.closest('.modal');
        Utils.hideModal(modal.id);
      });
    });

    // Detail annotation form
    document.getElementById('save-detail-annotation')?.addEventListener('click', () => {
      this.saveDetailAnnotation();
    });
  }

  async applyFilters() {
    try {
      const response = await Utils.postJSON('/api/filter_layers', {
        status: this.filters.status,
        severity: this.filters.severity
      });

      this.filteredLayers = response;
      this.renderLayers();
    } catch (error) {
      console.error('Error applying filters:', error);
      console.log('Error applying filters', 'error');
    }
  }

  renderLayers() {
    const grid = document.getElementById('review-layers-grid');
    if (!grid) return;

    grid.innerHTML = '';

    if (this.filteredLayers.length === 0) {
      grid.innerHTML = '<div class="no-layers">No layers match the current filters.</div>';
      return;
    }

    this.filteredLayers.forEach(layer => {
      const layerCard = this.createLayerCard(layer);
      grid.appendChild(layerCard);
    });
  }

  createLayerCard(layer) {
    const card = document.createElement('div');
    card.className = 'layer-card';
    card.dataset.layerId = layer.id;

    const status = layer.status || 'unlabeled';
    const annotation = layer.annotation || {};

    card.innerHTML = `
      <div class="layer-header">
        <h4>Layer ${layer.id} (Z: ${layer.z})</h4>
        <div class="layer-status">
          <span class="status-indicator ${status}"></span>
          <span class="status-text">${status.charAt(0).toUpperCase() + status.slice(1)}</span>
        </div>
      </div>
      
      <div class="layer-content">
        <div class="layer-image">
          <img src="data:image/png;base64,${layer.overlay}" 
               alt="Layer ${layer.id} overlay" 
               class="overlay-image">
        </div>
        
        <div class="layer-info">
          <div class="info-item">
            <span class="info-label">Has Mask:</span>
            <span class="info-value">${layer.has_mask ? 'Yes' : 'No'}</span>
          </div>
          ${layer.has_mask ? `
          <div class="info-item">
            <span class="info-label">Coverage:</span>
            <span class="info-value">${(layer.mask_coverage * 100).toFixed(1)}%</span>
          </div>
          ` : ''}
          ${annotation.error_type ? `
          <div class="info-item">
            <span class="info-label">Error Type:</span>
            <span class="info-value">${annotation.error_type}</span>
          </div>
          ` : ''}
          ${annotation.severity ? `
          <div class="info-item">
            <span class="info-label">Severity:</span>
            <span class="info-value">${annotation.severity}</span>
          </div>
          ` : ''}
        </div>
        
        <div class="layer-actions">
          <button class="btn btn-primary btn-sm" onclick="reviewInterface.showLayerDetail('${layer.id}')">
            View Details
          </button>
          <button class="btn btn-secondary btn-sm" onclick="reviewInterface.editAnnotation('${layer.id}')">
            Edit Annotation
          </button>
        </div>
      </div>
    `;

    return card;
  }

  async showLayerDetail(layerId) {
    try {
      const response = await Utils.fetchJSON(`/api/layer_details/${layerId}`);
      this.currentLayer = layerId;
      
      // Populate modal with layer data
      this.populateLayerDetailModal(response);
      Utils.showModal('layer-detail-modal');
    } catch (error) {
      console.error('Error loading layer details:', error);
      console.log('Error loading layer details', 'error');
    }
  }

  populateLayerDetailModal(data) {
    const { layer, analysis } = data;

    // Overview tab
    const detailImage = document.getElementById('detail-layer-image');
    if (detailImage && layer.overlay) {
      detailImage.src = `data:image/png;base64,${layer.overlay}`;
    }

    const layerInfo = document.getElementById('detail-layer-info');
    if (layerInfo) {
      layerInfo.innerHTML = `
        <div class="info-item">
          <span class="info-label">Layer ID:</span>
          <span class="info-value">${layer.id}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Z Index:</span>
          <span class="info-value">${layer.z}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Has Mask:</span>
          <span class="info-value">${layer.has_mask ? 'Yes' : 'No'}</span>
        </div>
        <div class="info-item">
          <span class="info-label">Mask Coverage:</span>
          <span class="info-value">${(layer.mask_coverage * 100).toFixed(1)}%</span>
        </div>
        <div class="info-item">
          <span class="info-label">Status:</span>
          <span class="info-value">${layer.status || 'unlabeled'}</span>
        </div>
      `;
    }

    // Analysis tab
    const errorAnalysis = document.getElementById('detail-error-analysis');
    if (errorAnalysis && analysis) {
      errorAnalysis.innerHTML = `
        <div class="analysis-item">
          <span class="analysis-label">Confidence:</span>
          <span class="analysis-value">${(analysis.confidence * 100).toFixed(1)}%</span>
        </div>
        <div class="analysis-item">
          <span class="analysis-label">Potential Issues:</span>
          <span class="analysis-value">${analysis.potential_issues.join(', ') || 'None detected'}</span>
        </div>
        <div class="analysis-item">
          <span class="analysis-label">Analysis Time:</span>
          <span class="analysis-value">${new Date(analysis.analysis_timestamp).toLocaleString()}</span>
        </div>
      `;
    }

    // Annotation tab
    this.populateAnnotationTab(layer);
  }

  populateAnnotationTab(layer) {
    const annotation = layer.annotation || {};
    
    // Current annotation display
    const currentAnnotation = document.getElementById('detail-current-annotation');
    if (currentAnnotation) {
      currentAnnotation.innerHTML = `
        <div class="annotation-display">
          <div class="annotation-item">
            <span class="annotation-label">Status:</span>
            <span class="annotation-value">${annotation.status || 'unlabeled'}</span>
          </div>
          ${annotation.error_type ? `
          <div class="annotation-item">
            <span class="annotation-label">Error Type:</span>
            <span class="annotation-value">${annotation.error_type}</span>
          </div>
          ` : ''}
          ${annotation.severity ? `
          <div class="annotation-item">
            <span class="annotation-label">Severity:</span>
            <span class="annotation-value">${annotation.severity}</span>
          </div>
          ` : ''}
          ${annotation.description ? `
          <div class="annotation-item">
            <span class="annotation-label">Description:</span>
            <span class="annotation-value">${annotation.description}</span>
          </div>
          ` : ''}
          ${annotation.notes ? `
          <div class="annotation-item">
            <span class="annotation-label">Notes:</span>
            <span class="annotation-value">${annotation.notes}</span>
          </div>
          ` : ''}
        </div>
      `;
    }

    // Form fields
    const form = document.getElementById('detail-annotation-form');
    if (form) {
      form.status.value = annotation.status || 'unlabeled';
      form.error_type.value = annotation.error_type || '';
      form.severity.value = annotation.severity || 'medium';
      form.description.value = annotation.description || '';
      form.confidence.value = annotation.confidence || 0.5;
      form.notes.value = annotation.notes || '';
    }
  }

  switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
      btn.classList.remove('active');
    });
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-pane').forEach(pane => {
      pane.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
  }

  async saveDetailAnnotation() {
    if (!this.currentLayer) {
      console.log('No layer selected', 'error');
      return;
    }

    const form = document.getElementById('detail-annotation-form');
    const formData = new FormData(form);
    
    const annotation = {
      status: formData.get('status'),
      error_type: formData.get('error_type') || null,
      severity: formData.get('severity') || null,
      description: formData.get('description') || '',
      confidence: parseFloat(formData.get('confidence')) || 0.5,
      notes: formData.get('notes') || ''
    };

    try {
      const response = await Utils.postJSON('/api/update_annotation', {
        layer_id: this.currentLayer,
        annotation: annotation
      });

      if (response.success) {
        console.log('Annotation updated successfully', 'success');
        Utils.hideModal('layer-detail-modal');
        this.loadLayers(); // Reload layers to reflect changes
        this.renderLayers();
      } else {
        throw new Error(response.error || 'Failed to update annotation');
      }
    } catch (error) {
      console.error('Error saving annotation:', error);
      console.log(`Error saving annotation: ${error.message}`, 'error');
    }
  }

  editAnnotation(layerId) {
    this.currentLayer = layerId;
    this.switchTab('annotation');
  }


  proceedToExport() {
    window.location.href = '/export';
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  window.reviewInterface = new ReviewInterface();
});
