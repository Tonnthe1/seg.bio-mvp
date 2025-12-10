// Error Detection Tool - Detection Interface

class DetectionInterface {
  constructor() {
    this.layers = [];
    this.currentLayer = null;
    this.annotations = new Map();
    this.currentScope = 'current'; // Default to current page
    this.progress = {
      total: 0,
      correct: 0,
      incorrect: 0,
      unsure: 0,
      unlabeled: 0
    };
  }

  async init(serverProgress = null) {
    this.loadLayers();
    this.setupEventListeners();
    this.updateProgress(serverProgress);
  }

  async loadLayers() {
    try {
      // Load layers from the page data
      const layerCards = document.querySelectorAll('.layer-card');
      this.layers = Array.from(layerCards).map(card => ({
        id: card.dataset.layerId,
        z: parseInt(card.dataset.z),
        element: card
      }));

      this.progress.total = this.layers.length;
      // Don't override server progress data
    } catch (error) {
      console.error('Error loading layers:', error);
      console.error('Error loading layers');
    }
  }

  setupEventListeners() {
    // Layer card interactions
    document.querySelectorAll('.layer-card').forEach(card => {
      const layerId = card.dataset.layerId;
      
      // Click to select layer
      card.addEventListener('click', (e) => {
        if (!e.target.closest('.layer-actions')) {
          this.selectLayer(layerId);
        }
      });

      // Status button clicks
      card.querySelectorAll('.status-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const status = btn.dataset.status;
          this.annotateLayer(layerId, status);
        });
      });
    });

    // Scope selection - radio button change events
    document.querySelectorAll('input[name="scope"]').forEach(radio => {
      radio.addEventListener('change', (e) => {
        this.setScope(e.target.value);
      });
    });

    // Batch operations
    document.getElementById('select-all-correct')?.addEventListener('click', () => {
      this.executeBatchAction('correct', this.currentScope);
    });

    document.getElementById('select-all-incorrect')?.addEventListener('click', () => {
      this.executeBatchAction('incorrect', this.currentScope);
    });

    document.getElementById('clear-selections')?.addEventListener('click', () => {
      this.executeBatchAction('clear', this.currentScope);
    });

    // Proceed button
    document.getElementById('proceed-to-review')?.addEventListener('click', () => {
      this.proceedToReview();
    });

    // Modal interactions
    this.setupModalEventListeners();
    
    // Page input functionality
    this.setupPageInputListeners();
  }
  
  setupPageInputListeners() {
    const pageInput = document.getElementById('page-input');
    const goToPageBtn = document.getElementById('go-to-page');
    
    if (pageInput && goToPageBtn) {
      // Go to page button
      goToPageBtn.addEventListener('click', () => {
        this.goToPage();
      });
      
      // Enter key in page input
      pageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
          this.goToPage();
        }
      });
      
      // Validate input on change
      pageInput.addEventListener('input', () => {
        this.validatePageInput();
      });
    }
  }
  
  validatePageInput() {
    const pageInput = document.getElementById('page-input');
    if (!pageInput) return;
    
    const value = parseInt(pageInput.value);
    const maxPages = parseInt(pageInput.getAttribute('max'));
    
    if (isNaN(value) || value < 1) {
      pageInput.value = '';
    } else if (value > maxPages) {
      pageInput.value = maxPages;
    }
  }
  
  goToPage() {
    const pageInput = document.getElementById('page-input');
    if (!pageInput) return;
    
    const page = parseInt(pageInput.value);
    const maxPages = parseInt(pageInput.getAttribute('max'));
    
    if (isNaN(page) || page < 1) {
      console.error('Please enter a valid page number');
      return;
    }
    
    if (page > maxPages) {
      console.log(`Page number cannot exceed ${maxPages}`, 'error');
      return;
    }
    
    // Navigate to the page
    const currentUrl = new URL(window.location);
    currentUrl.searchParams.set('page', page);
    window.location.href = currentUrl.toString();
  }

  setupModalEventListeners() {
    // Layer detail modal
    const layerModal = document.getElementById('layer-modal');
    const annotationModal = document.getElementById('annotation-modal');

    // Close modals
    document.querySelectorAll('.modal-close').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const modal = e.target.closest('.modal');
        Utils.hideModal(modal.id);
      });
    });

    // Annotation form submission
    document.getElementById('annotation-form')?.addEventListener('submit', (e) => {
      e.preventDefault();
      this.submitAnnotation();
    });

    // Status change handling
    document.getElementById('annotation-status')?.addEventListener('change', (e) => {
      this.handleStatusChange(e.target.value);
    });
  }

  selectLayer(layerId) {
    // Remove previous selection
    document.querySelectorAll('.layer-card.selected').forEach(card => {
      card.classList.remove('selected');
    });

    // Add selection to current layer
    const layerCard = document.querySelector(`[data-layer-id="${layerId}"]`);
    if (layerCard) {
      layerCard.classList.add('selected');
      this.currentLayer = layerId;
    }
  }

  async annotateLayer(layerId, status, annotation = {}) {
    try {
      const response = await Utils.postJSON('/api/annotate', {
        layer_id: layerId,
        status: status,
        annotation: annotation
      });

      if (response.success) {
        this.updateLayerStatus(layerId, status);
        // Use server progress data for accuracy
        this.updateProgress(response.progress);
        this.annotations.set(layerId, { status, annotation });
      } else {
        throw new Error(response.error || 'Failed to annotate layer');
      }
    } catch (error) {
      console.error('Error annotating layer:', error);
      console.log(`Error annotating layer: ${error.message}`, 'error');
    }
  }

  updateLayerStatus(layerId, status) {
    const layerCard = document.querySelector(`[data-layer-id="${layerId}"]`);
    if (!layerCard) return;

    // Update the data-status attribute
    layerCard.dataset.status = status;

    // Get the previous status to track changes
    const previousStatus = this.getLayerStatus(layerCard);
    
    // Update status indicator
    const statusIndicator = layerCard.querySelector('.status-indicator');
    const statusText = layerCard.querySelector('.status-text');
    
    statusIndicator.className = `status-indicator ${status}`;
    statusText.textContent = status.charAt(0).toUpperCase() + status.slice(1);

    // Update button states
    layerCard.querySelectorAll('.status-btn').forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.status === status) {
        btn.classList.add('active');
      }
    });

    // Update card appearance
    layerCard.classList.remove('correct', 'incorrect', 'unsure', 'unlabeled');
    layerCard.classList.add(status);
  }

  getLayerStatus(layerCard) {
    // Determine current status from CSS classes
    if (layerCard.classList.contains('correct')) return 'correct';
    if (layerCard.classList.contains('incorrect')) return 'incorrect';
    if (layerCard.classList.contains('unsure')) return 'unsure';
    return 'unlabeled';
  }

  updateProgress(progressData = null) {
    if (progressData) {
      this.progress = progressData;
    } else {
      // Calculate progress from current layer states
      this.calculateProgressFromLayers();
    }

    // Update progress display
    document.querySelectorAll('.stat-value').forEach(stat => {
      const label = stat.previousElementSibling.textContent.toLowerCase().trim();
      // Use exact matching to avoid substring issues (e.g., "incorrect" contains "correct")
      if (label === 'total:') stat.textContent = this.progress.total;
      else if (label === 'correct:') stat.textContent = this.progress.correct;
      else if (label === 'incorrect:') stat.textContent = this.progress.incorrect;
      else if (label === 'unsure:') stat.textContent = this.progress.unsure;
      else if (label === 'unlabeled:') stat.textContent = this.progress.unlabeled;
    });

    // Update progress bar
    const progressFill = document.querySelector('.progress-fill');
    if (progressFill) {
      const completionRate = this.progress.completion_rate || 0;
      progressFill.style.width = `${completionRate * 100}%`;
    }
  }

  calculateProgressFromLayers() {
    // Count layers by status from DOM
    const layerCards = document.querySelectorAll('.layer-card');
    let correct = 0, incorrect = 0, unsure = 0, unlabeled = 0;
    
    layerCards.forEach(card => {
      if (card.classList.contains('correct')) correct++;
      else if (card.classList.contains('incorrect')) incorrect++;
      else if (card.classList.contains('unsure')) unsure++;
      else unlabeled++;
    });
    
    const total = layerCards.length;
    const completionRate = (correct + incorrect + unsure) / total;
    
    this.progress = {
      total,
      correct,
      incorrect,
      unsure,
      unlabeled,
      completion_rate: completionRate
    };
  }

  async batchAnnotate(status) {
    const unlabeledLayers = this.layers.filter(layer => {
      const layerCard = document.querySelector(`[data-layer-id="${layer.id}"]`);
      const currentStatus = layerCard?.dataset.status || 'unlabeled';
      return currentStatus === 'unlabeled';
    });

    if (unlabeledLayers.length === 0) {
      console.log('No unlabeled layers to annotate', 'info');
      return;
    }

    const confirmMessage = `Are you sure you want to mark all ${unlabeledLayers.length} unlabeled layers as ${status}?`;
    if (!confirm(confirmMessage)) return;

    try {
      const annotations = unlabeledLayers.map(layer => ({
        layer_id: layer.id,
        status: status,
        annotation: {}
      }));

      const response = await Utils.postJSON('/api/batch_annotate', {
        annotations: annotations
      });

      if (response.success) {
        // Update all layers
        unlabeledLayers.forEach(layer => {
          this.updateLayerStatus(layer.id, status);
        });

        // Use server progress data for accuracy
        this.updateProgress(response.progress);
        console.log(`Successfully annotated ${unlabeledLayers.length} layers as ${status}`, 'success');
      } else {
        throw new Error(response.error || 'Failed to batch annotate');
      }
    } catch (error) {
      console.error('Error batch annotating:', error);
      console.log(`Error batch annotating: ${error.message}`, 'error');
    }
  }

  async clearSelections() {
    try {
      // Remove all visual selections
      document.querySelectorAll('.layer-card.selected').forEach(card => {
        card.classList.remove('selected');
      });

      this.currentLayer = null;

      // Reset all layer statuses to unlabeled
      const layerCards = document.querySelectorAll('.layer-card');
      for (const card of layerCards) {
        const layerId = card.dataset.layerId;
        if (layerId) {
          // Update the layer status to unlabeled
          await this.annotateLayer(layerId, 'unlabeled');
        }
      }

      console.log('All selections cleared', 'success');
    } catch (error) {
      console.error('Error clearing selections:', error);
      console.log(`Error clearing selections: ${error.message}`, 'error');
    }
  }

  setScope(scope) {
    this.currentScope = scope;
    
    // Update radio button states
    document.querySelectorAll('input[name="scope"]').forEach(radio => {
      radio.checked = (radio.value === scope);
    });
    
    console.log(`Scope set to: ${scope}`);
  }

  async executeBatchAction(action, scope) {
    try {
      if (scope === 'current') {
        // Apply to current page only
        await this.executeCurrentPageAction(action);
      } else if (scope === 'all') {
        // Apply to all pages
        await this.executeAllPagesAction(action);
      }
    } catch (error) {
      console.error(`Error executing ${action} for ${scope}:`, error);
      console.log(`Error executing ${action} for ${scope}`, 'error');
    }
  }

  async executeCurrentPageAction(action) {
    const layerCards = document.querySelectorAll('.layer-card');
    let updatedCount = 0;
    
    for (const card of layerCards) {
      const layerId = card.dataset.layerId;
      const currentStatus = card.dataset.status || 'unlabeled';
      
      // Only apply to unselected layers (skip already marked layers)
      // For 'clear' action, process all layers
      // For other actions, only process unlabeled layers
      if (action === 'clear') {
        // Clear action affects all layers
        await this.annotateLayer(layerId, 'unlabeled');
        updatedCount++;
      } else if (currentStatus === 'unlabeled') {
        // Other actions only affect unlabeled layers
        await this.annotateLayer(layerId, action);
        updatedCount++;
      }
    }
    
    if (updatedCount === 0) {
      console.log('No unselected layers to update on current page', 'info');
    } else {
      const actionText = action === 'clear' ? 'cleared' : `marked as ${action}`;
      console.log(`${updatedCount} unselected layers ${actionText} on current page`, 'success');
    }
  }

  async executeAllPagesAction(action) {
    try {
      const response = await fetch('/api/batch_update_all_pages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          action: action,
          scope: 'all'
        })
      });

      const result = await response.json();
      
      if (result.success) {
        if (result.updated_count === 0) {
          console.log('No unselected layers to update across all pages', 'info');
        } else {
          const actionText = action === 'clear' ? 'cleared' : `marked as ${action}`;
          console.log(`${result.updated_count} unselected layers ${actionText} across all pages`, 'success');
        }
        
        // Reload the page to show updated data
        window.location.reload();
      } else {
        throw new Error(result.error || 'Batch operation failed');
      }
    } catch (error) {
      console.error('Error executing all pages action:', error);
      console.log(`Error applying ${action} to all pages`, 'error');
    }
  }


  proceedToReview() {
    // Check if there are any incorrect or unsure layers
    const incorrectCount = this.progress.incorrect;
    const unsureCount = this.progress.unsure;

    if (incorrectCount === 0 && unsureCount === 0) {
      console.log('No layers require review. All layers are marked as correct.', 'info');
      return;
    }

    // Navigate to review page
    window.location.href = '/review';
  }

  handleStatusChange(status) {
    const errorTypeGroup = document.getElementById('error-type-group');
    const severityGroup = document.getElementById('severity-group');

    if (status === 'incorrect') {
      errorTypeGroup.style.display = 'block';
      severityGroup.style.display = 'block';
    } else {
      errorTypeGroup.style.display = 'none';
      severityGroup.style.display = 'none';
    }
  }

  async submitAnnotation() {
    if (!this.currentLayer) {
      console.log('No layer selected', 'error');
      return;
    }

    const form = document.getElementById('annotation-form');
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
      await this.annotateLayer(this.currentLayer, annotation.status, annotation);
      Utils.hideModal('annotation-modal');
      console.log('Annotation saved successfully', 'success');
    } catch (error) {
      console.log(`Error saving annotation: ${error.message}`, 'error');
    }
  }

  async loadLayerDetails(layerId) {
    try {
      const response = await Utils.fetchJSON(`/api/layer_details/${layerId}`);
      return response;
    } catch (error) {
      console.error('Error loading layer details:', error);
      throw error;
    }
  }

  async getErrorSuggestions(layerId) {
    try {
      const response = await Utils.fetchJSON(`/api/error_suggestions/${layerId}`);
      return response.suggestions;
    } catch (error) {
      console.error('Error getting error suggestions:', error);
      return [];
    }
  }

  showLayerModal(layerId) {
    this.currentLayer = layerId;
    Utils.showModal('layer-modal');
    this.loadLayerDetails(layerId);
  }

  showAnnotationModal(layerId) {
    this.currentLayer = layerId;
    Utils.showModal('annotation-modal');
    
    // Load current annotation if exists
    const currentAnnotation = this.annotations.get(layerId);
    if (currentAnnotation) {
      const form = document.getElementById('annotation-form');
      form.status.value = currentAnnotation.status;
      form.error_type.value = currentAnnotation.annotation.error_type || '';
      form.severity.value = currentAnnotation.annotation.severity || 'medium';
      form.description.value = currentAnnotation.annotation.description || '';
      form.confidence.value = currentAnnotation.annotation.confidence || 0.5;
      form.notes.value = currentAnnotation.annotation.notes || '';
      
      this.handleStatusChange(currentAnnotation.status);
    }
  }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  window.detectionInterface = new DetectionInterface();
});
