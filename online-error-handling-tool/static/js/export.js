// Error Detection Tool - Export Interface

class ExportInterface {
  constructor() {
    this.exportHistory = [];
    this.currentExport = null;
  }

  async init() {
    await this.loadExportHistory();
    this.setupEventListeners();
    this.renderExportHistory();
  }

  async loadExportHistory() {
    try {
      const response = await Utils.fetchJSON('/api/list_exports');
      this.exportHistory = response.exports || [];
    } catch (error) {
      console.error('Error loading export history:', error);
    }
  }

  setupEventListeners() {
    // Export buttons
    document.getElementById('export-session-json')?.addEventListener('click', () => {
      this.exportSession('json');
    });

    document.getElementById('export-session-csv')?.addEventListener('click', () => {
      this.exportSession('csv');
    });

    document.getElementById('export-proofreading-queue')?.addEventListener('click', () => {
      this.exportProofreadingQueue();
    });

    document.getElementById('open-proofreading')?.addEventListener('click', () => {
      this.openProofreading();
    });



    // Navigation buttons
    document.getElementById('back-to-review')?.addEventListener('click', () => {
      window.location.href = '/review';
    });

    // History controls
    document.getElementById('refresh-history')?.addEventListener('click', () => {
      this.loadExportHistory();
      this.renderExportHistory();
    });

    document.getElementById('clear-history')?.addEventListener('click', () => {
      this.clearExportHistory();
    });


    // Modal interactions
    this.setupModalEventListeners();
  }

  setupModalEventListeners() {
    // Close modals
    document.querySelectorAll('.modal-close').forEach(btn => {
      btn.addEventListener('click', (e) => {
        const modal = e.target.closest('.modal');
        Utils.hideModal(modal.id);
      });
    });

    // Download export
    document.getElementById('download-export')?.addEventListener('click', () => {
      this.downloadCurrentExport();
    });
  }

  async exportSession(format) {
    this.showExportProgress('Exporting session data...');

    try {
      const response = await Utils.postJSON('/api/export_session', {
        format: format,
        include_correct: false
      });

      if (response.success) {
        this.showExportResult(response);
        await this.loadExportHistory();
        this.renderExportHistory();
      } else {
        throw new Error(response.error || 'Export failed');
      }
    } catch (error) {
      console.error('Error exporting session:', error);
      console.log(`Error exporting session: ${error.message}`, 'error');
    } finally {
      this.hideExportProgress();
    }
  }

  async exportProofreadingQueue() {
    this.showExportProgress('Exporting proofreading queue...');

    try {
      const response = await Utils.postJSON('/api/export_proofreading_queue');

      if (response.success) {
        this.showExportResult(response);
        await this.loadExportHistory();
        this.renderExportHistory();
      } else {
        throw new Error(response.error || 'Export failed');
      }
    } catch (error) {
      console.error('Error exporting proofreading queue:', error);
      console.log(`Error exporting proofreading queue: ${error.message}`, 'error');
    } finally {
      this.hideExportProgress();
    }
  }

  async openProofreading() {
    this.showExportProgress('Opening proofreading interface...');

    try {
      const response = await Utils.postJSON('/api/open_proofreading');

      if (response.success) {
        // Navigate to proofreading page
        window.location.href = response.proofreading_url;
      } else {
        throw new Error(response.error || 'Failed to open proofreading');
      }
    } catch (error) {
      console.error('Error opening proofreading:', error);
      console.log(`Error opening proofreading: ${error.message}`, 'error');
    } finally {
      this.hideExportProgress();
    }
  }


  async exportLayerImages() {
    this.showExportProgress('Exporting layer images...');

    try {
      const includeOverlays = document.getElementById('include-overlays')?.checked || false;
      const statusFilter = document.getElementById('status-filter-export')?.value || 'all';

      const response = await Utils.postJSON('/api/export_layer_images', {
        include_overlays: includeOverlays,
        status_filter: statusFilter
      });

      if (response.success) {
        this.showExportResult(response);
        await this.loadExportHistory();
        this.renderExportHistory();
      } else {
        throw new Error(response.error || 'Export failed');
      }
    } catch (error) {
      console.error('Error exporting layer images:', error);
      console.log(`Error exporting layer images: ${error.message}`, 'error');
    } finally {
      this.hideExportProgress();
    }
  }

  showExportProgress(message) {
    const modal = document.getElementById('export-progress-modal');
    const progressText = document.getElementById('export-progress-text');
    
    if (progressText) {
      progressText.textContent = message;
    }
    
    Utils.showModal('export-progress-modal');
  }

  hideExportProgress() {
    Utils.hideModal('export-progress-modal');
  }

  showExportResult(result) {
    this.currentExport = result;
    
    const modal = document.getElementById('export-result-modal');
    const content = document.getElementById('export-result-content');
    
    if (content) {
      content.innerHTML = `
        <div class="export-result">
          <h4>Export Complete</h4>
          <div class="result-details">
            <div class="result-item">
              <span class="result-label">File:</span>
              <span class="result-value">${result.export_path || result.filename || 'Unknown'}</span>
            </div>
            ${result.file_size ? `
            <div class="result-item">
              <span class="result-label">Size:</span>
              <span class="result-value">${Utils.formatFileSize(result.file_size)}</span>
            </div>
            ` : ''}
            ${result.layers_exported ? `
            <div class="result-item">
              <span class="result-label">Layers Exported:</span>
              <span class="result-value">${result.layers_exported}</span>
            </div>
            ` : ''}
            ${result.files_exported ? `
            <div class="result-item">
              <span class="result-label">Files Exported:</span>
              <span class="result-value">${result.files_exported}</span>
            </div>
            ` : ''}
          </div>
        </div>
      `;
    }
    
    Utils.showModal('export-result-modal');
  }

  downloadCurrentExport() {
    if (!this.currentExport) {
      console.log('No export to download', 'error');
      return;
    }

    const filename = this.currentExport.export_path || this.currentExport.filename;
    if (filename) {
      const downloadUrl = `/api/download_export/${filename.split('/').pop()}`;
      Utils.downloadFile(downloadUrl, filename.split('/').pop());
      Utils.hideModal('export-result-modal');
    }
  }

  renderExportHistory() {
    const historyList = document.getElementById('export-history-list');
    if (!historyList) return;

    historyList.innerHTML = '';

    if (this.exportHistory.length === 0) {
      historyList.innerHTML = '<div class="no-exports">No export history available.</div>';
      return;
    }

    this.exportHistory.forEach(exportItem => {
      const historyItem = this.createHistoryItem(exportItem);
      historyList.appendChild(historyItem);
    });
  }

  createHistoryItem(exportItem) {
    const item = document.createElement('div');
    item.className = 'history-item';
    
    item.innerHTML = `
      <div class="history-info">
        <div class="history-filename">${exportItem.filename}</div>
        <div class="history-size">${Utils.formatFileSize(exportItem.size)}</div>
        <div class="history-date">${Utils.formatDate(exportItem.modified * 1000)}</div>
      </div>
      <div class="history-actions">
        <button class="btn btn-primary btn-sm" onclick="exportInterface.downloadHistoryFile('${exportItem.filename}')">
          Download
        </button>
        <button class="btn btn-danger btn-sm" onclick="exportInterface.deleteHistoryFile('${exportItem.filename}')">
          Delete
        </button>
      </div>
    `;

    return item;
  }

  async downloadHistoryFile(filename) {
    try {
      const downloadUrl = `/api/download_export/${filename}`;
      Utils.downloadFile(downloadUrl, filename);
    } catch (error) {
      console.error('Error downloading file:', error);
      console.log('Error downloading file', 'error');
    }
  }

  async deleteHistoryFile(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) {
      return;
    }

    try {
      const response = await Utils.fetchJSON(`/api/delete_export/${filename}`, {
        method: 'DELETE'
      });

      if (response.success) {
        console.log('File deleted successfully', 'success');
        await this.loadExportHistory();
        this.renderExportHistory();
      } else {
        throw new Error(response.error || 'Failed to delete file');
      }
    } catch (error) {
      console.error('Error deleting file:', error);
      console.log(`Error deleting file: ${error.message}`, 'error');
    }
  }

  async clearExportHistory() {
    if (!confirm('Are you sure you want to clear all export history? This action cannot be undone.')) {
      return;
    }

    try {
      // Delete all files in the export directory
      for (const exportItem of this.exportHistory) {
        await Utils.fetchJSON(`/api/delete_export/${exportItem.filename}`, {
          method: 'DELETE'
        });
      }

      console.log('Export history cleared', 'success');
      await this.loadExportHistory();
      this.renderExportHistory();
    } catch (error) {
      console.error('Error clearing export history:', error);
      console.log('Error clearing export history', 'error');
    }
  }

}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
  window.exportInterface = new ExportInterface();
});
