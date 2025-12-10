# EHTool - Error Handling Tool

A comprehensive web application for error detection and proofreading in image segmentation workflows. EHTool integrates error detection capabilities with advanced proofreading tools to help users identify and correct segmentation errors efficiently.

## 🚀 Features

### **Dual Workflow System**
- **Error Detection Workflow**: Identify and categorize segmentation errors with lazy loading for large datasets
- **Standalone Proofreading Workflow**: Direct image editing and mask correction with full PFTool functionality
- **Integrated Proofreading**: Seamlessly edit incorrect layers from error detection workflow

### **Error Detection Capabilities**
- Load and analyze 2D/3D image datasets (TIFF, PNG, JPG, JPEG)
- Interactive layer-by-layer error detection
- **Lazy loading with pagination** - Only loads 12 layers per page for better performance
- Categorize layers as: Correct, Incorrect, or Unsure
- Batch operations for efficient labeling
- Progress tracking and statistics
- nnUNet-style mask naming support (masks matching images without "_0000" suffix)

### **Advanced Proofreading Tools**
- Real-time mask editing with paint and erase tools
- Adjustable brush sizes (1-64 pixels)
- Custom circular cursor that shows brush size
- Keyboard shortcuts for efficient editing
- Layer navigation for 3D datasets
- Undo/Redo support for all operations
- Automatic mask generation and saving

### **User Interface**
- Modern, responsive web interface
- Intuitive navigation between workflows
- Real-time progress tracking
- Comprehensive keyboard shortcuts
- Custom cursor preview for paint/erase tools
- Layer-by-layer navigation controls

## 🛠️ Installation

### Prerequisites
- **Python 3.9 or higher**
- **pip** (Python package manager)
- **Modern web browser** (Chrome, Firefox, Safari, Edge)
- **8GB+ RAM** recommended for large datasets

### Setup Instructions

1. **Clone or download the repository:**
```bash
git clone <repository-url>
cd online-error-handling-tool
```

2. **Create a virtual environment (recommended):**
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
python app.py
```

5. **Open your browser:**
Navigate to:
```
http://localhost:5004
```

The application will start and you'll see the landing page where you can choose your workflow.

## 📁 Project Structure

```
online-error-handling-tool/
├── app.py                          # Main Flask application entry point
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── routes/                         # Flask route handlers
│   ├── landing.py                 # Home page and main navigation
│   ├── detection_workflow.py      # Error detection workflow routes
│   ├── proofreading_workflow.py    # Standalone proofreading routes
│   ├── proofreading.py            # Integrated proofreading routes
│   ├── detection.py               # Detection page logic
│   ├── review.py                  # Review and statistics
│   └── export.py                   # Export functionality
│
├── backend/                        # Core backend modules
│   ├── session_manager.py         # Session state management
│   ├── data_manager.py            # Data loading and processing
│   ├── volume_manager.py          # Volume and mask operations
│   ├── utils.py                   # Shared utility functions
│   └── ai/                         # AI model integration
│       └── error_detection.py     # Error detection logic
│
├── templates/                      # HTML templates
│   ├── base.html                  # Base template
│   ├── landing.html               # Home page
│   ├── detection_load.html        # Data loading interface
│   ├── detection.html             # Error detection interface
│   ├── proofreading_load.html     # Proofreading data loading
│   ├── proofreading_selection.html # Layer selection for proofreading
│   ├── proofreading_standalone.html # Standalone proofreading editor
│   └── proofreading.html          # Integrated proofreading editor
│
└── static/                         # Static assets
    ├── css/                       # Stylesheets
    └── js/                        # JavaScript files
```

## 🎯 Usage Guide

### **Getting Started**

1. **Launch EHTool**: Open your browser to `http://localhost:5004`
2. **Choose Workflow**: 
   - Click **"Error Detection"** to identify incorrect segmentation layers
   - Click **"Proofreading"** for direct image/mask editing
3. **Load Dataset**: Upload files or provide local file paths
4. **Start Working**: Begin your workflow

### **Error Detection Workflow**

#### Step 1: Load Dataset
- **Option A - Upload Files:**
  - Click "Choose File" under Image
  - Upload image file(s) or folder
  - Optionally upload corresponding mask files
  - Supports: TIFF, PNG, JPG, JPEG formats
  
- **Option B - File Paths:**
  - Enter the path to your image file or folder
  - Enter mask path (optional - will create empty masks if not provided)
  - Supports local paths and relative paths
  - **nnUNet support**: If mask names match image names without "_0000", they'll be automatically paired

#### Step 2: Detect Errors
- **Lazy Loading**: Only the first 12 layers load initially for faster startup
- **Navigate**: Use pagination controls to load more layers (12 per page)
- **Categorize Layers:**
  - Click **"Correct"** for properly segmented layers
  - Click **"Incorrect"** for layers needing correction
  - Click **"Unsure"** for layers requiring review
- **Batch Operations**: Select multiple layers and label them at once
- **Progress Tracking**: View statistics showing completed vs. remaining layers

#### Step 3: Proofread Incorrect Layers
- Click **"Edit Selected Layers"** to open the integrated proofreading editor
- Navigate between incorrect layers using arrow buttons or keyboard
- Edit masks using paint/erase tools
- Mark layers as "Corrected" when done
- Changes are automatically saved

### **Standalone Proofreading Workflow**

#### Step 1: Load Dataset
- Upload or provide path to image files
- Upload or provide path to mask files (optional)
- Supports single files, folders, or multi-file image stacks
- Automatically handles 2D images and 3D stacks

#### Step 2: Edit Masks

1. **Select Tool:**
   - Click **🖌️ Paint** to add mask areas (or press `P`)
   - Click **🧽 Erase** to remove mask areas (or press `E`)
   
2. **Adjust Brush Size:**
   - Use the "Paint Brush" slider (1-64 pixels)
   - Use the "Erase Brush" slider (1-64 pixels)
   - Circular cursor preview shows brush size in real-time

3. **Edit:**
   - Click and drag on the canvas to paint or erase
   - Use **Undo** (⌘/Ctrl+Z) to revert changes
   - Use **Redo** (⌘/Ctrl+Shift+Z) to reapply changes
   - Toggle mask visibility with "Hide Mask" button

#### Step 3: Navigate (3D Datasets)
- Use **←/→** buttons or arrow keys (`A`/`D`) to move between slices
- Use the slice number input to jump to a specific slice
- Current slice and total slices are displayed

#### Step 4: Save Changes
- Click **"Save"** button (or ⌘/Ctrl+S)
- Mask is saved to the original location or alongside image file
- For folder-based datasets, masks are saved as `{image_name}_mask.{ext}`
- Original file format is preserved

### **Integrated Proofreading Workflow**

This workflow is accessed from the Error Detection interface when you have incorrect layers.

1. **Select Layers**: Click "Edit Selected Layers" from the error detection page
2. **Navigate**: Use slice controls or keyboard (`A`/`D`) to move between incorrect layers
3. **Edit**: Use the same paint/erase tools as standalone proofreading
4. **Mark Corrected**: Click "Mark as Corrected" to return the layer to review
5. **Continue**: Automatically navigate to next incorrect layer or return to selection

## ⌨️ Keyboard Shortcuts

### **General Navigation**
- `A` or `←` - Previous layer/slice
- `D` or `→` - Next layer/slice
- `⌘/Ctrl + S` - Save current work

### **Editing Tools (Proofreading)**
- `P` - Switch to Paint mode
- `E` - Switch to Erase mode
- `⌘/Ctrl + Z` - Undo last action
- `⌘/Ctrl + Shift + Z` or `Y` - Redo action

### **Zoom (Proofreading Editor)**
- `⌘/Ctrl + Mouse Scroll` - Zoom in/out
- Canvas automatically centers on zoom

## 🔧 Technical Details

### **Supported Formats**
- **Images**: TIFF, TIFF stack, PNG, JPG, JPEG
- **Masks**: TIFF, PNG, JPG, JPEG
- **3D Data**: Multi-page TIFF files, folder of 2D images
- **nnUNet**: Automatic pairing of masks with names matching images without "_0000"

### **Data Management**
- **Session-based state management** - Work is saved in browser session
- **Automatic data clearing** between workflows
- **Lazy loading** - Only loads visible layers (12 per page in detection)
- **Memory-efficient processing** for large datasets
- **Separate storage** for each workflow (no interference)

### **Architecture**
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Backend**: Flask (Python 3.9+)
- **Image Processing**: OpenCV, PIL/Pillow, NumPy, Tifffile, scikit-image, scipy
- **State Management**: Custom session manager

## 🚨 Troubleshooting

### **Common Issues**

#### **1. Application Won't Start**
- **Check Python version**: Run `python --version` (need 3.9+)
- **Check dependencies**: Run `pip install -r requirements.txt` again
- **Port in use**: Change port in `app.py` if 5004 is unavailable
- **Firewall**: Ensure firewall allows local connections

#### **2. Data Not Loading**
- **File paths**: Check paths are correct and accessible
- **Permissions**: Ensure files/folders are readable
- **Format support**: Verify file format is supported (TIFF, PNG, JPG, JPEG)
- **Size limits**: Very large files (>2GB) may need more RAM
- **Browser console**: Check browser console (F12) for error messages

#### **3. Performance Issues**
- **Large datasets**: Use lazy loading (only 12 layers per page in detection)
- **Memory**: Close other applications to free RAM
- **3D stacks**: Consider splitting into smaller chunks

#### **4. Mask Not Saving**
- **Permissions**: Check write permissions for save directory
- **Path validity**: Ensure save path exists and is writable
- **Format**: Original format is preserved when saving
- **Browser console**: Check for error messages

### **Performance Tips**
- ✅ Use **TIFF format** for best performance with large datasets
- ✅ **Enable lazy loading** in detection workflow (default: 12 layers/page)
- ✅ **Close other applications** when working with large datasets
- ✅ **Batch operations** for labeling multiple layers at once
- ✅ **Use keyboard shortcuts** for faster editing

### **Browser Compatibility**
- ✅ **Chrome/Edge** (recommended) - Best performance
- ✅ **Firefox** - Full support
- ✅ **Safari** - Full support (Mac)
- ⚠️ **Internet Explorer** - Not supported

## 🔄 Workflow Separation

EHTool maintains separate data storage for different workflows to prevent interference:

- **Error Detection**: `DETECTION_*` config variables
- **Standalone Proofreading**: `PROOFREADING_*` config variables  
- **Integrated Proofreading**: `INTEGRATED_*` config variables
- **Landing Page**: `LANDING_*` config variables

Use the **"Reset"** button on the landing page to clear all workflow data.

## 📊 Export Options

- **Session Data**: Complete workflow state with all annotations
- **Annotations**: Layer classifications (Correct/Incorrect/Unsure)
- **Statistics**: Progress and accuracy metrics
- **Proofreading Queue**: List of incorrect layers for review
- **Edited Masks**: All corrected mask files in original format

## 🆕 Recent Updates

### **Recent Features**
- ⚡ **Lazy Loading** - Faster startup for large datasets (12 layers per page)
- 🎯 **nnUNet Support** - Automatic mask pairing for nnUNet-preprocessed data
- 🔧 **Code Refactoring** - Consolidated duplicate code, improved consistency
- 🐛 **Bug Fixes** - Fixed mask update inconsistencies, improved error handling
- ⚙️ **Performance Improvements** - Optimized memory usage, faster rendering

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test thoroughly
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on **Flask** web framework
- Image processing with **OpenCV**, **PIL/Pillow**, **NumPy**, **Tifffile**, **scikit-image**, **scipy**
- UI/UX inspired by modern web applications
- Keyboard shortcuts optimized for productivity

## 📞 Support

For issues, questions, or feature requests:
- Check the **Troubleshooting** section above
- Review browser console for error messages
- Check file permissions and paths
- Ensure all dependencies are installed

---

**EHTool** - Making error detection and proofreading efficient and intuitive! 🎯
