# Document Processing System

The **Document Processing System** is a Python-based project designed to process PDF documents by detecting layout elements, extracting text, and generating structured outputs. It is tailored for use cases such as research paper analysis, and document digitization.

## Features

- **PDF to Image Conversion**: Converts PDF pages into images using `pdf2image`.
- **Layout Detection**: Identifies structural elements like titles, text blocks, figures, and tables using a YOLO model.
- **Text Extraction**: Performs OCR on detected layout regions using `pytesseract`.
- **Annotated Images**: Adds bounding boxes and labels to images for visual verification.
- **Structured Output**: Saves extracted data as JSON for further use.

## Use Cases

- **Research Document Analysis**: Extract titles, abstracts, figures, and tables for indexing or research purposes.
- **Digitization of Documents**: Convert scanned PDFs into machine-readable formats.

## Installation

### Prerequisites

1. Python 3.8 or higher installed on your machine.
2. Install Tesseract OCR:
   ```bash
   sudo apt-get update
   sudo apt-get install -y tesseract-ocr
   ```
3. Install system dependencies for OpenCV:
   ```bash
   sudo apt-get install -y libgl1
   ```

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/neha13rana/Document-Processing-System.git
   cd Document-Processing-System
   ```

2. Run the setup script to install dependencies:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

## Usage

### Command-Line Processing

Process a PDF directly via the command line:

```bash
python new.py --pdf_path path/to/your/document.pdf --output_folder output_directory/
```

## Outputs

1. **Annotated Images**:
   - Images with bounding boxes and labels indicating detected elements.
   - Example: `output_directory/2022061018/annotated_images4/page_1.jpg`.

2. **Structured JSON**:
   - A JSON file containing extracted data (e.g., labels, bounding boxes, text).
   - Example: `output_directory/2022061018/2022061018_structured_results.json`.

## Project Structure

```plaintext
doclayoutanalysis/
├── inputs/                        # Directory for input PDF files
├── output_directory/              # Directory for processed outputs
│   ├── 2022061018/                # Session-specific folder (e.g., based on ID)
│   │   ├── annotated_images4/     # Folder containing annotated images
│   │   │   └── page_1.jpg         # Annotated image with bounding boxes and labels
│   │   └── 2022061018_structured_results.json  # Extracted structured data in JSON format
│   ├── 2024020622/                # Another session-specific folder
│   ├── 2024020637/                # Another session-specific folder
├── venv/                          # Virtual environment for Python dependencies
├── new.py                         # Script for processing a single PDF with text extraction  
├── process_directory_pdfs.py      # Script for processing all PDFs in a directory
├── process_pdf_with_text_extraction.py  # Experimental new script
├── requirements.txt               # List of all Python dependencies
├── setup.sh                       # Shell script for setting up the environment
└── README.md                      # Project documentation
```

## Dependencies

All required dependencies are listed in `requirements.txt` and installed via the `setup.sh` script.


## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push to your fork:
   ```bash
   git push origin feature-name
   ```
4. Submit a pull request to the main repository.

