# Developer Setup and Contribution Guidelines

This document details the development environment setup, coding standards, testing routines, and pull request flows for developers contributing to the 2D Geospatial Climate Digital Twin platform.

---

## Development Environment Setup

### 1. Requirements
* **Operating System:** Platform independent (tested on Windows 10/11, Ubuntu 22.04 LTS, macOS Sequoia).
* **Python version:** Python 3.12 or higher.
* **C++ Compiler:** Required by NetCDF4 C-libraries during pip compilation (pre-built wheels are installed automatically on Windows).

### 2. Sandbox Installation
Create an isolated virtual environment and install the package requirements:
```bash
python -m venv venv312
source venv312/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Code Quality and Styling Guidelines

To maintain clean code in the repository, all contributions must adhere to the following standards:

### 1. Document Integrity and Emojis
* **Documentation Formatting:** Do not include emojis in any Markdown documentation files. Use standard, structured, high-contrast headings.
* **Inline Comments:** Retain all existing code comments and docstrings. Document new functions with Google-style docstrings, detailing parameter types and dimensions (e.g. specifying input grid sizes like 129x135).

### 2. Code Styling
* **PEP 8 Compliance:** All Python files must adhere to PEP 8 standards. Run `flake8` or `ruff` to identify style violations.
* **Decoupled Module Structure:** Keep classes and helper functions focused on a single responsibility. Business logic (such as index calculations) must remain separated from UI presentation scripts (`app/streamlit_app.py`).

---

## Testing and Verification Routines

Before submitting changes, execute the verification scripts:

### 1. Verify Dataset Integrity
Run the grid check script to verify the structure and sizing of all compiled NetCDF4 arrays:
```bash
python scripts/check_downloaded_data.py
```

### 2. Verify Data Decoding Pipeline
Run the daily binary decoders to verify coordinate projections and scale mappings:
```bash
python scripts/decode_imd_binary.py
python scripts/decode_imd_temp.py
```

---

## Pull Request Guidelines

1. **Branch Naming Conventions:**
   * Features: `feature/your-feature-name`
   * Bug fixes: `bugfix/your-bug-name`
   * Documentation: `docs/your-doc-name`
2. **Review Checklist:**
   * Confirm that all modified math expressions render correctly in markdown engines without KaTeX parse errors.
   * Run local checks to verify that Streamlit launches cleanly (`streamlit run app/streamlit_app.py`) without caching errors.
   * Ensure `config.json` is ignored by git and only `config.json.example` is tracked to prevent login credentials leakage.
