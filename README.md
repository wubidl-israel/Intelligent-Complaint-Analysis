

A comprehensive analysis system for processing and analyzing financial service complaints using natural language processing and machine learning techniques.

## Project Overview

This project aims to analyze consumer complaints in the financial services sector, providing insights through data exploration, text preprocessing, and advanced analytics. The system processes complaint data, performs exploratory data analysis, and prepares the data for further machine learning tasks.

## Features

- **Data Loading & Preprocessing**: Efficient loading and cleaning of complaint data
- **Exploratory Data Analysis**: Comprehensive analysis of complaint patterns and trends
- **Text Processing**: Advanced text cleaning and normalization
- **Modular Architecture**: Well-organized codebase with separate modules for different tasks
- **Reproducible Analysis**: Jupyter notebooks for interactive analysis and visualization

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
.
├── data/                   # Raw and processed data
├── notebooks/              # Jupyter notebooks for analysis
├── reports/                # Generated analysis reports and visualizations
├── src/                    # Source code
│   ├── notebook/           # Additional notebook resources
│   └── task1_eda_preprocess.py  # Main data processing script
├── utils/                  # Utility functions
├── .gitignore              # Git ignore file
├── LICENSE                 # MIT License
└── requirements.txt        # Project dependencies
```

## Usage

1. Place your complaint data in the `data/` directory as `complaints.csv`
2. Run the EDA and preprocessing script:
   ```bash
   python src/task1_eda_preprocess.py
   ```
3. Explore the generated reports in the `reports/` directory

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- sentence-transformers
- faiss-cpu
- chromadb
- langchain
- fastapi


