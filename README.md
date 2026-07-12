# Customer Segmentation

This project is a simple Streamlit app that groups customers using K-Means clustering. Upload a CSV file with `Age` and `Income` columns, choose the number of clusters, and view the segmented output with a chart.

## Features

- Upload customer data as a CSV file
- Automatically validate required columns: `Age` and `Income`
- Remove invalid or non-numeric rows before clustering
- Choose the number of clusters interactively
- View clustered data, cluster centers, and a scatter plot

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- matplotlib

## Project Files

```text
customer-segmentation/
|-- custom.py
|-- CustomerData.csv
|-- README.md
```

## How to Run

1. Install dependencies:

```bash
pip install pandas matplotlib streamlit scikit-learn
```

2. Start the Streamlit app:

```bash
streamlit run custom.py
```

3. Open the local URL shown in the terminal and upload a CSV file.

## Expected CSV Format

The uploaded file must contain these columns:

- `Age`
- `Income`

Example:

```csv
Age,Income
25,30000
32,50000
41,70000
```

## Sample Data

A sample file is already included:

- `CustomerData.csv`

You can use it to quickly test the app.

## What the App Shows

After upload, the app:

- previews the cleaned data
- assigns each customer to a cluster
- displays cluster centers
- plots customer groups on an `Age` vs `Income` graph

## Notes

- Empty CSV files are rejected
- Invalid CSV format is handled with an error message
- Rows with missing or non-numeric `Age` or `Income` values are removed
- The maximum cluster count is limited by dataset size and capped at 6
