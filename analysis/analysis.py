import pandas as pd
from pathlib import Path

def load_data():
    filename = "data/Writing-Artifacts-Evaluation.xlsx"
    data = pd.read_excel(filename)
    print(data)

def main():
    load_data()

if __name__ == "__main__":
    main()