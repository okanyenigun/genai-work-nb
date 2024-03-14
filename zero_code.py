import os
import pickle
import multiprocessing
import pandas as pd
from transformers import pipeline
from datetime import date

cpu_count = multiprocessing.cpu_count()
num_processes = max(1, cpu_count - 1)

MODEL = "facebook/bart-large-mnli"

LABELS = ["error","performance","ux","pricing","product","atm","declartive","credit","angry","urgent", "not urgent"]

pipe = pipeline("zero-shot-classification", model=MODEL)

def get_df():
    df = pd.read_excel("ing_review_w_user_ratings.xlsx")

    start_date = date(2022,1,1)
    if start_date:
        df = df[df["date"] >= pd.to_datetime(start_date)]
        df.reset_index(inplace=True,drop=True)
    print("shape of df", df.shape[0])
    return df

def analyze_zero(args):
    index, text = args
    try:
        result = pipe(text, LABELS, multi_label=True)
        print(f"PID: {os.getpid()} - Row {index}")
        output = {
            index: result
        }
        return output
    except Exception as e:
        print(f"Error processing text in row {index}: {e}")
        output = {
            index: None
        }
    return output
    

if __name__ == '__main__':
    df = get_df()
    with multiprocessing.Pool(processes=num_processes) as pool:
        index_text_pairs = zip(df.index, df["deep_translator"])
        results = pool.map(analyze_zero, index_text_pairs)
    pickle.dump(results, open("zero_shot_new_labels.pkl", "wb"))
    

