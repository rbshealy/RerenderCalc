import csv

import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv("./data/closed_loop_trajectory.csv",header=0,quoting=csv.QUOTE_NONE,on_bad_lines='skip')
    row_c = len(df)
    print(df.shape)
    df = df.truncate(after=row_c / 34.0)  # Keeps rows from before index row_c / 34.0
    print(df.shape)
    df.to_csv('./data/closed_loop_trajectory_TRIMMED.csv', index=False)


