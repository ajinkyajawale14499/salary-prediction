import pandas as pd
import pickle

csv_file_path = 'data/final_salary_prediction.csv' 
df = pd.read_csv(csv_file_path)

pickle_file_path = 'final_salary_prediction.pkl'
with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(df, pickle_file)
