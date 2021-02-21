import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "/Data"

def main():

    dataset_1_csv_path = os.path.join(DATA_PATH, "diabetic.csv")
    dataset_1 = pd.read_csv(dataset_1_csv_path)

    print(dataset_1["class"].value_counts())
    

    #dataset_1 = dataset_1.sample(frac=1)

    #n = min(500, dataset_1["class"].value_counts().min())
    #df_ = dataset_1.groupby("class").apply(lambda x: x.sample(n))
    #df_.index = df_.index.droplevel(0)

    #df_ = df_.sample(frac=1)
    #df_["class"].hist()

    #df_.to_csv(os.path.join(DATA_PATH, "cmc_processed_subsample.csv"))

    #plt.show()

if __name__ == "__main__":

    main()
