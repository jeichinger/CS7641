import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer

'''
Standardizes the data and encodes categorical attributes as one hot.
'''
def preprocess_data(data):

    pipe_transformers = []

    numeric_features = list(data.select_dtypes(include=np.number))

    if len(numeric_features) != 0:
        pipe_transformers.append(('standard_scalar', StandardScaler(), numeric_features))

    cat_features = list(data.select_dtypes(exclude=np.number))

    if len(cat_features) != 0:
        pipe_transformers.append(("cat", OneHotEncoder(), cat_features))

    # class_encoder = LabelEncoder()
    # class_encoder.fit(["e", "p"])
    # encoded_classes = class_encoder.transform(dataset_1["class"])

    # dataset_1["class"] = encoded_classes

    # dataset_1.to_csv(os.path.join(DATA_PATH, "mushrooms_encoded.csv"))

    pipe = ColumnTransformer(pipe_transformers, sparse_threshold=0)

    return pipe