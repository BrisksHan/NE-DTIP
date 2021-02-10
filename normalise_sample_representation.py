from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
def standardscaler_transform(raw_embedding):
    transformed_data = StandardScaler().fit_transform(raw_embedding)
    return transformed_data

def min_max_0_1_transform(raw_embedding):
    transformed_data = MinMaxScaler(feature_range=(0,1)).fit_transform(raw_embedding)
    return transformed_data

def min_max_0_1_transform(raw_embedding):
    transformed_data = MinMaxScaler(feature_range=(0,1)).fit_transform(raw_embedding)
    return transformed_data

def min_max_1_1_transform(raw_embedding):
    transformed_data = MinMaxScaler(feature_range=(-1,1)).fit_transform(raw_embedding)
    return transformed_data