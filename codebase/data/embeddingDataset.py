from torch.utils.data import Dataset

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder

class EmbeddingDataset(Dataset):
    """Adult Income Dataset for embeddings."""
    def __init__(self, dataframe, 
                 y, 
                 embedding_variables = None, 
                 numerical_pipeline = None,
                 preprocessor = None
                ):
        """
        Args:
            dataframe (pandas.DataFrame): DataFrame containing the data.
            y (pandas.Series): Series containing the labels.
        """        
        # Identify categorical features
        self.categorical_features = dataframe.select_dtypes(include=['object']).columns.tolist()
        # Identify numerical features
        self.numerical_features = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()

        if preprocessor is None:
            # Create the preprocessing pipelines for both numerical and categorical data
            if numerical_pipeline is None:
                numerical_pipeline = Pipeline(steps=[
                    ('scaler', MinMaxScaler())
                ])
            
            categorical_embedding = Pipeline(steps=[
                ('encoder', OrdinalEncoder())
            ])
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_embedding, categorical_features)
                ])
            
            output_array = self.preprocessor.fit_transform(dataframe)
        else:
            self.preprocessor = preprocessor
            output_array = preprocessor.transform(dataframe)
        self.dataframe = pd.DataFrame(output_array, columns=self.preprocessor.get_feature_names_out())
        self.y = y

        #Renaming as output features
        self.categorical_features = [ "cat__"+feature for feature in self.categorical_features]
        self.numerical_features = [ "num__"+feature for feature in self.numerical_features]
        
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe[self.numerical_features].iloc[idx]
        row_cat = self.dataframe[self.categorical_features].iloc[idx]

        cat_gestures = {}
        
        for index_cat, cat_feature in enumerate(self.categorical_features):
            cat_gestures[cat_feature] = torch.tensor(row_cat.iloc[index_cat], dtype=torch.long)
        
        cont_features = torch.tensor(row.values, dtype=torch.float)
        y = torch.tensor(self.y.iloc[idx], dtype=torch.float)
        return cont_features, cat_gestures, y

def embedding_collate_fn(batch):
    """
    Custom collate function for datasets where categorical variables are 
    to be embedded, and continuous variables are passed directly.
    
    Args:
    - batch (list of tuples): The input samples as a list where each item is 
      (continuous_features, categorical_features_dict, label).
    
    Returns:
    - Tuple of (continuous_features, categorical_features, labels), where:
      * continuous_features is a tensor containing all continuous features of the batch.
      * categorical_features is a dictionary of tensors, each representing a batch for 
        each categorical feature ready for embedding.
      * labels is a tensor containing all labels of the batch.
    """
    continuous_features, categorical_dicts, labels = zip(*batch)
    
    # Stacking continuous features
    cont_features_batch = torch.stack(continuous_features)
    
    # Handling categorical features, we expect each element in categorical_dicts to be a dictionary
    categorical_features_batch = {key: torch.stack([d[key] for d in categorical_dicts]) for key in categorical_dicts[0]}
    
    # Stacking labels
    labels_batch = torch.tensor(labels, dtype=torch.float32)
    
    return cont_features_batch, categorical_features_batch, labels_batch