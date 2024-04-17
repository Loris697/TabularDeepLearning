import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

class EmbeddingModel(pl.LightningModule):
    """ 
    General neural network for tabular data that is able to handle categorical features with embeddings 
    This class needs an object (embeddings) that contains the incex of the categorical variables (index) and the cardinality
    of the values (num_values).
    The categorical features must be encoded 0 to num_values-1.
    """
    def __init__(self, 
                 ModelClass, 
                 input_size, 
                 embeddings = {}, 
                 embedding_size = 4,
                 device = torch.device('cuda:0')
                ):
        super(EmbeddingModel, self).__init__()
        self.input_size = input_size
        self.categotical_features = []
        self.embeddings_layer = {}
        self.device_to_use = device
        self.accuracy = BinaryAccuracy()

        ## creating the embeddings layers
        for embedding in embeddings:
            self.categotical_features.append(embedding)
            self.embeddings_layer[embedding] = torch.nn.Embedding(embeddings[embedding], embedding_size).to(self.device_to_use)
            self.input_size = self.input_size + embedding_size
        
        self.model = ModelClass(self.input_size)

    def forward(self, x, x_categorical):
        x = x.to(self.device_to_use)

        for x_cat in x_categorical:
            x_categorical[x_cat] = x_categorical[x_cat].to(self.device_to_use)
        
        #if there is no batch I create it
        if x.dim() <= 1:
            x = x.unsqueeze(0)

        x = self.embedCategoricalVariable(x, x_categorical)

        self.model = self.model.to(self.device_to_use)
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, x_categorical, y = batch

        #if there is no batch I create it
        if x.dim() <= 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = self.embedCategoricalVariable(x, x_categorical)
        
        y_pred = self.model(x).reshape(-1)
        
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_pred.sigmoid(), y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_categorical, y = batch

        #if there is no batch create it
        if x.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        x = self.embedCategoricalVariable(x, x_categorical)
            
        y_pred = self.model(x).reshape(-1)
        
        loss = F.binary_cross_entropy_with_logits(y_pred, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_pred.sigmoid(), y))
        return loss

    def evaluate(self, dataloader, test = False):
        self.eval()  # Set model to evaluation mode
        all_predictions = []
        all_targets = []
    
        # Iterate over batches in the dataloader
        for batch in dataloader:
            inputs, cat_features, targets = batch
    
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(inputs, cat_features)
    
            # Convert logits to probabilities (assuming sigmoid activation for binary classification)
            probabilities = torch.sigmoid(outputs)
    
            # Convert probabilities to binary predictions
            predicted_labels = (probabilities > 0.5).float()
    
            # Append predictions and targets to lists
            all_predictions.extend(predicted_labels.tolist())
            all_targets.extend(targets.tolist())
    
        # Convert predictions and targets to tensors
        predictions_tensor = torch.tensor(all_predictions)
        targets_tensor = torch.tensor(all_targets)
    
        # Calculate accuracy
        accuracy = accuracy_score(targets_tensor, predictions_tensor)
        
        if test:
            # Print classification report
            print(classification_report(targets_tensor, predictions_tensor))
    
        return accuracy

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def embedCategoricalVariable(self, x, x_categorical):
        for cat_feature in self.categotical_features:
            x_cat = x_categorical[cat_feature]
            if x_cat.dim() < 1:
                x_cat = x_cat.unsqueeze(0)
            x_feature = self.embeddings_layer[cat_feature](x_cat).squeeze(1)
            x = torch.cat((x, x_feature), 1)

        return x