import torch
import pytorch_lightning as pl
from torchmetrics.classification import BinaryAccuracy
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

class NeuralNet(pl.LightningModule):
    def __init__(self, model):
        super(NeuralNet, self).__init__()
        self.model = model

        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.view(-1, 1))
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_pred.sigmoid(), y.view(-1, 1)))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.view(-1, 1))
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_pred.sigmoid(), y.view(-1, 1)))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def evaluate(self, dataloader, test = False):
        self.eval()  # Set model to evaluation mode
        all_predictions = []
        all_targets = []
    
        # Iterate over batches in the dataloader
        for batch in dataloader:
            inputs, targets = batch
    
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(inputs)
    
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