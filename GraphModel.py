import torch
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_lightning as pl
import torchmetrics 

import GNN_models

def select_model(model_name, num_node_features, n_conv_layers, layer_sizes, dropout_p):
    if model_name == 'mlp':
        return GNN_models.MLP()
    elif model_name=='gin_simple':
        return GNN_models.GIN()
    elif model_name == 'gcn':
        print('gcn model')
        return GNN_models.BrainGCN(num_node_features, n_conv_layers, layer_sizes, dropout_p)
    elif model_name == 'gat':
        print('gat model')
        return GNN_models.BrainGAT(num_node_features, n_conv_layers, layer_sizes, dropout_p)
    elif model_name == 'sage':
        print('sage model')
        return GNN_models.BrainSage(num_node_features, n_conv_layers, layer_sizes, dropout_p)
    elif model_name == 'graphconv':
        print('graphconv model')
        return GNN_models.GraphConv(num_node_features, n_conv_layers, layer_sizes, dropout_p)
    elif model_name == 'cheb':
        print('chebconv model')
        return GNN_models.BrainCheb(num_node_features, n_conv_layers, layer_sizes, dropout_p)
    else:
        print('Invalid or no model selected: {}'.format(model_name))
        quit()     

class GraphModel_train_val_loaders(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.layer_sizes =self.hparams.layer_sizes
        self.n_conv_layers = self.hparams.n_conv_layers
        self.dropout_p = self.hparams.dropout_p
        self.num_node_features = self.hparams.num_node_features
        self.model_name = self.hparams.model_name

        self.model = select_model(self.model_name, self.num_node_features, self.n_conv_layers, self.layer_sizes, self.dropout_p)
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay

        self.task = self.hparams.task
        self.batch_size = 1
        self.num_classes = self.hparams.num_classes

        if self.task == "regression":
            self.criterion = torch.nn.HuberLoss() 
            
            # Metrics for regression
            self.mean_absolute_error = torchmetrics.MeanAbsoluteError()
            self.mean_squared_error = torchmetrics.MeanSquaredError()
            self.rscore = torchmetrics.PearsonCorrCoef()
            self.r2score = torchmetrics.R2Score()

        elif self.task == "classification":
            self.criterion= torch.nn.CrossEntropyLoss() 
            # self.criterion = torch.nn.BCEWithLogitsLoss()

            # Metrics for classification
            self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
            self.auc = torchmetrics.AUROC(task="multiclass", num_classes=self.num_classes)
            self.f1score = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)#, average='macro')
            self.sensitivity = torchmetrics.Recall(task="multiclass", num_classes=self.num_classes)#, average='macro')
            self.specificity = torchmetrics.Specificity(task="multiclass", num_classes=self.num_classes)#, average='macro')
        else:
            raise ValueError('Task should be either regression or classification.')

    def forward(self, data, mode='train'):
        x, y, mask, phenotypes, edge_index = data
        edge_index = edge_index[0]
        
        assert(x.shape[0]==1) #only works in transductive setting
        mask=mask[0]
        y = y[0]
        x = x[0]

        if self.task == 'regression':
            x = self.model(x, edge_index).squeeze_()
        elif self.task == 'classification':
            x = self.model(x, edge_index)
        else:
            raise ValueError('Task should be either regression or classification.')
        
        if self.task == 'regression':
            loss = self.criterion(x[mask].squeeze_(), y[mask].squeeze_())
            abs_error = abs(x[mask].squeeze_() - y[mask].squeeze_()).mean()

            mean_absolute_error = self.mean_absolute_error(x[mask].squeeze_(), y[mask].squeeze_())
            mean_squared_error = self.mean_squared_error(x[mask].squeeze_(), y[mask].squeeze_())
            rscore = self.rscore(x[mask].squeeze_(),y[mask].squeeze_())
            r2score = self.r2score(x[mask].squeeze_(),y[mask].squeeze_())

            return loss, abs_error, mean_absolute_error, mean_squared_error, rscore, r2score
        elif self.task == "classification":
            loss = self.criterion(x[mask], y[mask])
            acc = (x[mask].argmax(dim=-1) == y[mask].argmax(dim=-1)).sum().float() / mask.sum()

            accuracy = self.accuracy(x[mask].argmax(dim=-1), y[mask].argmax(dim=-1))
            auc = self.auc(x[mask], y[mask].argmax(dim=-1))
            f1score = self.f1score(x[mask].argmax(dim=-1), y[mask].argmax(dim=-1))
            sensitivity = self.sensitivity(x[mask].argmax(dim=-1), y[mask].argmax(dim=-1))
            specificity = self.specificity(x[mask].argmax(dim=-1), y[mask].argmax(dim=-1))

            return loss, acc, accuracy, auc, f1score, sensitivity, specificity
        else:
            raise ValueError('Task should be either regression or classification.')       
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
            
    def training_step(self, batch, batch_idx):
        if self.task == "regression":
            loss, abs_error, mean_absolute_error, mean_squared_error, rscore, r2score = self.forward(batch, mode="train")
            self.log('train_loss', loss)
            self.log('train_abs_error', abs_error)         
            return loss
        elif self.task == "classification":
            loss, acc, accuracy, auc, f1_score, sensitivity, specificity = self.forward(batch, mode="train")
            self.log('train_loss', loss)
            self.log('train_acc', 100*acc)
            return loss
        else:
            raise ValueError('Task should be either regression or classification.')

    def validation_step(self, batch, batch_idx):
        if self.task == "regression":
            loss, abs_error, mean_absolute_error, mean_squared_error, rscore, r2score = self.forward(batch, mode='val')
            self.log('val_loss', loss)
            self.log('val_abs_error', abs_error)

            self.log('val_mean_absolute_error', mean_absolute_error)
            self.log('val_mean_squared_error', mean_squared_error)
            self.log('val_rscore', rscore)
            self.log('val_r2score', r2score)

        elif self.task == "classification":
            loss, acc, accuracy, auc, f1_score, sensitivity, specificity = self.forward(batch, mode="val")
            self.log('val_loss', loss)
            self.log('val_acc', 100*acc)

            self.log('val_accuracy', accuracy)
            self.log('val_AUC', auc)
            self.log('val_f1score', f1_score)
            self.log('val_sensitivity', sensitivity)
            self.log('val_specificity', specificity)
        else:
            raise ValueError('Task should be either regression or classification.')
    
    def test_step(self, batch, batch_idx):
        if self.task == 'regression':
            loss, abs_error, mean_absolute_error, mean_squared_error, rscore, r2score = self.forward(batch, mode='test')
            self.log('test_loss', loss)
            self.log('test_abs_error', abs_error)

            self.log('test_mean_absolute_error', mean_absolute_error)
            self.log('test_mean_squared_error', mean_squared_error)
            self.log('test_rscore', rscore)
            self.log('test_r2score', r2score)

        elif self.task == "classification":
            loss, acc, accuracy, auc,  f1_score, sensitivity, specificity = self.forward(batch, mode="test")
            self.log('test_loss', loss)
            self.log('test_acc', 100*acc)

            self.log('test_accuracy', accuracy)
            self.log('test_AUC', auc)
            self.log('test_f1score', f1_score)
            self.log('test_sensitivity', sensitivity)
            self.log('test_specificity', specificity)

        else:
            raise ValueError('Task should be either regression or classification.')