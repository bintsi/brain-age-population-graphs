from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch_geometric.loader import DataLoader
import json

import sys
from graph_construct_static import PopulationGraphUKBB, UKBBageDataset

from GraphModel import GraphModel_train_val_loaders

params = {
    "gpus":'-1',
    "epochs": 10,
    
    "layer_sizes": [512, 128, 1],
    "n_conv_layers": 1, 
    "dropout_p": 0.01, 
    
    "learning_rate": 0.001,
    "weight_decay": 5e-4,
    
    "num_node_features": 68, #68 if you want only imaging,  88 if you want both imaging and non imaging
    "num_classes": 1,
    "task": 'regression',
    "model_name": 'gcn',
    
    "k": 5,
    'edges': 'phenotypes',
    'construction': 'parisot',

    "phenotype_columns": ['Sex', 'Weight', 'Height', 'Body mass index (BMI)', 'Systolic blood pressure', 'Diastolic blood pressure',
                        'College education', 'Smoking status', 'Alcohol intake frequency', 'Stroke', 'Diabetes', 'Walking per week', 'Moderate per week',
                        'Vigorous per week', 'Fluid intelligence', 'Tower rearranging: number of puzzles correct', 'Trail making task: duration to complete numeric path trail 1', 
                        'Trail making task: duration to complete alphanumeric path trail 2', 'Matrix pattern completion: number of puzzles correctly solved', 
                        'Matrix pattern completion: duration spent answering each puzzle'],
}

def costruct_graph(params):
    # Load graph

    # We have selected imaging features + non-imaging features 
    # that are found relevant to brain-age from J.Cole's paper
    # https://pubmed.ncbi.nlm.nih.gov/32380363/
    data_dir = 'data/'
    filename_train = 'train.csv' 
    filename_val = 'val.csv' 
    filename_test = 'test.csv'

    node_columns = [0, 1, 22, 90]
    num_node_features = params.num_node_features
    task = params.task
    num_classes = params.num_classes
    k = params.k
    edges = params.edges
    construction = params.construction

    #Phenotypes chosen for the extraction of the edges
    phenotype_columns = params.phenotype_columns

    static_graph = PopulationGraphUKBB(data_dir, filename_train, filename_val, filename_test, phenotype_columns, node_columns, 
                            num_node_features, task, num_classes, k, edges, construction)
    static_graph = static_graph.get_population_graph()
    return static_graph

def main(params, graph):
    
    print('Active CUDA Device: GPU', torch.cuda.current_device())
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    train_data = UKBBageDataset(graph=graph, split='train', device='cuda', num_classes = params.num_classes)
    val_data = UKBBageDataset(graph=graph, split='val', samples_per_epoch=1, num_classes = params.num_classes)
    test_data = UKBBageDataset(graph=graph, split='test', samples_per_epoch=1, num_classes = params.num_classes)
        
    train_loader = DataLoader(train_data, batch_size=1,num_workers=0)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)

    # Logger 
    logger = TensorBoardLogger('logs/', name=("UKBBGraph" + params.model_name))
    
    if params.task == 'regression':
        checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min'
        )
        callbacks = [checkpoint_callback]
        trainer = pl.Trainer(callbacks=callbacks,
                             gpus=params.gpus,
                             max_epochs=params.epochs,
                             progress_bar_refresh_rate=1, 
                             check_val_every_n_epoch = 1,
                             log_every_n_steps=1,
                             logger=logger) 
    elif params.task == 'classification':
        checkpoint_callback = ModelCheckpoint(
            save_last=False,
            save_top_k=1,
            verbose=False,
            monitor='val_loss',
            mode='min')
        callbacks = [checkpoint_callback]
        trainer = pl.Trainer(callbacks=callbacks,
                             gpus=params.gpus,
                             max_epochs=params.epochs,
                             progress_bar_refresh_rate=1,
                             check_val_every_n_epoch = 1,
                             log_every_n_steps=1,
                             logger=logger) 
    else:
        raise ValueError('Task should be either regression or classification.')                

    model = GraphModel_train_val_loaders(params)
    trainer.fit(model, train_loader, val_loader)
    # Evaluate results on validation and test set
    val_results = trainer.validate(ckpt_path=checkpoint_callback.best_model_path, dataloaders=val_loader)
    test_results = trainer.test(ckpt_path=checkpoint_callback.best_model_path, dataloaders=test_loader)

    # Save results
    path = '/'.join(checkpoint_callback.best_model_path.split("/")[:-1])
    with open(path + "/val_results.json", "w") as outfile:
        json.dump(val_results, outfile)

    with open(path + "/test_results.json", "w") as outfile:
        json.dump(test_results, outfile)

    return val_results, test_results, path

if type(params) is not Namespace:
    params = Namespace(**params)

static_graph = costruct_graph(params)
val_results, test_results, best_model_checkpoint_path = main(params, static_graph)