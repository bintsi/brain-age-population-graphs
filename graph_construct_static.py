import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.utils import to_undirected, to_dense_adj ,dense_to_sparse, erdos_renyi_graph
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph

"""
Population graph construction. 
Takes the imaging and non-imaging data and creates the graph that will be used in the network.
"""
class PopulationGraphUKBB:
    def __init__(self, data_dir, filename_train, filename_val, filename_test, phenotype_columns, columns_kept, num_node_features, task, num_classes, k, edges, construction):
        self.data_dir = data_dir
        self.filename_train = filename_train
        self.filename_val = filename_val
        self.filename_test = filename_test
        self.phenotype_columns = phenotype_columns
        self.columns_kept = columns_kept
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.k = k
        self.edges = edges
        self.construction = construction
        
        self.task = task

    def load_data(self):
        """
        Loads the dataframes for the train, val, and test, and returns 1 dataframe for all.
        """

        # Read csvs for train, val, test
        data_df_train = pd.read_csv(self.data_dir + self.filename_train)
        data_df_val = pd.read_csv(self.data_dir+self.filename_val)
        data_df_test = pd.read_csv(self.data_dir+self.filename_test)
        
        # Give labels for classification 
        if self.task == 'classification':        
                frames = [data_df_train, data_df_val, data_df_test]
                df = pd.concat(frames)

                labels = list(range(0,self.num_classes))
                df['Age'] = pd.qcut(df['Age'], q=self.num_classes, labels=labels).astype('int') #Balanced classes
                # df['Age'] = pd.cut(df['Age'], bins=self.num_classes, labels=labels).astype('int') #Not balanced classes
                
                a = data_df_train.shape[0]
                b = data_df_val.shape[0]

                data_df_train = df.iloc[:a, :]
                data_df_val = df.iloc[a:a+b, :]
                data_df_test = df.iloc[a+b:, :]
        
        a = data_df_train.shape[0] 
        b = data_df_train.shape[0]+data_df_val.shape[0] 
        num_nodes = b + data_df_test.shape[0] 

        train_idx = np.arange(0, a, dtype=int)
        val_idx = np.arange(a, b, dtype=int)
        test_idx = np.arange(b, num_nodes, dtype=int)

        frames = [data_df_train, data_df_val, data_df_test] 

        data_df = pd.concat(frames, ignore_index=True)

        return data_df, train_idx, val_idx, test_idx, num_nodes

    def get_phenotypes(self, data_df):
        """
        Takes the dataframe for the train, val, and test, and returns 1 dataframe with only the phenotypes.
        """
        phenotypes_df = data_df[self.phenotype_columns]      

        return phenotypes_df  

    def get_features_demographics(self, phenotypes_df):
        """
        Returns the phenotypes of every node, meaning for every subject. 
        The node features are defined by the non-imaging information
        """
        phenotypes = phenotypes_df.to_numpy()
        phenotypes = torch.from_numpy(phenotypes).float()
        return phenotypes

    def get_node_features(self, data_df):
        """
        Returns the features of every node, meaning for every subject.
        """

        df_node_features = data_df.iloc[:, -self.num_node_features:]
        node_features = df_node_features.to_numpy()
        node_features = torch.from_numpy(node_features).float()
        return node_features

    def get_subject_masks(self, train_index, validate_index, test_index):
        """Returns the boolean masks for the arrays of integer indices.

        Parameters:
        train_index: indices of subjects in the train set.
        validate_index: indices of subjects in the validation set.
        test_index: indices of subjects in the test set.

        Returns:
        a tuple of boolean masks corresponding to the train/validate/test set indices.
        """

        num_subjects = len(train_index) + len(validate_index) + len(test_index)

        train_mask = np.zeros(num_subjects, dtype=bool)
        train_mask[train_index] = True
        train_mask = torch.from_numpy(train_mask)

        validate_mask = np.zeros(num_subjects, dtype=bool)
        validate_mask[validate_index] = True
        validate_mask = torch.from_numpy(validate_mask)

        test_mask = np.zeros(num_subjects, dtype=bool)
        test_mask[test_index] = True
        test_mask = torch.from_numpy(test_mask)

        return train_mask, validate_mask, test_mask

    def get_labels(self, data_df):
        """
        Returns the labels for every node, in our case, age.
        """
        if self.task == 'regression':
            labels = data_df['Age'].values   
            labels = torch.from_numpy(labels).float()
        elif self.task == 'classification':
            labels = data_df['Age'].values 
            print(np.unique(labels, return_counts=True))
            labels = torch.from_numpy(labels)
        else:
            raise ValueError('Task should be either regression or classification.')
        return labels
                        
    def get_edge_list(self, phenotypes_df):
        """
        Finds similarity score among the nodes based on the phenotypes and returns edge list.
        Two nodes are connected if they have more phenotypes similar than a threshold.
        For the categorical phenotypes, we need same value.
        For the continuous ones, we need them to be in a selected range.
        """
        num_nodes = len(phenotypes_df)
        phenotypes = list(phenotypes_df)
        threshold = 18 # hyperparameter
        chosen_range = 0.1 # hyperparameter
                                                                                 
        head_edge = []
        tail_edge = []

        for node_i in range(num_nodes):
            if node_i % 100 == 0:
                print('node_i', node_i)
            for node_j in range(node_i+1, num_nodes):
                count = 0
                for ph in phenotypes:
                    # If phenotype is categorical
                    if ph in ['Sex', 'College education', 'Smoking status', 'Alcohol intake frequency', 'Stroke', 'Diabetes',
                              'Walking per week', 'Moderate per week', 'Vigorous per week']: 
                        if phenotypes_df.loc[node_i, ph] == phenotypes_df.loc[node_j, ph]:
                            count += 1
                    # If phenotype is continuous
                    elif ph in ['Weight', 'Height', 'Body mass index (BMI)', 'Systolic blood pressure', 'Diastolic blood pressure', 
                                'Fluid intelligence', 'Tower rearranging: number of puzzles correct', 
                                'Trail making task: duration to complete numeric path trail 1', 
                                'Trail making task: duration to complete alphanumeric path trail 2', 
                                'Matrix pattern completion: number of puzzles correctly solved', 
                                'Matrix pattern completion: duration spent answering each puzzle']: 
                        if phenotypes_df.loc[node_i, ph] - phenotypes_df.loc[node_j, ph] < chosen_range:
                            count += 1
                    else:
                        raise ValueError('Include phenotype in one of the categories: categorical or continuous.')
                if count > threshold: 
                    head_edge.append(node_i)
                    tail_edge.append(node_j)
                    head_edge.append(node_j)
                    tail_edge.append(node_i)
        edge_list = torch.tensor([head_edge, tail_edge], dtype = torch.long)
        return edge_list
    
    def get_similarity_matrix_phenotypes(self, phenotypes_df):
        """
        Finds similarity among the nodes and returns similarity matrix based on the phenotypes
        I need this function for the get_weighted_similarity_matrix one.
        Similar to kamilest

        Inputs:
        phenotypes_df: dataframe with phenotypes

        Returns:
        similarities_norm: normalised similarity matrix 
        """
        num_nodes = len(phenotypes_df)
        phenotypes = list(phenotypes_df)
        threshold = 18 # hyperparameter
        chosen_range = 0.1 # hyperparameter
        similarities = np.zeros((num_nodes, num_nodes), dtype=np.float32)
                                                                          
        for node_i in range(num_nodes):
            if node_i % 100 == 0:
                print('node_i', node_i)
            for node_j in range(node_i+1, num_nodes):
                for ph in phenotypes:
                    # If phenotype is categorical
                    if ph in ['Sex', 'College education', 'Smoking status', 'Alcohol intake frequency', 'Stroke', 'Diabetes',
                              'Walking per week', 'Moderate per week', 'Vigorous per week']: 
                        if phenotypes_df.loc[node_i, ph] == phenotypes_df.loc[node_j, ph]:
                            similarities[node_i, node_j] +=1
                            similarities[node_j, node_i] +=1
                    # If phenotype is continuous
                    elif ph in ['Weight', 'Height', 'Body mass index (BMI)', 'Systolic blood pressure', 'Diastolic blood pressure', 
                                'Fluid intelligence', 'Tower rearranging: number of puzzles correct', 
                                'Trail making task: duration to complete numeric path trail 1', 
                                'Trail making task: duration to complete alphanumeric path trail 2', 
                                'Matrix pattern completion: number of puzzles correctly solved', 
                                'Matrix pattern completion: duration spent answering each puzzle']: 
                        if phenotypes_df.loc[node_i, ph] - phenotypes_df.loc[node_j, ph] < chosen_range:
                            similarities[node_i, node_j] +=1
                            similarities[node_j, node_i] +=1
                    else:
                        raise ValueError('Include phenotype in one of the categories: categorical or continuous.')
        similarities_norm = (similarities-similarities.min())/(similarities.max()-similarities.min())
        similarities_norm = torch.from_numpy(similarities_norm)
        return similarities_norm                   
    
    def get_similarity_matrix_cosine_similarity(self, node_features):
        """
        Finds similarity score among the nodes based on the node features using cosine similarity and returns edge list based on cosine similarity.
        I need this function for the get_weighted_similarity_matrix one.

        Inputs:
        node_features: features that will be used as node features

        Returns:
        adj[0]: adjacency matrix of size [node_features, node_features], where 0 if [node a, node b] not connected and 1 if connected.
        """
        num_nodes = node_features.shape[0]
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        similarities = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        for node_i in range(num_nodes):
            if node_i % 100 == 0:
                print('node_i', node_i)
            for node_j in range(node_i+1, num_nodes):
                sim_score = cos(node_features[node_i], node_features[node_j])
                similarities[node_i, node_j] = sim_score
                similarities[node_j, node_i] = sim_score

        similarities = torch.from_numpy(similarities)
        return similarities
  
    def get_edges_using_KNNgraph(self, dataset, k):
        """
        Extracts edge index based on the cosine similarity of the node features.
        Uses the existing pyg class KNNGraph.

        Inputs:
        dataset: the population graph (without edge_index).
        k: number of edges that will be kept for every node.

        Returns: 
        dataset: graph dataset with the acquired edges.
        """
        if self.edges == 'phenotypes':
            dataset.pos = dataset.phenotypes   
        elif self.edges == 'imaging':
            dataset.pos = dataset.x  
        else:
            raise ValueError('Choose appropriate edge connection.')

        dataset.cuda()
        dataset = KNNGraph(k=k, force_undirected=True, cosine=True)(dataset)
        dataset.to('cpu')
        dataset = Data(x = dataset.x, y = dataset.y, phenotypes = dataset.phenotypes, train_mask=dataset.train_mask, val_mask= dataset.val_mask, test_mask=dataset.test_mask, edge_index=dataset.edge_index, num_nodes=dataset.num_nodes)
        return dataset
    
    def get_weighted_similarity_matrix(self, phenotypes_df, node_features, top_k):
        sim_matrix_phenotypes = self.get_similarity_matrix_phenotypes(phenotypes_df)
        sim_matrix_cosine_similarity = self.get_similarity_matrix_cosine_similarity(node_features)
        
        weighted_sim_matrix = sim_matrix_phenotypes*sim_matrix_cosine_similarity

        edge_list = torch.stack([torch.repeat_interleave(torch.arange(weighted_sim_matrix.shape[1]),top_k), weighted_sim_matrix.topk(k=top_k, dim=-1).indices.view(-1)])
        edge_list = to_undirected(edge_list)

        edge_index = edge_list.type(torch.LongTensor) 
        return edge_index, None

    def get_random_graph(self, num_nodes):
        edge_list = erdos_renyi_graph(num_nodes, 0.002)
        return edge_list
  
    def get_population_graph(self):
        """
        Creates the population graph.
        """
        # Load data
        data_df, train_idx, val_idx, test_idx, num_nodes = self.load_data()

        # Take phenotypes and node_features dataframes
        phenotypes_df = self.get_phenotypes(data_df)
        phenotypes = self.get_features_demographics(phenotypes_df)
        node_features = self.get_node_features(data_df) 

        # Mask val & test subjects
        train_mask, val_mask, test_mask = self.get_subject_masks(train_idx, val_idx, test_idx)
        # Get the labels
        labels = self.get_labels(data_df) 

        if  self.task == 'classification':
            labels= one_hot_embedding(labels,abs(self.num_classes)) 

        # Connect the edges (use different function for different ways of connection)
        if self.construction == 'kamilest':
            ## For Kamilė Stankevičiūtė' (Population Graph GNNs for Brain Age Prediction)
            edge_list = self.get_edge_list(phenotypes_df) 
            population_graph = Data(x = node_features, edge_index= edge_list, phenotypes=phenotypes, y= labels, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes)        
        elif self.construction == 'parisot':
            ## For Sarah Parisot (Disease prediction using graph convolutional networks: application to autism spectrum disorder and Alzheimer’s disease)
            edge_list, edge_weights = self.get_weighted_similarity_matrix(phenotypes_df, node_features, top_k=5)
            population_graph = Data(x = node_features, edge_index= edge_list, edge_attr=edge_weights, phenotypes=phenotypes, y= labels, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes)
        elif self.construction == 'random':
            # For random graph
            edge_list = self.get_random_graph(num_nodes)
            population_graph = Data(x = node_features, edge_index= edge_list, phenotypes=phenotypes, y= labels, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes)        
        elif self.construction == 'only phenotypes':
            # If I want phenotypes for node feautures and for connecting edges
            population_graph = Data(x = phenotypes, y= labels, phenotypes= phenotypes, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes)        
            population_graph = self.get_edges_using_KNNgraph(population_graph, k=self.k)
        elif self.construction == 'reversed':
            # If I want phenotypes for node feautures and imaging features for connecting edges
            population_graph = Data(x = phenotypes, y= labels, phenotypes= node_features, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes)        
            population_graph = self.get_edges_using_KNNgraph(population_graph, k=self.k)
        elif self.construction == 'knn':  
            # Get edges using existing pyg KNNGraph class
            population_graph = Data(x = node_features, y= labels, phenotypes= phenotypes, train_mask= train_mask, val_mask=val_mask, test_mask=test_mask, num_nodes=num_nodes, k=self.k)        
            population_graph = self.get_edges_using_KNNgraph(population_graph, k=self.k)
        else:
            raise ValueError('Choose appropriate graph construction method.')   
        return population_graph

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes) 
    return y[labels] 

class UKBBageDataset(torch.utils.data.Dataset):
    def __init__(self, graph, split='train', samples_per_epoch=100, device='cpu', num_classes=2) -> None:
        dataset = graph
        self.n_features = dataset.num_node_features
        self.num_classes = abs(num_classes)
        self.X = dataset.x.float().to(device)
        self.y = dataset.y.float().to(device)
        self.phenotypes = dataset.phenotypes.float().to(device)
        self.edge_index = dataset.edge_index.to(device)

        if split=='train':
            self.mask = dataset.train_mask.to(device)
        if split=='val':
            self.mask = dataset.val_mask.to(device)
        if split=='test':
            self.mask = dataset.test_mask.to(device)
         
        self.samples_per_epoch = samples_per_epoch
    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self.X,self.y,self.mask,self.phenotypes, self.edge_index