import os
import yaml
from pathlib import Path
import pandas as pd
import torch
import torch_geometric
from torch_geometric.nn import MetaPath2Vec
from torch_geometric.data import HeteroData
import workers
import sqlalchemy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


ROOT_dir = Path(__file__).parent.parent
with open(os.path.join(ROOT_dir, 'dbs', 'keys.yaml')) as f:
    keys_manager = yaml.load(f, Loader=yaml.FullLoader)

default_metapath = [('individual', 'visits', 'hexagon'),
                    ('hexagon', 'visited_by', 'individual')]

# Data location
user = workers.keys_manager['database']['user']
password = workers.keys_manager['database']['password']
port = workers.keys_manager['database']['port']
db_name = workers.keys_manager['database']['name']
engine = sqlalchemy.create_engine(f'postgresql://{user}:{password}@localhost:{port}/{db_name}?gssencmode=disable')


class CityDataToGraph:
    def __init__(self, data_path=None, data_space_group=None):
        self.data = pd.read_parquet(data_path)
        self.space_group = pd.read_csv(data_space_group)
        self.df_ib = pd.read_sql("""SELECT device_aid, "group" FROM device_group;""", con=engine)
        self.df_ib = self.df_ib[self.df_ib['device_aid'].isin(self.data['device_aid'].unique())]
        self.data_hf = None
        self.graph = None
        self.individuals_mapping = None
        self.h3_mapping = None
        self.poi_mapping = None

    def edges_processing(self, basic=True, space_removed=False):
        self.data = self.data[self.data.device_aid.isin(self.df_ib['device_aid'].unique())]
        print(f"No of edges {len(self.data)} from {self.data['device_aid'].nunique()} unique devices")

        self.individuals_mapping = dict(zip(self.data['device_aid'].unique(), range(0, self.data['device_aid'].nunique())))
        self.h3_mapping = dict(zip(self.data['h3_id'].unique(), range(0, self.data['h3_id'].nunique())))

        if space_removed:
            self.data.loc[:, 'to_explode'] = self.data.loc[:, 'freq_wi'].apply(lambda x: [1]*x)
            self.data = self.data.explode('to_explode')
            self.data.drop(columns=['to_explode', 'freq_wi', 'freq_w'], inplace=True)
        if not basic:
            # Hexagon-Function edges
            df_stops_h = self.data[['h3_id', 'kind']].explode('kind')
            df_stops_h.dropna(inplace=True)
            df_stops_h = df_stops_h.groupby(['h3_id', 'kind']).size().rename('count').reset_index()

            def count_norm(data):
                data['count_n'] = np.ceil(data['count'] / data['count'].sum() * 10)
                return data
            self.data_hf = df_stops_h.groupby('h3_id').\
                apply(lambda data: count_norm(data), include_groups=False).\
                reset_index()
            self.data_hf['count_n'] = self.data_hf['count_n'].astype(int)
            self.data_hf['count_r'] = self.data_hf['count_n'].apply(lambda x: [1] * x)
            self.data_hf = self.data_hf[['h3_id', 'kind', 'count_r']].explode('count_r').drop(columns=['count_r'])
            self.poi_mapping = dict(zip(self.data_hf['kind'].unique(), range(0, self.data_hf['kind'].nunique())))

            self.data_hf.loc[:, 'src_id'] = self.data_hf.loc[:, 'h3_id'].map(self.h3_mapping)
            self.data_hf.loc[:, 'dst_id'] = self.data_hf.loc[:, 'kind'].map(self.poi_mapping)

        # Convert nodes into node indexes
        self.data.loc[:, 'src_id'] = self.data.loc[:, 'device_aid'].map(self.individuals_mapping)
        self.data.loc[:, 'dst_id'] = self.data.loc[:, 'h3_id'].map(self.h3_mapping)
        self.df_ib.loc[:, 'dst_id'] = self.df_ib.loc[:, 'device_aid'].map(self.individuals_mapping)
        self.df_ib.loc[:, 'src_id'] = self.df_ib.loc[:, 'group']

    def hetero_graph_maker(self, basic=True, group_node=False):
        self.graph = HeteroData()
        # Add node features
        self.graph['individual'].y_index = torch.tensor([v for _, v in self.individuals_mapping.items()], dtype=torch.long)
        self.graph['hexagon'].y_index = torch.tensor([v for _, v in self.h3_mapping.items()], dtype=torch.long)

        # Add edge - individual visits hexagon
        edge_index = torch.tensor(self.data[['src_id', 'dst_id']].values.T, dtype=torch.long)
        self.graph['individual', 'visits', 'hexagon'].edge_index = edge_index

        # Add edge - hexagon visited by individual
        edge_index = torch.tensor(self.data[['dst_id', 'src_id']].values.T, dtype=torch.long)
        self.graph['hexagon', 'visited_by', 'individual'].edge_index = edge_index

        if not basic:
            # Add extra nodes
            self.graph['poi'].y_index = torch.tensor([v for _, v in self.poi_mapping.items()], dtype=torch.long)

            # Add edge - hexagon contains poi
            edge_index = torch.tensor(self.data_hf[['src_id', 'dst_id']].values.T, dtype=torch.long)
            self.graph['hexagon', 'contains', 'poi'].edge_index = edge_index

            # Add edge - hexagon contains poi
            edge_index = torch.tensor(self.data_hf[['dst_id', 'src_id']].values.T, dtype=torch.long)
            self.graph['poi', 'located_in', 'hexagon'].edge_index = edge_index

        if group_node:
            # Add extra node
            self.graph['group'].y_index = torch.tensor([v for v in self.df_ib['group'].unique()], dtype=torch.long)

            # Add edge - group includes individual
            edge_index = torch.tensor(self.df_ib[['src_id', 'dst_id']].values.T, dtype=torch.long)
            self.graph['group', 'includes', 'individual'].edge_index = edge_index

            # Add edge - individual belongs_to group
            edge_index = torch.tensor(self.df_ib[['dst_id', 'src_id']].values.T, dtype=torch.long)
            self.graph['individual', 'belongs_to', 'group'].edge_index = edge_index

    def prediction_target(self, individual=True, space=True):
        if individual:
            # For demonstration, we define an integer label for each author node.
            individual_group_dict = self.df_ib.set_index('device_aid')['group'].to_dict()
            individual_labels = [individual_group_dict[k] for k, _ in self.individuals_mapping.items()]
            self.graph["individual"].y = torch.tensor(individual_labels, dtype=torch.int32)

        if space:
            space_group_dict = self.space_group.set_index('h3_id')['group'].to_dict()
            space_labels = [space_group_dict[k] for k, _ in self.h3_mapping.items()]
            self.graph["hexagon"].y = torch.tensor(space_labels, dtype=torch.int32)
        print("Constructed HeteroData object:", self.graph)
        print("Individual labels:", self.graph["individual"].y)
        print("Hexagon labels:", self.graph["hexagon"].y)


class Graph2EmbeddingSpace:
    def __init__(self, graph=None):
        self.graph = graph
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch_geometric.is_xpu_available():
            self.device = torch.device('xpu')
        else:
            self.device = torch.device('cpu')
        print('Device: {}'.format(self.device))
        self.model = None
        self.optimizer = None
        self.loader = None
        self.loss_tracker = dict()

    def model_init(self, embedding_dim=64, walk_length=20, context_size=7,
                   walks_per_node=200, num_negative_samples=5,
                   metapath=default_metapath, batch_size=16):
        self.model = MetaPath2Vec(
            self.graph.edge_index_dict,
            embedding_dim=embedding_dim,  # Smaller dimension for our toy data
            metapath=metapath,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            sparse=True  # Use a sparse embedding for memory efficiency
        ).to(self.device)

        self.optimizer = torch.optim.SparseAdam(list(self.model.parameters()), lr=0.01)
        self.loader = self.model.loader(batch_size=batch_size, shuffle=True, num_workers=0)

    def train_r(self, epoch, log_steps=50, patience=2, min_delta=0.001):
        self.model.train()
        total_loss = 0
        best_loss = float('inf')  # Track the best loss
        epochs_without_improvement = 0  # Counter for early stopping
        self.loss_tracker[epoch] = []
        for i, (pos_rw, neg_rw) in enumerate(self.loader):
            self.optimizer.zero_grad()
            loss = self.model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

            if (i + 1) % log_steps == 0:
                avg_loss = total_loss / log_steps
                print(f'Epoch: {epoch}, Step: {i + 1:03d}/{len(self.loader)}, '
                      f'Loss: {avg_loss:.4f}')
                self.loss_tracker[epoch].append((log_steps, avg_loss))
                total_loss = 0

                # Early stopping check
                if avg_loss < best_loss - min_delta:
                    best_loss = avg_loss
                    epochs_without_improvement = 0  # Reset counter
                else:
                    epochs_without_improvement += 1

                # Stop training if loss hasn't improved for `patience` steps
                if epochs_without_improvement >= patience:
                    print(f'Early stopping at Epoch {epoch}, Step {i + 1}: Loss has not improved for {patience} steps.')
                    return True  # Signal to stop training
        return False  # Continue training

    def prediction_task(self, individual=True):
        self.model.eval()
        if individual:
            z = self.model('individual', batch=self.graph["individual"].y_index.to(self.device)).cpu().detach().numpy()
            y = self.graph["individual"].y.cpu().detach().numpy()
        else:
            z = self.model('hexagon', batch=self.graph["hexagon"].y_index.to(self.device)).cpu().detach().numpy()
            y = self.graph["hexagon"].y.cpu().detach().numpy()

        # Split the data into training and testing sets
        z_train, z_test, y_train, y_test = train_test_split(z, y, test_size=0.2, random_state=42)

        # Initialize the logistic regression model
        logistic_model = LogisticRegression(class_weight='balanced')

        # Train the model on the training data
        logistic_model.fit(z_train, y_train)

        # Predict on the test set
        y_pred = logistic_model.predict(z_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")

        # Print classification report
        c_report = classification_report(y_test, y_pred)
        print("Classification Report:")
        print(c_report)

        # Print confusion matrix
        print("Confusion Matrix:")
        c_matrix = confusion_matrix(y_test, y_pred)
        print(c_matrix)
        return accuracy, c_report, c_matrix


def paths_design(basic=True, group_node=False):
    if basic & (not group_node):
        paths = [
            ('individual', 'visits', 'hexagon'),
            ('hexagon', 'visited_by', 'individual')
        ]
    if basic & group_node:
        paths = [
            ('group', 'includes', 'individual'),
            ('individual', 'visits', 'hexagon'),
            ('hexagon', 'visited_by', 'individual'),
            ('individual', 'belongs_to', 'group'),
        ]
    if (not basic) & (not group_node):
        paths = [
            ('individual', 'visits', 'hexagon'),
            ('hexagon', 'contains', 'poi'),
            ('poi', 'located_in', 'hexagon'),
            ('hexagon', 'visited_by', 'individual')
        ]
    if (not basic) & group_node:
        paths = [
            ('group', 'includes', 'individual'),
            ('individual', 'visits', 'hexagon'),
            ('hexagon', 'contains', 'poi'),
            ('poi', 'located_in', 'hexagon'),
            ('hexagon', 'visited_by', 'individual'),
            ('individual', 'belongs_to', 'group')
        ]
    return paths
