import sys
from pathlib import Path
import os
import pandas as pd
import pickle
from pprint import pprint
import torch
import gc

ROOT_dir = Path(__file__).parent.parent.parent
sys.path.append(ROOT_dir)
sys.path.insert(0, os.path.join(ROOT_dir, 'lib'))

import graph_workers


def results_save(graph_embedding=None, city_graph=None):
    # Extract individual embeddings, groups, indexes
    z = graph_embedding.model('individual',
                              batch=graph_embedding.graph["individual"].y_index.to(graph_embedding.device)).\
        cpu().detach().numpy()
    y = graph_embedding.graph["individual"].y_index.cpu().detach().numpy()
    g = graph_embedding.graph["individual"].y.cpu().detach().numpy()
    i_reverse_mapping = {v: k for k, v in city_graph.individuals_mapping.items()}
    df_res = pd.DataFrame(z, columns=[f'x{i}' for i in range(z.shape[1])])
    df_res.loc[:, 'y'] = y
    df_res.loc[:, 'device_aid'] = df_res.loc[:, 'y'].apply(lambda x: i_reverse_mapping[x])
    df_res.loc[:, 'group'] = g

    # Extract hexagon embeddings, groups, indexes
    z = graph_embedding.model('hexagon', batch=graph_embedding.graph["hexagon"].y_index.to(ge.device)).\
        cpu().detach().numpy()
    y = graph_embedding.graph["hexagon"].y_index.cpu().detach().numpy()
    g = graph_embedding.graph["hexagon"].y.cpu().detach().numpy()
    h_reverse_mapping = {v: k for k, v in city_graph.h3_mapping.items()}

    df_resh = pd.DataFrame(z, columns=[f'x{i}' for i in range(z.shape[1])])
    df_resh.loc[:, 'y'] = y
    df_resh.loc[:, 'h3_id'] = df_resh.loc[:, 'y'].apply(lambda x: h_reverse_mapping[x])
    df_resh.loc[:, 'group'] = g
    return df_res, df_resh


if __name__ == '__main__':
    cdtg = graph_workers.CityDataToGraph(data_path='dbs/cities/stockholm_w.parquet',
                                         data_space_group='dbs/cities/stockholm_space_group.csv')
    cdtg.edges_processing(basic=True, space_removed=True)  # set_id=10, it means de-distance baseline
    # Baseline parameter set (alter number of walks and dimensions)
    # Alter dimensions to start with, 64, 128
    parameter_set = {'walks_per_node': 100,
                      'embedding_dim': 64,
                      'walk_length': 40,
                      'context_size': 7,
                      'num_negative_samples': 5}
    # set_id = 0
    bg_set = [(True, False), (True, True), (False, False), (False, True)]  # (True, False), (False, False) are done
    bg, set_id = bg_set[0], 9  # set_id=9, it means de-distance baseline
    basic, group_node = bg
    para = parameter_set.copy()
    para['basic'] = basic
    para['group_node'] = group_node
    # Generate graphs and prediction targets
    cdtg.hetero_graph_maker(basic=para['basic'], group_node=para['group_node'])
    # Always predict individual and hexagon labels
    cdtg.prediction_target(individual=True, space=True)

    # Define metapaths
    paths = graph_workers.paths_design(basic=para['basic'], group_node=para['group_node'])
    print(paths)
    # for walks_per_node in [150, 200]:   # 100, 200, 400 | 64,
        #para['walks_per_node'] = walks_per_node
    pprint(f"Set {set_id + 1}")
    pprint(para)
    # Initialize the model
    ge = graph_workers.Graph2EmbeddingSpace(graph=cdtg.graph)
    ge.model_init(walks_per_node=para['walks_per_node'],
                  embedding_dim=para['embedding_dim'],
                  walk_length=para['walk_length'],
                  context_size=para['context_size'],
                  num_negative_samples=para['num_negative_samples'],
                  metapath=paths)

    # Training loop
    max_epochs = 6
    for epoch in range(max_epochs):
        should_stop = ge.train_r(epoch, log_steps=50, patience=5, min_delta=0.0005)
        if should_stop:
            break

    set_id += 1
    # Prediction task and save results
    result = para.copy()
    print("Predict individual labels:")
    result['accuracy_i'], result['c_report_i'], result['c_matrix_i'] = ge.prediction_task(individual=True)

    print("Predict hexagon's presence of transit stations:")
    result['accuracy_h'], result['c_report_h'], result['c_matrix_h'] = ge.prediction_task(individual=False)
    with open(os.path.join(ROOT_dir, f'dbs/embeddings/result_set{set_id}.pickle'), 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save embeddings
    print(para)
    df_res, df_resh = results_save(graph_embedding=ge, city_graph=cdtg)

    df_res.to_parquet(os.path.join(ROOT_dir, f'dbs/embeddings/baseline_set{set_id}_individual.parquet'), index=False)
    df_resh.to_parquet(os.path.join(ROOT_dir, f'dbs/embeddings/baseline_set{set_id}_hexagon.parquet'), index=False)

    # Save loss time history
    result_list = [item for sublist in ge.loss_tracker.values() for item in sublist]
    df_loss = pd.DataFrame(result_list, columns=['iter', 'loss'])
    df_loss.loc[:, 'step'] = df_loss.index
    df_loss.to_parquet(os.path.join(ROOT_dir, f'dbs/embeddings/baseline_set{set_id}_loss.parquet'), index=False)

    # Clear memory
    # End of current training run: clear GPU memory
    del ge  # Delete objects
    gc.collect()  # Run garbage collection
    torch.cuda.empty_cache()  # Free cached memory on GPU
