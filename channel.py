import networkx as nx
import pandas as pd
import torch
data_df = pd.read_csv("C:/Users/85433/Desktop/demo/datadata/esca_log.csv")
gene_data = data_df.iloc[2:, 0].tolist()
print(gene_data)
pathway_file = 'gene_relationship_ke.csv'
pathway_df = pd.read_csv(pathway_file, header=None)
pathway_data = pathway_df.values.tolist()
gene_id_map = {gene: idx for idx, gene in enumerate(gene_data)}
G = nx.Graph()
for pathway in pathway_data:
    source = pathway[0]
    target = pathway[1]
    if source in gene_data and target in gene_data:
        G.add_edge(gene_id_map[source], gene_id_map[target])

edge_index = torch.tensor(list(G.edges())).t().contiguous()
print(edge_index)
edge_df = pd.DataFrame(edge_index.numpy().transpose())
edge_file = 'edge_index.csv'
edge_df.to_csv(edge_file,header=False, index=False)
