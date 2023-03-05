from typing import List, Optional
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


def visualize_graph(graphs_indexes: List[List[int]], coordinates: pd.DataFrame, save_path: Optional[str] = None):
    """
    Visualize graph
    :param graphs_indexes: list of lists of indexes of nodes in each graph
    :param coordinates: nodes coordinates
    :param save_path: path to save the plot
    :return:
    """
    G = nx.Graph()
    G.add_nodes_from(range(len(coordinates)))
    for graph_indexes in graphs_indexes:
        G.add_edges_from([(graph_indexes[i], graph_indexes[i+1]) for i in range(len(graph_indexes)-1)])
        G.add_edges_from([(graph_indexes[-1], graph_indexes[0])])

    pos = {i: (coordinates.iloc[i]['x'], coordinates.iloc[i]['y']) for i in range(len(coordinates))}
    colors = ['red' if u in graphs_indexes[0] else 'blue' for u, v in G.edges()]

    plt.figure(figsize=(20, 20))
    nx.draw(G, pos, edge_color=colors, with_labels=True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
