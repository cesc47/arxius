import networkx as nx


def load_graph(file_path):
    """
    Load a graph (tree) from a GEXF file.

    Args:
        file_path (str): Path to the GEXF file.

    Returns:
        networkx.Graph: The loaded graph.
    """
    try:
        graph = nx.read_gexf(file_path)
        print("Graph successfully loaded.")
        return graph
    except Exception as e:
        print(f"Error loading graph: {e}")
        return None


# Example Usage:
if __name__ == "__main__":
    graph_path = '/home/francesc/PycharmProjects/CVC/arxius/categoritzacio/src/simple_ontology_graph.gexf'  # Path to the uploaded file
    graph = load_graph(graph_path)

    if graph:
        print(f"Number of nodes: {graph.number_of_nodes()}")
        print(f"Number of edges: {graph.number_of_edges()}")
        print("Sample nodes:", list(graph.nodes)[:5])



