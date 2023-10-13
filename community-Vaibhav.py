import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 
from collections import OrderedDict as OD
from tqdm import tqdm
import pandas as pd

def import_facebook_data(file_path):
    try:
        with open(file_path, 'r') as file:
            edges_set = set()  
            for line in file:
                node1, node2 = sorted(map(int, line.strip().split())) 
                edges_set.add((node1, node2)) 

        edges = np.array(list(edges_set))
        return edges
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return np.array([]) 

def compute_adjacency_matrix(edge_list):

    num_nodes = np.max(edge_list) + 1  
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge in edge_list:
        i, j = edge
        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1
    
    return adjacency_matrix

def compute_adjacency_matrix_and_Fiedler_vector(edge_list, cut_type):

    num_nodes = np.max(edge_list) + 1  
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge in edge_list:
        i, j = edge
        adjacency_matrix[i, j] = adjacency_matrix[j, i] = 1

    degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))
    laplacian_matrix = degree_matrix - adjacency_matrix

    if cut_type == 'normalized':
        degree_sqrt_inverse = np.sqrt(np.linalg.inv(degree_matrix))
        normalized_laplacian = degree_sqrt_inverse @ laplacian_matrix @ degree_sqrt_inverse
        eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian_matrix)

    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    fiedler_vector = eigenvectors[:, 1]

    return adjacency_matrix, fiedler_vector

def spectralDecomp_OneIter(edge_list, cut_type, give_partitions=False):
   
    # Compute the adjacency matrix and Fiedler vector
    adjacency_matrix, fiedler_vector = compute_adjacency_matrix_and_Fiedler_vector(edge_list, cut_type)

    # Segregate nodes into positive and negative partitions based on the Fiedler vector
    positive_nodes = np.argwhere(fiedler_vector >= 0).flatten()
    negative_nodes = np.argwhere(fiedler_vector < 0).flatten()

    if give_partitions:
        return positive_nodes, negative_nodes

    positive_partition = [[node, positive_nodes[0]] for node in positive_nodes]
    negative_partition = [[node, negative_nodes[0]] for node in negative_nodes]

    graph_partition = positive_partition + negative_partition
    graph_partition = np.array(graph_partition)

    return fiedler_vector, adjacency_matrix, graph_partition

def spectralDecomposition(node_list, cut_type):
    stopping_cond = np.max(node_list) // 10

    # Dictionary to store the clusters with cluster ID as the key
    graph_parts = {}

    def make_sub_clusters(sub_node_list, adjacency_matrix):

        if len(sub_node_list) < stopping_cond:
            # If the subgraph size is below the stopping condition, assign it as a cluster
            graph_parts[min(sub_node_list)] = sub_node_list
            return None

        # Create mappings for transforming data in and out of the recursion
        internal_list = []
        current_set = set(sub_node_list)

        for i, node_i in enumerate(sub_node_list):
            common_nodes = set(np.argwhere(adjacency_matrix[node_i, :] != 0).flatten()).intersection(current_set)
            internal_list.extend([(i, sub_node_list.index(node_j)) for node_j in common_nodes])
        
        positive, negative = spectralDecomp_OneIter(np.array(internal_list), cut_type, give_partitions=True)

        sub_positive = [sub_node_list[i] for i in positive]
        sub_negative = [sub_node_list[i] for i in negative]

        # Check if there are no elements in sub_positive or sub_negative
        if not sub_positive:
            graph_parts[min(sub_negative)] = sub_negative
            return None
        elif not sub_negative:
            graph_parts[min(sub_positive)] = sub_positive
            return None
        else:
            make_sub_clusters(sub_negative, adjacency_matrix)
            make_sub_clusters(sub_positive, adjacency_matrix)
        

    # Compute the adjacency matrix of the entire graph
    adjacency_matrix = compute_adjacency_matrix(node_list)
    unique_nodes = np.unique(node_list).tolist()

    # Start the recursive clustering process
    make_sub_clusters(unique_nodes, adjacency_matrix)

    # Sort the graph partitions by cluster ID
    graph_parts = OD(sorted(graph_parts.items()))

    final_partitions = [[node, min(nodes)] for min_node, nodes in graph_parts.items() for node in nodes]

    final_partitions = np.array(final_partitions)
    second_elements = final_partitions[:, 1]
    unique_count = len(np.unique(second_elements))
    print("Total Communities", unique_count)

    return final_partitions

def createSortedAdjMat(graph_partition, nodes_connectivity_list):
    # Extract unique cluster IDs from the graph partition
    cluster_ids = np.unique(graph_partition[:, 1])
    
    # Initialize a list to store individual cluster partitions
    partitions = []
    for cluster_id in cluster_ids:
        partition = graph_partition[np.where(graph_partition[:, 1] == cluster_id)][:, 0]
        partitions.append(partition)
    
    # Create mappings to manage node order for the adjacency matrix
    node_order_to_index = {}
    index_to_node_order = {}
    current_index = 0
    for partition in partitions:
        for node in partition:
            node_order_to_index[current_index] = node
            index_to_node_order[node] = current_index
            current_index += 1
    
    # Compute the original adjacency matrix
    original_adjacency_matrix = compute_adjacency_matrix(nodes_connectivity_list)
    
    # Initialize a sorted adjacency matrix with zeros
    sorted_adjacency_matrix = np.zeros_like(original_adjacency_matrix)
    
    # Sort the adjacency matrix based on the defined node order
    for current_index in range(len(original_adjacency_matrix)):
        connected_nodes = np.argwhere(original_adjacency_matrix[node_order_to_index[current_index], :] != 0)
        for neighbor_index in connected_nodes:
            if neighbor_index[0] in index_to_node_order:
                sorted_adjacency_matrix[current_index, index_to_node_order[neighbor_index[0]]] = 1
                sorted_adjacency_matrix[index_to_node_order[neighbor_index[0]], current_index] = 1


    plt.figure(figsize=(10, 10))
    plt.title("Sorted Adjacency Matrix")
    plt.imshow(sorted_adjacency_matrix)
    plt.show()

    return sorted_adjacency_matrix

def plot(fielder_vec_fb, adj_mat_fb, graph_partition_fb,nodes_connectivity_list_fb, count):
     # Plot the sorted Fiedler vector and adjacency matrix
    fig1 = plt.figure(figsize=(10, 4))
    ax1 = fig1.add_subplot(131) #131 to specify the position of the fig (1 row 3 col)
    ax1.plot(np.sort(fielder_vec_fb))
    ax1.set_title("Sorted Fiedler Vector")

    ax2 = fig1.add_subplot(132)
    ax2.imshow(adj_mat_fb, cmap='gray')
    ax2.set_title("Adjacency Matrix")

    plt.tight_layout()

    # Save the first set of plots
    fig1.savefig(f"{count}spectral_plots.png")  # Specify the filename and format

    # Display the plots
    plt.show()

    # Create a NetworkX graph to visualize the results
    G = nx.Graph()
    n = np.max(nodes_connectivity_list_fb) + 1 
    G.add_nodes_from(np.arange(n))
    G.add_edges_from(nodes_connectivity_list_fb)


    # Plot the graph with colored edges
    fig2 = plt.figure(figsize=(8, 6))
    ax3 = fig2.add_subplot(111)
    pos = nx.spring_layout(G)  # Layout for visualization
    nx.draw(G, pos, node_color=[graph_partition_fb[node, 1] for node in G.nodes], node_size=50, cmap=plt.cm.coolwarm, with_labels=False, edge_color='green', ax=ax3)
    ax3.set_title("Graph Partition Visualization")

    # Save the second set of plots
    fig2.savefig(f"{count}graph_partition_plots.png")  # Specify the filename and format

    # Display the plots
    plt.show()

def calculate_modularity(nodes, adjacency_matrix, degree_matrix, total_edge_weight):
    modularity = 0.0
    for i in nodes:
        for j in nodes:
            modularity += adjacency_matrix[i, j] - (degree_matrix[i] * degree_matrix[j]) / (2 * total_edge_weight)

    modularity /= (2 * total_edge_weight)
    
    return modularity

def louvain_one_iter(node_list, adjacency_matrix, degree_matrix):
    unique_nodes = np.unique(node_list)
    partitions = [set([node]) for node in unique_nodes]
    total_edge_weight = np.count_nonzero(adjacency_matrix)
    new_partitions = []  
    visited = [False] * len(partitions)

    for i in tqdm(range(len(partitions))): 

        if False in visited :
            current_node = list(partitions[0])[0]  
            temp_partitions = partitions.copy()
            nodes_to_remove = []
            best_modularity_increase = 0.0
            best_partition_change = []
            visited[i] = True

            for j in range(len(temp_partitions)):
                subgraph_nodes = temp_partitions[j].copy()
                modularity_before_addition = calculate_modularity(subgraph_nodes, adjacency_matrix, degree_matrix, total_edge_weight)
                subgraph_nodes.add(current_node)
                modularity_after_addition = calculate_modularity(subgraph_nodes, adjacency_matrix, degree_matrix, total_edge_weight)
                subgraph_nodes.remove(current_node)
                modularity_change = modularity_after_addition - modularity_before_addition

                if modularity_change > best_modularity_increase:
                    node_to_remove = temp_partitions[j]
                    index = j
                    best_partition_change = [subgraph_nodes, {current_node}]
                    best_modularity_increase = modularity_change

            if node_to_remove in partitions:
                partitions.remove(node_to_remove)
                
            visited[index] = True 
            if best_partition_change:
                partitions.append(set.union(*best_partition_change))
        else :
            break
        
    new_partitions.extend(partitions)

    return new_partitions

def import_bitcoin_data(path):
    df = pd.read_csv(path, skiprows=0, header=None)

    df = df.iloc[:,:2]
    df.columns = ['source', 'target']

    unique_values = np.unique(df[['source', 'target']].values)
    value_to_int = {value: index for index, value in enumerate(unique_values)}

    # Replace values in the DataFrame using the mapping dictionary
    df['source'] = df['source'].map(value_to_int)
    df['target'] = df['target'].map(value_to_int)

    connectivity_data = df.to_numpy()

    return connectivity_data

def remove_isolated(nodes_connectivity_list_btc):
    graph = nx.Graph()
    graph.add_edges_from(nodes_connectivity_list_btc)
    connected_components = list(nx.connected_components(graph))
    for component_nodes in connected_components:
        subgraph = graph.subgraph(component_nodes)
        component_edges = list(subgraph.edges())
        break
    component_edges = np.array(component_edges)
    return component_edges

def plot_btc(nodes_connectivity_list_btc, graph_partition_btc,count):

    G = nx.Graph()
    nodes = graph_partition_btc[:, 0]
    partitions = graph_partition_btc[:, 1]

    for node, partition in zip(nodes, partitions):
        G.add_node(node, partition=partition)

    # Add edges from nodes_connectivity_list_btc
    G.add_edges_from(nodes_connectivity_list_btc)

    # Plot the graph with colored nodes
    fig2 = plt.figure(figsize=(8, 6))
    ax3 = fig2.add_subplot(111)
    pos = nx.spring_layout(G)  # Layout for visualization

    # Extract node colors based on the partition attribute
    node_colors = [G.nodes[node]['partition'] for node in G.nodes]

    nx.draw(G, pos, node_color=node_colors, cmap=plt.cm.coolwarm, with_labels=False, edge_color='green', ax=ax3)
    ax3.set_title("Graph Partition Visualization")

    # Save the second set of plots
    fig2.savefig(f"{count}graph_partition_plots.png")  # Specify the filename and format

    # Display the plots
    plt.show()

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("/data/home/vvaibhav/AI/DA/Ass2/data/facebook_combined.txt")
    #print(nodes_connectivity_list_fb, (nodes_connectivity_list_fb.shape))

    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
    cut = 'Normalized'  #or min_cut
    fielder_vec_fb, adj_mat_fb, graph_partition_fb = spectralDecomp_OneIter(nodes_connectivity_list_fb , cut)
    plot(fielder_vec_fb, adj_mat_fb, graph_partition_fb,nodes_connectivity_list_fb ,1)

    # # This is for question no. 2. Use the function 
    # # written for question no.1 iteratetively within this function.
    # # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # # identified as part of question 2. The naming convention for the community id is as before.
    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb,cut)
    plot(fielder_vec_fb, adj_mat_fb, graph_partition_fb, nodes_connectivity_list_fb ,2)

    # This is for question no. 3
    # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # adjacency matrix is to be sorted in an increasing order of communitites.
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)

    # This is for question no. 4
    # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # graph_partition vector is as before.
    adjacency = compute_adjacency_matrix(nodes_connectivity_list_fb)
    degree =  np.sum(adjacency , axis=1)
    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb, adjacency , degree )
    print(graph_partition_louvain_fb)


    ############ Answer qn 1-4 for bitcoin data #################################################
    # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("/data/home/vvaibhav/AI/DA/Ass2/data/soc-sign-bitcoinotc.csv")

    # Question 1
    fielder_vec_btc, adj_mat_btc, graph_partition_btc = spectralDecomp_OneIter(nodes_connectivity_list_btc, cut)

    # Question 2
    nodes_connectivity_list_btc = remove_isolated(nodes_connectivity_list_btc)
    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc, cut)
    plot_btc(nodes_connectivity_list_btc, graph_partition_btc,4)

    # Question 3
    #clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    # Question 4
    adjacency = compute_adjacency_matrix(nodes_connectivity_list_btc)
    degree =  np.sum(adjacency, axis=1)
    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc, adjacency, degree)
    print(graph_partition_louvain_btc)
    




