from graphviz import Digraph
from IPython.display import Image

def draw_graph(graph_data, output_file="graph.gv"):
    g = Digraph('G', filename=output_file)
    
    g.attr(
        rankdir='LR', #left to right layout
        splines='curved', # shape of edges
        overlap='false', # how to handle nodes overlap
        nodesep='0.5',
        ranksep='1.0'
    )

    for item in graph_data:
        if len(item) > 1:
            o1, rel = item[0], item[1]
            g.node(o1, shape='ellipse', color='lightblue')
            g.node(rel, shape='box', color='pink')
            g.edge(o1, rel)
            
            if len(item) > 2:
                o2 = item[2]
                g.node(o2, shape='ellipse', color='lightblue')
                g.edge(rel, o2)
                
    return g

def draw_graph_2(graph_data, output_file="graph.gv"):
    g = Digraph('G', filename=output_file)
    
    g.attr(
        rankdir='LR', #left to right layout
        splines='line', # shape of edges
        overlap='false', # how to handle nodes overlap
        nodesep='0.5',
        ranksep='1.0'
    )

    for item in graph_data:
        if len(item) > 1:
            o1, rel = item[0], item[1]
            g.node(o1, shape='ellipse', color='lightblue')
        
            if len(item) > 2:
                o2 = item[2]
                g.node(o2, shape='ellipse', color='lightblue')
                g.edge(o1, o2, label=rel)
            else:
                g.edge(o1, o1, label=rel)
                
    return g


def draw_tripartite_graph(graph_data, output_file="graph.gv"):
    """
    Draws a proper tripartite graph with three distinct layers:
    Left (objects) - Middle (relations) - Right (objects)
    
    Args:
        graph_data: List of relationships as [['o1', 'rel', 'o2'],...] or [['o1', 'rel'],...]
        output_file: Output file path (default: "graph.gv")
    Returns:
        The graphviz Digraph object
    """
    g = Digraph('G', filename=output_file)
    
    # Graph styling
    g.attr(rankdir='LR', splines='line', overlap='false', 
           nodesep='0.5', ranksep='1.0')
    
    # Track nodes and their positions
    left_nodes = {}
    middle_nodes = {}
    right_nodes = {}
    edges = set()
    
    # Process all relationships
    for item in graph_data:
        if len(item) >= 2:
            src, rel = item[0], item[1]
            
            # key -> label
            left_nodes[f"l_{src}"] = src
            
            if len(item) >= 3:
                dest = item[2]
                right_nodes[f"r_{dest}"] = dest

                # keeping the dest as id for disambiguation
                middle_nodes[f"m_{rel}_{dest}"] = rel
                edges.add((f"l_{src}", f"m_{rel}_{dest}"))
                edges.add((f"m_{rel}_{dest}", f"r_{dest}"))
            else:
                middle_nodes[f"m_{rel}"] = rel
                edges.add((f"l_{src}", f"m_{rel}"))
                

    for node, label in left_nodes.items():
        g.node(node, label, shape='ellipse', color='lightblue')
    
    for node, label in right_nodes.items():
        g.node(node, label, shape='ellipse', color='lightblue')
        
    for node, label in middle_nodes.items():
        g.node(node, label, shape='box', color='pink')
        
    for src, dest in edges:
        g.edge(src, dest)
    # Enforce tripartite structure using rank groups
    # a subgraph can be used to represent graph structure, 
    # indicating that certain nodes and edges should be 
    # grouped together
    with g.subgraph() as left:
        left.attr(rank='same')
        for node in left_nodes:
            left.node(node)
    
    with g.subgraph() as middle:
        middle.attr(rank='same')
        for node in middle_nodes:
            middle.node(node)
    
    with g.subgraph() as right:
        right.attr(rank='same')
        for node in right_nodes:
            right.node(node)
    
    # Render and display
    g.render(format='png', cleanup=True)
    # display(Image(filename=f"{output_file}.png"))
    return g
