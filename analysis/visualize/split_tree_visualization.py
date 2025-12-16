from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx


def plot_split_tree(root, show_recon_error: bool = False, figsize=(14, 8)):
    """
    Visualize a split tree produced by analyze_split_tree_levels(...).

    Expects each node to have:
      - node.depth
      - node.matrix
      - node.num_permutations
      - node.cycle_length
      - node.recon_error (optional)
      - node.left / node.right
    """
    if root is None:
        print("plot_split_tree: root is None (empty tree).")
        return

    G = nx.DiGraph()

    def _nnz(node) -> int:
        # faster than node.matrix.nonzero for large matrices
        return int((node.matrix != 0).sum())

    def visit(node, parent_key: Optional[int] = None):
        key = node.node_id  # stable id (better than id(node))

        perms = node.num_permutations if node.num_permutations is not None else "?"
        cycle = node.cycle_length if node.cycle_length is not None else None
        cycle_str = f"{cycle:.3f}" if isinstance(cycle, (int, float)) else "?"

        label = (
            f"id={node.node_id}\n"
            f"d={node.depth}\n"
            f"nnz={_nnz(node)}\n"
            f"perms={perms}\n"
            f"cycle={cycle_str}"
        )

        if show_recon_error:
            err = node.recon_error
            err_str = f"{err:.1e}" if isinstance(err, (int, float)) else "?"
            label += f"\nerr={err_str}"

        G.add_node(key, label=label)

        if parent_key is not None:
            G.add_edge(parent_key, key)

        if getattr(node, "left", None) is not None:
            visit(node.left, key)
        if getattr(node, "right", None) is not None:
            visit(node.right, key)

    visit(root)

    # Layout: prefer graphviz 'dot' if available
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        # fallback (no graphviz/pygraphviz installed)
        pos = nx.spring_layout(G, seed=0)

    labels = nx.get_node_attributes(G, "label")

    plt.figure(figsize=figsize)
    nx.draw(
        G,
        pos,
        labels=labels,
        node_size=3000,
        node_color="lightblue",
        font_size=8,
        arrows=True,
    )
    plt.title("Split-Tree (nodes annotated with WFA perms & cycle)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def compute_tree_positions(root):
    """
    Compute (x, y) positions for a binary tree.
    y = -depth
    x = in-order index
    """
    pos = {}
    x_counter = [0]

    def dfs(node):
        if node.left:
            dfs(node.left)

        # Assign position
        pos[node.node_id] = (x_counter[0], -node.depth)
        x_counter[0] += 1

        if node.right:
            dfs(node.right)

    dfs(root)
    return pos


def build_tree_graph(root):
    import networkx as nx
    G = nx.DiGraph()

    def visit(node):
        label = (
            f"id={node.node_id}\n"
            f"d={node.depth}\n"
            f"nnz={node.nnz}"
        )
        G.add_node(node.node_id, label=label)

        if node.left:
            G.add_edge(node.node_id, node.left.node_id)
            visit(node.left)
        if node.right:
            G.add_edge(node.node_id, node.right.node_id)
            visit(node.right)

    visit(root)
    return G


def plot_split_tree_as_tree(root, figsize=(14, 8)):
    """
    Plot the split tree using a true tree layout.
    Returns the matplotlib Figure (does NOT call plt.show()).
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    if root is None:
        raise ValueError("plot_split_tree_as_tree: root is None")

    G = build_tree_graph(root)
    pos = compute_tree_positions(root)
    labels = nx.get_node_attributes(G, "label")

    fig = plt.figure(figsize=figsize)

    nx.draw(
        G,
        pos,
        labels=labels,
        node_size=3000,
        node_color="lightblue",
        font_size=9,
        arrows=True,
    )

    plt.title("Split Tree Structure (True Tree Layout)")
    plt.axis("off")
    plt.tight_layout()

    return fig