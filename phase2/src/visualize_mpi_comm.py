import argparse
import matplotlib.pyplot as plt
import networkx as nx
import os


OUT_DIR = "./diagrams"
os.makedirs(OUT_DIR, exist_ok=True)


def draw_diagram(name, edges, ranks, edge_style):
    """
    Draws a detailed diagram with:
    - custom colors
    - dashed/curved arrows
    - labels on edges
    - improved node styling
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(ranks))
    G.add_edges_from([e[0] for e in edges])  # edges are (edge, label)

    # Layout: rank 0 left, others right
    pos = {}
    pos[0] = (-1.3, 0.5)
    for i in range(1, ranks):
        pos[i] = (1.3, 1 - i / ranks)

    plt.figure(figsize=(6, 4))

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        node_color="#87CEFA",
        edgecolors="black"
    )

    # Labels for nodes
    labels = {i: f"Rank {i}" + ("\n(root)" if i == 0 else "") for i in range(ranks)}
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight="bold")

    # Draw arrows
    for (src, dst), label in edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(src, dst)],
            arrowstyle="-|>",
            arrowsize=25,
            width=edge_style["width"],
            edge_color=edge_style["color"],
            connectionstyle=edge_style["curve"]
        )
        # Add edge label
        label_pos = {
            "horizontalalignment": "center",
            "verticalalignment": "center"
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels={(src, dst): label},
            font_size=10,
            rotate=False
        )

    plt.title(name, fontsize=16, fontweight="bold")
    plt.axis("off")

    out_path = os.path.join(OUT_DIR, f"{name.replace(' ', '_').lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def generate_all(ranks):

    # 1) Broadcast
    broadcast_edges = [((0, r), "bcast()") for r in range(1, ranks)]
    draw_diagram(
        "Broadcast",
        broadcast_edges,
        ranks,
        edge_style={"color": "#1f77b4", "width": 2, "curve": "arc3,rad=0.0"}
    )

    # 2) Blocking P2P
    blocking_edges = [((0, 1), "send() / recv()")]
    draw_diagram(
        "Blocking P2P",
        blocking_edges,
        ranks,
        edge_style={"color": "red", "width": 3, "curve": "arc3,rad=0.0"}
    )

    # 3) Non-blocking P2P
    nonblocking_edges = [
        ((0, 1), "Isend() / Irecv() + wait()")
    ]
    draw_diagram(
        "Non-blocking P2P",
        nonblocking_edges,
        ranks,
        edge_style={"color": "green", "width": 2, "curve": "arc3,rad=0.3"}
    )

    # 4) Scatter
    scatter_edges = [
        ((0, r), f"chunk {r}") for r in range(1, ranks)
    ]
    draw_diagram(
        "Scatter",
        scatter_edges,
        ranks,
        edge_style={"color": "purple", "width": 2, "curve": "arc3,rad=0.0"}
    )

    # 5) Gather
    gather_edges = [
        ((r, 0), f"value {r}") for r in range(1, ranks)
    ]
    draw_diagram(
        "Gather",
        gather_edges,
        ranks,
        edge_style={"color": "orange", "width": 3, "curve": "arc3,rad=0.0"}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranks", type=int, default=4)
    args = parser.parse_args()

    generate_all(args.ranks)
