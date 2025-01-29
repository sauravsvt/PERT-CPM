import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np

# Function to calculate the expected time using the PERT formula
def calculate_expected_time(o, m, p):
    return int((o + 4 * m + p) / 6)

# Function to calculate the standard deviation using the PERT formula
def calculate_standard_deviation(o, m, p):
    return (p - o) / 6

# Function to find the critical path in the project network
def find_critical_path(G, activities):
    EST = {}
    EFT = {}

    # Calculate EST and EFT
    for node in nx.topological_sort(G):
        if G.in_degree(node) == 0:  # Starting node (no predecessors)
            EST[node] = 0
        else:
            EST[node] = max(EFT[predecessor] for predecessor in G.predecessors(node))  # Maximum of predecessors' EFT

        # Calculating EFT: it is EST + expected time of the activity
        edge_key = None
        for predecessor in G.predecessors(node):
            edge_key = f"{predecessor}-{node}"
            if edge_key in activities:
                expected_time = activities[edge_key]['E']
                break
        EFT[node] = EST[node] + expected_time if edge_key else EST[node]

    # Calculate LFT and LST
    LFT = {}
    LST = {}

    for node in reversed(list(nx.topological_sort(G))):
        if G.out_degree(node) == 0:  # If no successors, LFT is EFT
            LFT[node] = EFT[node]
        else:
            LFT[node] = min(LST[successor] for successor in G.successors(node))  # Minimum of successors' LST

        # LST is LFT - expected time
        for predecessor in G.predecessors(node):
            edge_key = f"{predecessor}-{node}"
            if edge_key in activities:
                LST[node] = LFT[node] - activities[edge_key]['E']

    critical_path = []
    max_duration = 0
    # Find the critical path (longest path)
    for path in nx.all_simple_paths(G, source=min(G.nodes), target=max(G.nodes)):
        path_duration = sum(activities[f"{path[i]}-{path[i + 1]}"]['E'] for i in range(len(path) - 1) if f"{path[i]}-{path[i + 1]}" in activities)
        if path_duration > max_duration:
            max_duration = path_duration
            critical_path = [(path[i], path[i + 1]) for i in range(len(path) - 1) if f"{path[i]}-{path[i + 1]}" in activities]

    total_duration = max_duration
    return critical_path, total_duration

# Function to visualize the network graph
def visualize_graph(G, activities, critical_path):
    pos = nx.spring_layout(G, k=0.5)  # Use spring layout with adjusted spacing

    plt.figure(figsize=(15, 10))  # Increased figure size for better spacing
    edge_labels = {}

    # Prepare edge labels with expected duration
    for u, v in G.edges():
        edge_key = f"{u}-{v}"
        if edge_key in activities:
            edge_labels[(u, v)] = f"{activities[edge_key]['E']}"

    # Draw the network
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color="lightblue", font_size=12, font_weight="bold", edge_color="gray")

    # Highlight critical path edges in red
    critical_edges = [(u, v) for u, v in critical_path]
    nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="red", width=2)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Project Network Diagram with Critical Path Highlighted")
    plt.show()

# Function to plot shaded region under the curve up to Z-score
def plot_shaded_region(Z):
    # Create a range of x values for the plot
    x = np.linspace(-4, 4, 1000)

    # Plot the standard normal distribution curve
    plt.plot(x, norm.pdf(x), label="Standard Normal Distribution", color='blue')

    # Shade the area under the curve up to the Z value
    plt.fill_between(x, 0, norm.pdf(x), where=(x <= Z), color='skyblue', alpha=0.5, label=f'P(Z <= {Z:.2f})')

    # Add labels and title
    plt.title(f"Standard Normal Distribution with Shaded Region (Z = {Z:.2f})")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def main():
    print("PERT Analysis Tool")
    G = nx.DiGraph()
    activities = {}
    critical_path = []
    total_duration = 0

    # Input activities
    while True:
        activity = input("Enter an activity (e.g., 1-2, 2-3) or type 'done' to finish: ").strip()
        if activity.lower() == 'done':
            break
        try:
            start, end = activity.split('-')
            start, end = int(start), int(end)
            G.add_edge(start, end)

            print(f"Enter times for activity {activity}:")
            optimistic = int(input("  Optimistic time (O): "))
            most_likely = int(input("  Most likely time (M): "))
            pessimistic = int(input("  Pessimistic time (P): "))

            expected_time = calculate_expected_time(optimistic, most_likely, pessimistic)
            activities[f"{start}-{end}"] = {
                'O': optimistic,
                'M': most_likely,
                'P': pessimistic,
                'E': expected_time
            }
        except ValueError:
            print("Invalid input. Please use the format 'start-end' (e.g., 1-2).")

    # Perform PERT analysis
    try:
        critical_path, total_duration = find_critical_path(G, activities)
        print("\nCritical Path Analysis:")
        print("Critical Path:", " -> ".join(f"{edge[0]}-{edge[1]}" for edge in critical_path))
        print("Total Project Duration:", total_duration)
    except Exception as e:
        print("An error occurred during analysis:", e)

    # Get user input for project duration and calculate probability
    project_duration = int(input("\nEnter the desired project duration: "))
    total_std_dev = 0
    # Calculate total standard deviation for the critical path
    for edge in critical_path:
        edge_key = f"{edge[0]}-{edge[1]}"
        if edge_key in activities:
            total_std_dev += calculate_standard_deviation(activities[edge_key]['O'], activities[edge_key]['M'], activities[edge_key]['P'])**2
    total_std_dev = math.sqrt(total_std_dev)

    # Calculate the Z-score for the given project duration
    Z = (project_duration - total_duration) / total_std_dev
    Z = math.floor(Z * 100) / 100  # Truncate to 2 decimal places

    probability = norm.cdf(Z)  # Get the cumulative probability

    print(f"\nProbability that the project will be completed in {project_duration} days is given by P(Z <= 1): "
          f"({project_duration} - {total_duration})/√({total_std_dev ** 2:.2f}) = {project_duration - total_duration}/{total_std_dev}\nFinal Z-Score = {Z}")

    # Printing the area under the standard normal curve (tabulated value) for the Z-score
    print(f"Tabulated area under the curve for Z = {Z} is approximately: {probability:.4f}\n")

    probability_percent = probability * 100
    print(f"if the project is performed hundred times under the same condition, \ntheir will be {probability_percent:.2f}% occasions for this job to be completed in {project_duration} days")

    # Visualize the network diagram with durations and highlight the critical path
    visualize_graph(G, activities, critical_path)

    # Plot the shaded region under the normal distribution curve up to Z-score
    plot_shaded_region(Z)

if __name__ == "__main__":
    main()
