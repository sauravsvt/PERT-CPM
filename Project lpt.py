import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np
from PIL import Image, ImageTk

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
        path_duration = sum(activities[f"{path[i]}-{path[i + 1]}"]['E'] for i in range(len(path) - 1) if
                            f"{path[i]}-{path[i + 1]}" in activities)
        if path_duration > max_duration:
            max_duration = path_duration
            critical_path = [(path[i], path[i + 1]) for i in range(len(path) - 1) if
                             f"{path[i]}-{path[i + 1]}" in activities]

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
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color="lightblue", font_size=12, font_weight="bold",
            edge_color="gray")

    # Highlight critical path edges in red
    critical_edges = [(u, v) for u, v in critical_path]
    nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color="red", width=2)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Project Network Diagram with Critical Path Highlighted")
    plt.show()


# Function to plot shaded region under the curve up to Z-score
def plot_shaded_region(Z):
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, norm.pdf(x), label="Standard Normal Distribution", color='blue')
    plt.fill_between(x, 0, norm.pdf(x), where=(x <= Z), color='skyblue', alpha=0.5, label=f'P(Z <= {Z:.2f})')
    plt.title(f"Standard Normal Distribution with Shaded Region (Z = {Z:.2f})")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


# GUI function to collect input and display results
def on_add_activity():
    activity = entry_activity.get().strip()
    optimistic = entry_optimistic.get().strip()
    most_likely = entry_most_likely.get().strip()
    pessimistic = entry_pessimistic.get().strip()

    # Check if any of the fields are empty
    if not activity or not optimistic or not most_likely or not pessimistic:
        messagebox.showerror("Input Error", "Please fill in all fields (Activity, Optimistic, Most Likely, Pessimistic).")
        return

    try:
        # Validate that activity contains valid start and end values (e.g., '1-2')
        start, end = activity.split('-')
        start, end = int(start), int(end)  # Convert start and end to integers

        # Validate that optimistic, most likely, and pessimistic are valid integers
        optimistic = int(optimistic)
        most_likely = int(most_likely)
        pessimistic = int(pessimistic)

        # Calculate the expected time using the PERT formula
        expected_time = calculate_expected_time(optimistic, most_likely, pessimistic)

        # Calculate the standard deviation using the PERT formula
        standard_deviation=calculate_standard_deviation(optimistic, most_likely, pessimistic)

        # Store the activity and its times
        activities[f"{start}-{end}"] = {
            'O': optimistic,
            'M': most_likely,
            'P': pessimistic,
            'E': expected_time,
            'V': standard_deviation
        }

        # Update the activity list in the GUI
        activity_listbox.insert(tk.END, f"{activity}: O={optimistic}, M={most_likely}, P={pessimistic}, E={expected_time}, V={standard_deviation}")
        # Add edge to the graph
        G.add_edge(start, end)

        # Clear the input fields
        entry_activity.delete(0, tk.END)
        entry_optimistic.delete(0, tk.END)
        entry_most_likely.delete(0, tk.END)
        entry_pessimistic.delete(0, tk.END)

    except ValueError as e:
        messagebox.showerror("Input Error", "Please enter valid numerical values for times and activity.")

def on_done():
    # Perform the PERT analysis
    try:
        critical_path, total_duration = find_critical_path(G, activities)

        # Display results in the GUI
        result_label.config(text=f"Critical Path: {' -> '.join(f'{edge[0]}-{edge[1]}' for edge in critical_path)}")
        result_duration.config(text=f"Total Project Duration: {total_duration}")

        # Get user input for project duration and calculate probability
        project_duration = int(entry_project_duration.get())
        total_std_dev = 0
        # Calculate total standard deviation for the critical path
        for edge in critical_path:
            edge_key = f"{edge[0]}-{edge[1]}"
            if edge_key in activities:
                total_std_dev += calculate_standard_deviation(activities[edge_key]['O'], activities[edge_key]['M'],
                                                              activities[edge_key]['P']) ** 2
        total_std_dev = math.sqrt(total_std_dev)

        # Calculate the Z-score for the given project duration
        Z = (project_duration - total_duration) / total_std_dev
        Z = math.floor(Z * 100) / 100  # Truncate to 2 decimal places

        probability = norm.cdf(Z)  # Get the cumulative probability
        probability_percent = probability * 100

        # Display probability and tabulated value in the GUI
        result_probability.config(
            text=f"Probability of Completion in {project_duration} days: {probability_percent:.2f}%")
        result_zscore.config(text=f"Z-Score: {Z}")
        result_tabulated.config(text=f"Tabulated Probability (Z={Z}): {probability:.4f}")

        # Message for repeated project completion chances
        result_message.config(
            text=f"if the project is performed hundred times under the same condition, \ntheir will be {probability_percent:.2f}% occasions for this job to be completed in {project_duration} Days/Weeks")


        # Plot the results
        visualize_graph(G, activities, critical_path)
        plot_shaded_region(Z)

    except Exception as e:
        messagebox.showerror("Analysis Error", f"An error occurred during analysis: {e}")
        print("Error details:", e)  # Debugging the error message


# Create the main window
root = tk.Tk()
root.title("PERT Analysis Tool")

root.state("zoomed")

# Set the background image
background_image_path = r".\pert.png"
try:
    bg_image = Image.open(background_image_path)
    bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Error loading background image: {e}")


# Initialize Graph and Activities
G = nx.DiGraph()
activities = {}


# Function to create label and entry side by side
def create_label_entry_pair(frame, label_text, entry_variable):
    label = tk.Label(frame, text=label_text, bg="#e5e5e5", font=("Arial", 10, "bold"))
    label.pack(side='left', padx=5, pady=5)
    entry = tk.Entry(frame, textvariable=entry_variable)
    entry.pack(side='left', padx=5, pady=5)
    return entry


# Create frames and input fields for the activity
frame_activity = tk.Frame(root, bg="#e5e5e5")
frame_activity.pack(pady=5)
entry_activity = create_label_entry_pair(frame_activity, "Enter activity (start-end):", tk.StringVar())

# Create frames and input fields for times
frame_optimistic = tk.Frame(root, bg="#e5e5e5")
frame_optimistic.pack(pady=5)
entry_optimistic = create_label_entry_pair(frame_optimistic, "Enter optimistic time (O):", tk.StringVar())

frame_most_likely = tk.Frame(root, bg="#e5e5e5")
frame_most_likely.pack(pady=5)
entry_most_likely = create_label_entry_pair(frame_most_likely, "Enter most likely time (M):", tk.StringVar())

frame_pessimistic = tk.Frame(root, bg="#e5e5e5")
frame_pessimistic.pack(pady=5)
entry_pessimistic = create_label_entry_pair(frame_pessimistic, "Enter pessimistic time (P):", tk.StringVar())

# Create and place input field for project duration
frame_project_duration = tk.Frame(root, bg="#e5e5e5")
frame_project_duration.pack(pady=5)
entry_project_duration = create_label_entry_pair(frame_project_duration, "Enter desired project duration (in Days/Weeks):",
                                                 tk.StringVar())

# Create and place buttons for adding activity and done
add_activity_button = tk.Button(root, text="Add Activity", command=on_add_activity, bg="#a89dd1")
add_activity_button.pack(pady=5)

done_button = tk.Button(root, text="Done", command=on_done, bg="#90f18e")
done_button.pack(pady=5)

# Create labels to display results
result_label = tk.Label(root, text="Critical Path: ", bg="#e5e5e5", font=("Arial", 12, "bold"))
result_label.pack(pady=5)

result_duration = tk.Label(root, text="Total Project Duration: ", bg="#e5e5e5", font=("Arial", 12, "bold"))
result_duration.pack(pady=5)

result_probability = tk.Label(root, text="Probability of Completion: ", bg="#fcfcfc", font=("Arial", 12, "bold"))
result_probability.pack(pady=5)

result_zscore = tk.Label(root, text="Z-Score: ", bg="#fcfcfc", font=("Arial", 12, "bold"))
result_zscore.pack(pady=5)

result_tabulated = tk.Label(root, text="Tabulated Probability: ", bg="#fcfcfc", font=("Arial", 12, "bold"))
result_tabulated.pack(pady=5)

result_message = tk.Label(root, text="Message: ", bg="#fcfcfc", font=("Arial", 12, "bold"))
result_message.pack(pady=5)

# Create the listbox to display entered activities
activity_listbox = tk.Listbox(root, width=50)
activity_listbox.pack(pady=5)

# Start the GUI
root.mainloop()
