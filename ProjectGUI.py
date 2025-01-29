import tkinter as tk
from tkinter import messagebox
import networkx as nx
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np
from PIL import Image, ImageTk

# --- PERT Calculation Functions ---
def calculate_expected_time(o, m, p):
    return int((o + 4 * m + p) / 6)

def calculate_standard_deviation(o, m, p):
    return (p - o) / 6

def find_critical_path(G, activities):
    # Find start and end nodes
    start_nodes = [node for node in G.nodes if G.in_degree(node) == 0]
    end_nodes = [node for node in G.nodes if G.out_degree(node) == 0]

    if not start_nodes or not end_nodes:
        raise ValueError("Network is missing start/end nodes. Check activity connections.")

    start = start_nodes[0]
    end = end_nodes[0]

    # Check if there's a valid path
    if not nx.has_path(G, start, end):
        raise ValueError("No path exists between start and end nodes. Network is disconnected.")

    # Calculate longest path using expected time as weights
    critical_path = nx.dag_longest_path(G, weight=lambda u, v, _: activities[f"{u}-{v}"]['E'])
    critical_edges = [(critical_path[i], critical_path[i+1]) for i in range(len(critical_path)-1)]
    
    total_duration = sum(activities[f"{u}-{v}"]['E'] for u, v in critical_edges)
    return critical_edges, total_duration

# --- Visualization Functions ---
def visualize_graph(G, activities, critical_path):
    pos = nx.spring_layout(G, k=0.5)
    plt.figure(figsize=(15, 10))
    
    edge_labels = {(u, v): f"{activities[f'{u}-{v}']['E']}" for u, v in G.edges() if f"{u}-{v}" in activities}
    
    nx.draw(G, pos, with_labels=True, node_size=4000, node_color="lightblue", 
            font_size=12, font_weight="bold", edge_color="gray")
    nx.draw_networkx_edges(G, pos, edgelist=critical_path, edge_color="red", width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Project Network Diagram with Critical Path Highlighted")
    plt.show()

def plot_shaded_region(Z):
    x = np.linspace(-4, 4, 1000)
    plt.plot(x, norm.pdf(x), label="Standard Normal Distribution", color='blue')
    plt.fill_between(x, 0, norm.pdf(x), where=(x <= Z), color='skyblue', alpha=0.5, label=f'P(Z <= {Z:.2f})')
    plt.title(f"Standard Normal Distribution with Shaded Region (Z = {Z:.2f})")
    plt.xlabel("Z-Score")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

# --- GUI Logic ---
def on_add_activity():
    activity = entry_activity.get().strip()
    o = entry_optimistic.get().strip()
    m = entry_most_likely.get().strip()
    p = entry_pessimistic.get().strip()

    if not activity or not o or not m or not p:
        messagebox.showerror("Input Error", "All fields are required.")
        return

    try:
        # Validate activity format
        start, end = activity.split('-')
        start, end = int(start), int(end)
        
        # Validate times
        o = int(o)
        m = int(m)
        p = int(p)
        if not (o <= m <= p):
            messagebox.showerror("Input Error", "Optimistic ≤ Most Likely ≤ Pessimistic required.")
            return

        # Store activity
        expected_time = calculate_expected_time(o, m, p)
        std_dev = calculate_standard_deviation(o, m, p)
        activities[f"{start}-{end}"] = {'O': o, 'M': m, 'P': p, 'E': expected_time, 'V': std_dev}
        
        # Update UI
        activity_listbox.insert(tk.END, f"{activity}: O={o}, M={m}, P={p}, E={expected_time}")
        G.add_edge(start, end)
        
        # Clear fields
        entry_activity.delete(0, tk.END)
        entry_optimistic.delete(0, tk.END)
        entry_most_likely.delete(0, tk.END)
        entry_pessimistic.delete(0, tk.END)
        
        # Enable Done button
        done_button.config(state=tk.NORMAL)

    except ValueError as e:
        messagebox.showerror("Input Error", f"Invalid input: {str(e)}")

def on_done():
    try:
        # Perform analysis
        critical_path, total_duration = find_critical_path(G, activities)
        result_label.config(text=f"Critical Path: {' -> '.join(f'{u}-{v}' for u, v in critical_path)}")
        result_duration.config(text=f"Total Duration: {total_duration} days")

        # Enable project duration input
        entry_project_duration.config(state=tk.NORMAL)
        calculate_prob_button.config(state=tk.NORMAL)

    except Exception as e:
        messagebox.showerror("Analysis Error", str(e))

def on_calculate_prob():
    try:
        project_duration = int(entry_project_duration.get())
        critical_path, total_duration = find_critical_path(G, activities)
        
        # Calculate total standard deviation
        total_var = sum((activities[f"{u}-{v}"]['V'] ** 2) for u, v in critical_path)
        total_std_dev = math.sqrt(total_var)
        
        if total_std_dev == 0:
            messagebox.showinfo("Info", "All activities have zero variance (O=P). Probability is 100% if duration ≥ total time.")
            return

        Z = (project_duration - total_duration) / total_std_dev
        Z = round(Z, 2)
        probability = norm.cdf(Z) * 100

        # Update results
        result_probability.config(text=f"Completion Probability: {probability:.2f}%")
        result_zscore.config(text=f"Z-Score: {Z}")
        result_tabulated.config(text=f"Standard Deviation: {total_std_dev:.2f} days")

        # Show plots
        visualize_graph(G, activities, critical_path)
        plot_shaded_region(Z)

    except ValueError:
        messagebox.showerror("Input Error", "Invalid project duration.")

# --- GUI Setup ---
root = tk.Tk()
root.title("PERT Analysis Tool")
root.state("zoomed")

# Background Image
try:
    bg_image = Image.open("pert.png")
    bg_image = bg_image.resize((root.winfo_screenwidth(), root.winfo_screenheight()), Image.Resampling.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    bg_label = tk.Label(root, image=bg_photo)
    bg_label.place(relx=0, rely=0, relwidth=1, relheight=1)
except Exception as e:
    print(f"Background image error: {e}")

# Activity Input Fields
input_frame = tk.Frame(root, bg="#e5e5e5")
input_frame.pack(pady=10)

tk.Label(input_frame, text="Activity (e.g., 1-2):", bg="#e5e5e5").grid(row=0, column=0, padx=5)
entry_activity = tk.Entry(input_frame)
entry_activity.grid(row=0, column=1, padx=5)

tk.Label(input_frame, text="Optimistic (O):", bg="#e5e5e5").grid(row=0, column=2, padx=5)
entry_optimistic = tk.Entry(input_frame, width=8)
entry_optimistic.grid(row=0, column=3, padx=5)

tk.Label(input_frame, text="Most Likely (M):", bg="#e5e5e5").grid(row=0, column=4, padx=5)
entry_most_likely = tk.Entry(input_frame, width=8)
entry_most_likely.grid(row=0, column=5, padx=5)

tk.Label(input_frame, text="Pessimistic (P):", bg="#e5e5e5").grid(row=0, column=6, padx=5)
entry_pessimistic = tk.Entry(input_frame, width=8)
entry_pessimistic.grid(row=0, column=7, padx=5)

add_activity_button = tk.Button(input_frame, text="Add Activity", command=on_add_activity, bg="#a89dd1")
add_activity_button.grid(row=0, column=8, padx=10)

# Activity List
activity_listbox = tk.Listbox(root, width=80, height=8)
activity_listbox.pack(pady=10)

# Buttons
button_frame = tk.Frame(root, bg="#e5e5e5")
button_frame.pack(pady=10)

done_button = tk.Button(button_frame, text="Analyze Critical Path", command=on_done, bg="#90f18e", state=tk.DISABLED)
done_button.pack(side='left', padx=5)

tk.Label(button_frame, text="Desired Duration (days):", bg="#e5e5e5").pack(side='left', padx=5)
entry_project_duration = tk.Entry(button_frame, width=10, state=tk.DISABLED)
entry_project_duration.pack(side='left', padx=5)

calculate_prob_button = tk.Button(button_frame, text="Calculate Probability", command=on_calculate_prob, bg="#89CFF0", state=tk.DISABLED)
calculate_prob_button.pack(side='left', padx=5)

# Results Display
result_label = tk.Label(root, text="Critical Path: ", bg="#e5e5e5", font=("Arial", 12))
result_label.pack(pady=5)

result_duration = tk.Label(root, text="Total Duration: ", bg="#e5e5e5", font=("Arial", 12))
result_duration.pack(pady=5)

result_probability = tk.Label(root, text="Probability: ", bg="#fcfcfc", font=("Arial", 12))
result_probability.pack(pady=5)

result_zscore = tk.Label(root, text="Z-Score: ", bg="#fcfcfc", font=("Arial", 12))
result_zscore.pack(pady=5)

result_tabulated = tk.Label(root, text="Standard Deviation: ", bg="#fcfcfc", font=("Arial", 12))
result_tabulated.pack(pady=5)

# Initialize Data Structures
G = nx.DiGraph()
activities = {}

root.mainloop()