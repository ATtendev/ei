import numpy as np
import os
import matplotlib.pyplot as plt
import uuid
from pathlib import Path

def plot_radar_chart(proficiency, clarity, completeness, accuracy):
    # Define the labels and maximum values
    labels = np.array(['Proficiency', 'Clarity', 'Completeness', 'Accuracy'])
    max_values = np.array([5, 5, 5, 5])  # Maximum values for each category
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    values = np.array([proficiency, clarity, completeness, accuracy])
    
    # Plotting
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, max_values, color='lightgray', alpha=0.5)  # Fill with light gray for maximum values
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)
    
    # Customize axes
    ax.set_yticklabels([])
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=12)
    
    # Save the plot
    file_name = f"{uuid.uuid4().hex}.png"
    path = Path(__file__).parent.absolute()
    file_path = os.path.join(path, "results", file_name)
    fig.savefig(file_path)
    plt.close(fig)
    
    return [file_path]

if __name__ == "__main__":
    plot_radar_chart(5, 3, 3, 2)
