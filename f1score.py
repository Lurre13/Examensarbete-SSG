import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Data for the 2x2 plot
data = [
    [29, 14238],
    [1, 2]
]

# Colors for the cells
colors = [
    ['#d5f5d8', '#f5d5e5'],  # light green and light pink
    ['#f5d5e5', '#d5f5d8']   # light pink and light green
]

# Create the plot
fig, ax = plt.subplots(figsize=(5, 5))

# Draw the rectangles and add the text
for i in range(2):
    for j in range(2):
        rect = Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor='green', linewidth=0)
        ax.add_patch(rect)
        ax.text(j + 0.5, i + 0.5, str(data[i][j]), ha='center', va='center', fontsize=20)

model4 = "gpt-4-turbo-2024-04-09"
model3 = "gpt-3.5-turbo-0125"

# Add custom labels at the positions
ax.text(0.5, -0.05, '+Positive', ha='center', va='center', fontsize=12)
ax.text(1.5, -0.05, '-Negative', ha='center', va='center', fontsize=12)
ax.text(-0.1, 0.5, '+True', ha='center', va='center', fontsize=12, rotation=-50)
ax.text(-0.1, 1.5, '-False', ha='center', va='center', fontsize=12, rotation=-50)
ax.text(1, -0.2, f'F1 Score {model3}', ha='center', va='center', fontsize=15, rotation=0)

# Adjust the plot limits and remove the axis
ax.set_xlim(0, 2)
ax.set_ylim(0, 2)
ax.invert_yaxis()
ax.axis('off')

# Add title

# Display the plot
plt.show()
