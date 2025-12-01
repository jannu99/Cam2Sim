import json
import matplotlib.pyplot as plt

# Define the file path
file_path = "/home/davide/Cam2Sim/maps/guerickestrae_alte_heide_munich_25_11_30/vehicle_data.json"

# Read the JSON file
with open(file_path, 'r') as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(12, 10))

# Plot Hero Car
hero_pos = data['hero_car']['position']
ax.scatter(hero_pos[0], hero_pos[1], c='red', marker='*', s=300, label='Hero Car', zorder=10)
# Optional: Label the hero car as well
ax.text(hero_pos[0] + 0.5, hero_pos[1] + 0.5, "Hero", fontsize=12, color='red', fontweight='bold')

# Plot Spawn Positions
for sp in data['spawn_positions']:
    start = sp['start']
    cluster_id = sp.get('cluster_id', 'N/A')
    
    # Parse color string "R,G,B" to [0-1] range
    color_str = sp.get('color', '0,0,0')
    if color_str:
        try:
            color_rgb = [float(c)/255.0 for c in color_str.split(',')]
        except ValueError:
            color_rgb = [0, 0, 0]
    else:
        color_rgb = [0, 0, 0]
    
    # Plotting marker
    ax.scatter(start[0], start[1], color=color_rgb, s=100, marker='o', edgecolors='black')
    
    # Add the ID text
    # We add a small offset (0.5, 0.5) so the text doesn't overlap perfectly with the dot
    ax.text(start[0] + 0.5, start[1] + 0.5, str(cluster_id), fontsize=10, zorder=15)

ax.set_aspect('equal')
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Hero Car and Spawn Positions with IDs')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.show()