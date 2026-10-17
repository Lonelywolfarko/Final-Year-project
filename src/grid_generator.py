
import os
import cv2
import numpy as np

# Load PCB layout
image = cv2.imread("input/pcb_sample.jpeg")

if image is None:
    raise FileNotFoundError("PCB image not found!")

height, width = image.shape[:2]

# Grid configuration
CELL_SIZE = 5

grid_cols = (width + CELL_SIZE - 1) // CELL_SIZE
grid_rows = (height + CELL_SIZE - 1) // CELL_SIZE

# Cell types
FREE = 0
OBSTACLE = 1

# Initially, all cells are free
grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

# Convert PCB image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 50, 150)

# Create output directory
os.makedirs("output", exist_ok=True)

# Classify every grid cell
for row in range(grid_rows):
    for col in range(grid_cols):

        y1 = row * CELL_SIZE
        y2 = min(y1 + CELL_SIZE, height)

        x1 = col * CELL_SIZE
        x2 = min(x1 + CELL_SIZE, width)

        cell_edges = edges[y1:y2, x1:x2]

        edge_count = np.count_nonzero(cell_edges)

        # Diagnostic rule: any edge pixel marks the cell
        if edge_count > 0:
            grid[row, col] = OBSTACLE

# Visualize grid classification
grid_view = np.zeros((grid_rows, grid_cols, 3), dtype=np.uint8)

# Free cells = white
grid_view[grid == FREE] = (255, 255, 255)

# Obstacle cells = red
grid_view[grid == OBSTACLE] = (0, 0, 255)

# Enlarge grid for easier inspection
grid_view = cv2.resize(
    grid_view,
    (width, height),
    interpolation=cv2.INTER_NEAREST
)

# Save diagnostic images
cv2.imwrite("output/pcb_grayscale.jpg", gray)
cv2.imwrite("output/pcb_edges.jpg", edges)
cv2.imwrite("output/pcb_grid_classification.png", grid_view)

# Print results
print("Image size:", width, "x", height)
print("Grid shape:", grid.shape)
print("Free cells:", np.count_nonzero(grid == FREE))
print("Obstacle cells:", np.count_nonzero(grid == OBSTACLE))
print("Saved output/pcb_grid_classification.png")

# Step 5: Overlay grid lines on the original PCB

grid_overlay = image.copy()

# Draw vertical grid lines
for x in range(0, width, CELL_SIZE):
    cv2.line(
        grid_overlay,
        (x, 0),
        (x, height - 1),
        (0, 255, 0),
        1
    )

# Draw horizontal grid lines
for y in range(0, height, CELL_SIZE):
    cv2.line(
        grid_overlay,
        (0, y),
        (width - 1, y),
        (0, 255, 0),
        1
    )

# Save the overlay
cv2.imwrite("output/pcb_grid_overlay.png", grid_overlay)

print("Saved output/pcb_grid_overlay.png")

# Step 6: Create a thresholded PCB image for inspection

# Convert grayscale into a black-and-white image
_, binary = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Save the thresholded image
cv2.imwrite("output/pcb_binary.png", binary)

print("Saved output/pcb_binary.png")
