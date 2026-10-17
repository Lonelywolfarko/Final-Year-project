
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
START = 2
DESTINATION = 3
ROUTED_TRACE = 4

# Initially, every cell is unclassified free space
grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)


def pixel_to_grid(x, y):
    """Convert image pixel coordinates (x, y) to (row, col)."""
    row = y // CELL_SIZE
    col = x // CELL_SIZE
    return row, col


def mark_cell(x, y, cell_type):
    """Mark the grid cell containing pixel (x, y)."""
    row, col = pixel_to_grid(x, y)

    if 0 <= row < grid_rows and 0 <= col < grid_cols:
        grid[row, col] = cell_type


# Example coordinate conversion
print("Image size:", width, "x", height)
print("Grid shape:", grid.shape)

print("Top-left pixel:", pixel_to_grid(0, 0))
print(
    "Bottom-right pixel:",
    pixel_to_grid(width - 1, height - 1)
)

# Example only: these coordinates are NOT detected pads.
# Uncomment to test marking one grid cell.
# mark_cell(500, 400, START)

# Display original PCB
cv2.imshow("PCB Layout", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a grayscale image for inspection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect image edges
edges = cv2.Canny(gray, 50, 150)

# Save the diagnostic outputs
cv2.imwrite("output/pcb_grayscale.jpg", gray)
cv2.imwrite("output/pcb_edges.jpg", edges)

print("Saved output/pcb_grayscale.jpg")
print("Saved output/pcb_edges.jpg")
