import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# ------------------- Day 4 Part -------------------

# Load a placeholder image as the river (640x480) or your own
img = cv2.imread("river.jpg")

# Resize for consistency
img = cv2.resize(img, (640, 480))

# Simulate random mining zones (10 circles)
for _ in range(10):
    x = random.randint(50, 590)
    y = random.randint(50, 430)
    radius = random.randint(10, 30)
    color = (0, 0, 255)  # Red
    thickness = -1  # Filled circle
    cv2.circle(img, (x, y), radius, color, thickness)

# Save simulated image
cv2.imwrite("day4_simulated_river.jpg", img)

# ------------------- Day 5 Part -------------------

# Load DEM data
before = np.load("before_testing.npy")
after = np.load("after_mining.npy")

# Compute difference
diff = after - before
change_mask = np.abs(diff) > 0.5  # Set threshold

# ------------------- Visualization -------------------

# Convert OpenCV BGR to RGB for Matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(15, 5))

# Show river with mining zones
plt.subplot(1, 3, 1)
plt.title("Simulated Mining Zones")
plt.imshow(img_rgb)
plt.axis("off")

# Show elevation difference
plt.subplot(1, 3, 2)
plt.title("Elevation Difference")
plt.imshow(diff, cmap="coolwarm")
plt.colorbar(label="Δ Height (m)")

# Show mining mask
plt.subplot(1, 3, 3)
plt.title("Detected Zones from DEM")
plt.imshow(change_mask, cmap="gray")
plt.colorbar(label="Change Mask")

plt.tight_layout()
plt.savefig("eco_watch_result.jpg")
plt.show()