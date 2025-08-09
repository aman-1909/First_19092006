import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
import os
from datetime import datetime

# Hide the Tkinter root window (we just want the file dialog)
Tk().withdraw()

# --- Step 1: Let the user choose their .npy files ---
print("Select BEFORE mining .npy file")
before_file = filedialog.askopenfilename(title="Select BEFORE mining .npy file", filetypes=[("NumPy files", "*.npy")])
print("Select AFTER mining .npy file")
after_file = filedialog.askopenfilename(title="Select AFTER mining .npy file", filetypes=[("NumPy files", "*.npy")])

if not before_file or not after_file:
    print("File selection cancelled. Exiting...")
    exit()

# --- Step 2: Load the DEM data ---
before = np.load(before_file)
after = np.load(after_file)

# --- Step 3: Mining detection logic ---
difference = before - after
threshold = 1.0  # Elevation drop threshold in meters
mining_mask = difference > threshold
mining_area = np.sum(mining_mask)
mining_percentage = (mining_area / difference.size) * 100

# --- Step 4: Plot before, after, and difference ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
im1 = axes[0].imshow(before, cmap="terrain")
axes[0].set_title("Before Mining")
plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

im2 = axes[1].imshow(after, cmap="terrain")
axes[1].set_title("After Mining")
plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

im3 = axes[2].imshow(mining_mask, cmap="Reds")
axes[2].set_title("Detected Mining Areas")
plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()

# Save the plot image for PDF
plot_filename = "mining_detection.png"
plt.savefig(plot_filename, dpi=300)
plt.show()

# --- Step 5: PDF Export ---
pdf_filename = "EcoSandWatch_Report.pdf"
c = canvas.Canvas(pdf_filename, pagesize=A4)
width, height = A4

# Title
c.setFont("Helvetica-Bold", 20)
c.drawString(50, height - 50, "Eco-SandWatch Mining Detection Report")

# Date
c.setFont("Helvetica", 10)
c.drawString(50, height - 70, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Branding tagline
c.setFont("Helvetica-Oblique", 12)
c.drawString(50, height - 90, "AI-powered illegal sand mining detection")

# Summary
summary_text = [
    f"Before file: {os.path.basename(before_file)}",
    f"After file: {os.path.basename(after_file)}",
    f"Mining Area Pixels: {mining_area}",
    f"Mining Percentage: {mining_percentage:.2f}%",
    f"Detection Threshold: {threshold} m elevation drop"
]

y_pos = height - 120
c.setFont("Helvetica", 11)
for line in summary_text:
    c.drawString(50, y_pos, line)
    y_pos -= 15

# Add plot image to PDF
c.drawImage(plot_filename, 50, 150, width=500, height=300)

# Footer
c.setFont("Helvetica-Oblique", 9)
c.drawString(50, 130, "Eco-SandWatch | Prototype Report")
c.drawString(50, 115, "Disclaimer: This is an AI-assisted analysis and may require field verification.")

# Save PDF
c.save()

print(f"✅ Report generated: {pdf_filename}")
print("You can find it in your current working directory.")