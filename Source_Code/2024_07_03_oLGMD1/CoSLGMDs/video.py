import cv2
import numpy as np

# Video parameters
fps = 30
duration = 3  # seconds
frame_count = fps * duration
height, width = 720, 990  # resolution

# Video Writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec
out = cv2.VideoWriter('expanding_circle.mp4', fourcc, fps, (width, height))

# Create each frame
for frame_idx in range(frame_count):
    # White background
    frame = np.ones((height, width), dtype=np.uint8) * 255  # pure white background

    # Calculate the radius of the black circle
    max_radius = int(np.sqrt(height**2 + width**2) / 2)  # max radius to cover the frame
    current_radius = int(max_radius * (frame_idx / frame_count))  # current radius

    # Draw the black circle in the center
    center = (width // 2, height // 2)
    cv2.circle(frame, center, current_radius, 0, -1)  # black circle, filled

    # Write the frame
    out.write(cv2.merge([frame, frame, frame]))  # Convert single channel to 3 channel

# Release the video writer
out.release()

# Confirm that the video has been created
"Video creation complete: expanding_circle.mp4"
