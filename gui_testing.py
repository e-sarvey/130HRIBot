import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import time

# Initialize OpenCV camera
vid = cv2.VideoCapture(0)

# Set up DearPyGui window and texture
dpg.create_context()
dpg.create_viewport(title="Camera Feed", width=800, height=600)

# Set up texture dimensions based on desired window size
frame_width, frame_height = 640, 480
texture_data = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)  # Placeholder data for the texture
with dpg.texture_registry():
    texture_id = dpg.add_raw_texture(frame_width, frame_height, texture_data, format=dpg.mvFormat_Float_rgb)

# DearPyGui window and image display
with dpg.window(label="Camera Feed Window"):
    dpg.add_image(texture_id)

def update_texture():
    while True:
        ret, frame = vid.read()
        if not ret:
            break

        # Convert the frame to RGB (DearPyGui expects RGB format)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame if necessary to fit the window dimensions
        frame_resized = cv2.resize(frame, (frame_width, frame_height))

        # Update the texture with the new frame data
        dpg.set_value(texture_id, frame_resized.flatten() / 255.0)  # Normalize to 0-1 for DearPyGui

        # Control the frame rate (optional)
        time.sleep(1 / 30)  # Target 30 FPS

# Start DearPyGui
dpg.create_viewport()
dpg.setup_dearpygui()
dpg.show_viewport()

# Run the update function in a new thread so DearPyGui stays responsive
import threading
threading.Thread(target=update_texture, daemon=True).start()

dpg.start_dearpygui()
dpg.destroy_context()

# Release camera after closing DearPyGui
vid.release()
