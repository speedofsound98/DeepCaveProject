import threading
import tkinter as tk
from tkinter import messagebox
import os
from PIL import Image, ImageTk, ImageDraw
import PIL
from file_system import filter_table, DATABASE_PATH, update_record, create_record

"""photo_class_with_cut_and_cave_name.pyw
Tkinter-based labeling tool for categorizing cave photos.

- Loads images from a directory.
- Allows the user to crop (cut) a region of interest and assign a category / 
cave name.
- Saves annotations to disk (e.g., CSV/JSON) for later training or curation.
- Provides quick keyboard shortcuts for efficient labeling.

Usage:
    python photo_class_with_cut_and_cave_name.pyw --images <folder> 
    --out labels.json
"""

# Sample categories for classification - adding new categories
CATEGORIES = ["פתח", "מפה", "פתח מבפנים", "אחר"]
# Add your new categories here
NEW_CATEGORIES = ["ארכיאולוגיה", "משקעי מערות", "עטלף", "בעלי חיים", "פנורמה \ תמונה מרחבית"]

# Colors and font sizes - for original categories
BG = ["green", 'blue', 'black', 'light gray']
FG = ['white', 'white', 'white', 'black']
FONT_SIZES = [17, 17, 17, 15]

# Add colors and font sizes for new categories
NEW_BG = [ 'light gray', 'light gray', 'light gray', 'light gray', 'light gray']
NEW_FG = ['black', 'black', 'black', 'black', 'black']
NEW_FONT_SIZES = [15, 15, 15, 15, 15]

PICTURES_TABLE = 'tbleSBboDC5DMiJLw'


class CaveClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cave Image Classifier")
        self.root.state('zoomed')  # Set window to maximized
        self.username = ""
        self.root.config(bg='white')

        self.image_records = None
        self.images = None
        self.current_image_index = 0
        self.rotations = 0

        self.header_frame = None
        self.cave_name_label = None

        self.next_image = None
        self.pre_fetch = None

        # Crop mode variables
        self.is_crop_mode = False
        self.crop_start_x = None
        self.crop_start_y = None
        self.crop_rect = None
        self.original_image = None
        self.current_displayed_image = None
        self.scale_factor_width = 1.0
        self.scale_factor_height = 1.0
        self.canvas_img_id = None

        self.create_login_screen()

    def create_login_screen(self):
        """Creates the login screen."""
        self.clear_screen()
        frame = tk.Frame(self.root, bg='white')
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)  # Centering the frame

        tk.Label(frame, text=":שם מלא", font=('Ariel', 15), bg='white').pack(side=tk.RIGHT, padx=10, anchor=tk.E)

        self.username_entry = tk.Entry(frame, font=('Ariel', 15))
        self.username_entry.pack(padx=5, side=tk.RIGHT)

        tk.Button(frame, text="התחברות", font=('Ariel', 13), command=self.login).pack(padx=10)

    def login(self):
        """Handles user login."""
        username = self.username_entry.get().strip()
        if username:
            self.username = username
            self.create_main_screen()
        else:
            messagebox.showerror("Login Error", "Please enter a username.")

    def create_main_screen(self):
        """Creates the main image classification screen."""
        self.image_records = fetch_assigned_photos(self.username)
        self.images = [os.path.join(DATABASE_PATH, record['fields']['path']) for record in self.image_records]
        self.clear_screen()

        # Header frame
        self.header_frame = tk.Frame(self.root, bg='white')
        self.header_frame.pack(fill=tk.X, pady=5)

        # User greeting label
        tk.Label(self.header_frame, text="שלום " + f"{self.username}", font=("Arial", 12, "bold"), bg='white').pack()

        # Create a label for the cave name that will be updated for each image
        self.cave_name_label = tk.Label(self.header_frame, text="", font=("Arial", 12, "bold"), bg='white')
        self.cave_name_label.pack()

        # Create canvas frame that fills available space
        self.canvas_frame = tk.Frame(self.root, bg='white')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Create canvas that fills the frame
        self.image_canvas = tk.Canvas(self.canvas_frame, bg='white', highlightthickness=0)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind mouse events for cropping
        self.image_canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.image_canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Bind window resize event to recenter the image
        self.root.bind("<Configure>", self.on_window_resize)

        # Create a centered button frame at the bottom
        button_frame_container = tk.Frame(self.root, bg='white')
        button_frame_container.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        # Create separate frames for each row of buttons
        button_frame1 = tk.Frame(button_frame_container, bg='white')
        button_frame1.pack(side=tk.TOP, anchor=tk.CENTER, pady=(0, 5))

        button_frame2 = tk.Frame(button_frame_container, bg='white')
        button_frame2.pack(side=tk.TOP, anchor=tk.CENTER)

        # Add back button on the left of first row
        tk.Button(button_frame1, text='חזרה', font=('Ariel', 15, 'bold'),
                  command=self.backwards).pack(side=tk.LEFT, padx=5)

        # Add original category buttons in the first row
        for i in range(len(CATEGORIES))[::-1]:
            tk.Button(button_frame1, text=CATEGORIES[i], font=('Ariel', FONT_SIZES[i], 'bold'), bg=BG[i], fg=FG[i],
                      command=lambda c=CATEGORIES[i]: self.classify_image(c)).pack(side=tk.LEFT, padx=5)

        # Add rotate button to first row
        tk.Button(button_frame1, text=' ↺ ', font=('Ariel', 17),
                  command=self.rotate).pack(side=tk.LEFT, padx=5)

        # Add crop button to first row
        self.crop_button = tk.Button(button_frame1, text="✂️ חתוך (פתח)", font=('Ariel', 15, 'bold'),
                                     command=self.toggle_crop_mode)
        self.crop_button.pack(side=tk.LEFT, padx=5)

        # Add new category buttons to the second row
        for i in range(len(NEW_CATEGORIES))[::-1]:
            tk.Button(button_frame2, text=NEW_CATEGORIES[i], font=('Ariel', NEW_FONT_SIZES[i], 'bold'),
                      bg=NEW_BG[i], fg=NEW_FG[i],
                      command=lambda c=NEW_CATEGORIES[i]: self.classify_image(c)).pack(side=tk.LEFT, padx=5)

        # Load the first image
        self.load_next_image()

    def on_window_resize(self, event):
        """Handle window resize event to recenter the image"""
        # Only respond to main window resizes, not child widget resizes
        if event.widget == self.root:
            # Wait a bit to ensure all layout calculations are done
            self.root.after(100, self.center_image)

    def center_image(self):
        """Center the current image in the canvas"""
        if self.tk_image:
            # Get canvas dimensions
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()

            # Get image dimensions
            img_width = self.tk_image.width()
            img_height = self.tk_image.height()

            # Calculate center position
            x_center = canvas_width // 2
            y_center = canvas_height // 2

            # If we already have an image on the canvas, move it
            if self.canvas_img_id is not None:
                self.image_canvas.coords(self.canvas_img_id, x_center, y_center)
            # If no image yet (shouldn't happen normally), create one
            else:
                self.canvas_img_id = self.image_canvas.create_image(
                    x_center, y_center, image=self.tk_image, anchor=tk.CENTER
                )

    def toggle_crop_mode(self):
        """Toggle between crop mode and normal mode"""
        self.is_crop_mode = not self.is_crop_mode
        if self.is_crop_mode:
            self.crop_button.config(text="✓ סיים חיתוך", bg='light green')
            # messagebox.showinfo("Crop Mode", "Drag to select an area to crop.
            # Press the crop button again to save.")
        else:
            self.crop_button.config(text="✂️ חתוך (פתח)", bg='SystemButtonFace')
            if self.crop_rect:
                self.save_cropped_area()

    def on_mouse_down(self, event):
        """Handle mouse down event for cropping"""
        if self.is_crop_mode:
            self.crop_start_x = self.image_canvas.canvasx(event.x)
            self.crop_start_y = self.image_canvas.canvasy(event.y)
            # Remove previous rectangle if it exists
            if self.crop_rect:
                self.image_canvas.delete(self.crop_rect)
                self.crop_rect = None

    def on_mouse_drag(self, event):
        """Handle mouse drag event for cropping"""
        if self.is_crop_mode and self.crop_start_x is not None and self.crop_start_y is not None:
            cur_x = self.image_canvas.canvasx(event.x)
            cur_y = self.image_canvas.canvasy(event.y)

            # Remove previous rectangle if it exists
            if self.crop_rect:
                self.image_canvas.delete(self.crop_rect)

            # Draw new rectangle
            self.crop_rect = self.image_canvas.create_rectangle(
                self.crop_start_x, self.crop_start_y, cur_x, cur_y,
                outline='red', width=2
            )

    def on_mouse_up(self, event):
        """Handle mouse up event for cropping"""
        # Just finish drawing the rectangle, but don't apply the crop until the
        # button is clicked
        pass

    def save_cropped_area(self):
        """Save the cropped area as a new image"""
        if self.crop_rect and self.original_image:
            # Get the coordinates from the canvas
            x1, y1, x2, y2 = self.image_canvas.coords(self.crop_rect)

            # Make sure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Get canvas dimensions and image position
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()
            img_x, img_y = self.image_canvas.coords(self.canvas_img_id)
            img_width = self.tk_image.width()
            img_height = self.tk_image.height()

            # Calculate image boundaries
            img_left = img_x - (img_width / 2)
            img_top = img_y - (img_height / 2)

            # Adjust coordinates relative to image position
            x1 = x1 - img_left
            y1 = y1 - img_top
            x2 = x2 - img_left
            y2 = y2 - img_top

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(0, min(x2, img_width))
            y2 = max(0, min(y2, img_height))

            # Convert canvas coordinates to original image coordinates
            orig_x1 = int(x1 / self.scale_factor_width)
            orig_y1 = int(y1 / self.scale_factor_height)
            orig_x2 = int(x2 / self.scale_factor_width)
            orig_y2 = int(y2 / self.scale_factor_height)

            # Crop the original image
            cropped_img = self.original_image.crop((orig_x1, orig_y1, orig_x2, orig_y2))

            # Get the original file path and create a name for the cropped image
            original_path = self.images[self.current_image_index]
            dir_name = os.path.dirname(original_path)
            base_name = os.path.basename(original_path)
            name, ext = os.path.splitext(base_name)

            # Create new filename with "_cropped" suffix
            new_name = f"{name}_cropped{ext}"
            cropped_path = os.path.join(dir_name, new_name)

            # Save the cropped image
            cropped_img.save(cropped_path)
            # messagebox.showinfo("Success",
            # f"Cropped image saved as:\n{cropped_path}")

            # Save new record for the cropped image
            self.save_new_record(new_name, cropped_path)

            # Clean up
            self.image_canvas.delete(self.crop_rect)
            self.crop_rect = None
            self.crop_start_x = None
            self.crop_start_y = None

    def save_new_record(self, name, new_path):
        current_record = self.image_records[self.current_image_index]
        path = new_path
        file_name = name

        region = current_record['fields'].get('region', '')
        cave_name = current_record['fields'].get('cave_name', '')
        cave_access = current_record['fields'].get('cave_access', '')
        assignee = current_record['fields'].get('assignee', '')
        new_record = {'path': path,
                      'region': region,
                      'cave_name': cave_name,
                      'cave_access': cave_access,
                      'file_name': file_name,
                      'assignee': assignee,
                      'type': "פתח חתוך"}
        create_record(PICTURES_TABLE, new_record)

    def load_next_image(self):
        """Loads and displays the next image to fill available screen space
        without distorting the aspect ratio."""
        if self.current_image_index < len(self.images):
            if self.pre_fetch is not None:
                self.pre_fetch.join()

            # Update the cave name in the header
            current_record = self.image_records[self.current_image_index]
            cave_name = current_record['fields'].get('cave_name', '')
            if cave_name:
                self.cave_name_label.config(text=f"מערה: {cave_name}")
            else:
                self.cave_name_label.config(text="")

            if self.next_image is not None and self.current_image_index == self.next_image[0]:
                image = self.next_image[1]
            else:
                try:
                    image = Image.open(self.images[self.current_image_index])
                except PIL.UnidentifiedImageError:
                    print('Bad file, skipping...')
                    self.current_image_index += 1
                    self.load_next_image()
                    return

            self.pre_fetch = threading.Thread(target=self.fetch_next_image)
            self.pre_fetch.start()

            # Store the original image before any transformations
            self.original_image = image.copy()

            for _ in range(self.rotations):
                image = image.transpose(Image.Transpose.ROTATE_90)
                self.original_image = self.original_image.transpose(Image.Transpose.ROTATE_90)

            # Get current dimensions of the canvas
            self.image_canvas.update()
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()

            # If canvas dimensions are very small, use window size as fallback
            if canvas_width < 100 or canvas_height < 100:
                canvas_width = self.root.winfo_width() - 40  # Some padding
                canvas_height = self.root.winfo_height() - 200  # Accounting for
                # buttons and header

            # Get original image dimensions
            img_width, img_height = image.size

            # Calculate the scaling factor while maintaining the aspect ratio
            aspect_ratio = img_width / img_height
            if canvas_width / canvas_height > aspect_ratio:
                # Fit based on height
                new_height = canvas_height
                new_width = int(aspect_ratio * new_height)
            else:
                # Fit based on width
                new_width = canvas_width
                new_height = int(new_width / aspect_ratio)

            # Calculate scale factors for later use in cropping
            self.scale_factor_width = new_width / img_width
            self.scale_factor_height = new_height / img_height

            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.current_displayed_image = resized_image
            self.tk_image = ImageTk.PhotoImage(resized_image)

            # Clear canvas
            self.image_canvas.delete("all")

            # Calculate center position
            x_center = canvas_width // 2
            y_center = canvas_height // 2

            # Display the image centered
            self.canvas_img_id = self.image_canvas.create_image(
                x_center, y_center, image=self.tk_image, anchor=tk.CENTER
            )

            # Reset crop-related variables
            self.crop_rect = None
            self.crop_start_x = None
            self.crop_start_y = None
        else:
            messagebox.showinfo("Completed", "No more images to classify!")
            self.root.destroy()

    def classify_image(self, category):
        """Handles image classification and moves to the next image."""

        def update():
            update_record(PICTURES_TABLE, self.image_records[self.current_image_index]['id'], {'type': category})

        threading.Thread(target=update).start()

        self.current_image_index += 1
        self.rotations = 0
        self.load_next_image()

    def clear_screen(self):
        """Removes all widgets from the window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def backwards(self):
        if self.current_image_index:
            self.current_image_index -= 1
            self.load_next_image()

    def rotate(self):
        self.rotations += 1
        self.load_next_image()

    def fetch_next_image(self):
        if self.current_image_index < len(self.images) - 1:
            try:
                next_img = Image.open(self.images[self.current_image_index + 1])
                self.next_image = (self.current_image_index + 1, next_img)
            except (PIL.UnidentifiedImageError, FileNotFoundError):
                print(f'Error loading next image: {self.images[self.current_image_index + 1]}')
                self.next_image = None


def fetch_assigned_photos(username):
    records = filter_table(PICTURES_TABLE, "and(assignee = '" + username + "', type='')")
    return records


if __name__ == "__main__":
    root = tk.Tk()
    app = CaveClassifierApp(root)
    root.mainloop()