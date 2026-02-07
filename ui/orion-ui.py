import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter, ImageTk
import math
import threading
import time

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class AnimatedOrb:
    def __init__(self, canvas, x, y, size=200):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.size = size
        self.phase = 0
        self.running = True

        # Start animation thread
        self.animation_thread = threading.Thread(
            target=self.animate, daemon=True)
        self.animation_thread.start()

    def create_orb_image(self, intensity):
        # Create image with transparency
        img = Image.new('RGBA', (self.size, self.size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        center_x = self.size // 2
        center_y = self.size // 2

        # Adjusted glow layers with reduced blur
        glow_layers = [
            (50, 20, (0, 220, 255)),    # Outermost glow
            (40, 30, (0, 200, 255)),    # Mid glow
            (30, 40, (0, 180, 255)),    # Inner glow
            (20, 60, (100, 240, 255)),  # Core glow
        ]

        for radius, alpha, color in glow_layers:
            current_alpha = int(alpha * intensity)
            glow_color = (*color, current_alpha)
            bbox = [center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius]
            draw.ellipse(bbox, fill=glow_color)

        # Apply lighter blur for sharper glow
        img = img.filter(ImageFilter.GaussianBlur(radius=4))

        # Main orb body with gradient
        main_radius = 50
        for i in range(15):
            r = main_radius - (i * 2.5)
            if r <= 0:
                break
            alpha = int((150 - i * 8) * intensity)
            if alpha <= 0:
                continue
            if i < 5:
                color = (100, 240, 255, alpha)
            elif i < 10:
                color = (0, 200, 255, alpha)
            else:
                color = (0, 150, 200, alpha)
            bbox = [center_x - r, center_y - r, center_x + r, center_y + r]
            draw.ellipse(bbox, fill=color)

        # Bright ring
        ring_alpha = int(120 * intensity)
        ring_width = 2
        ring_radius = main_radius - 5
        for w in range(ring_width):
            bbox = [center_x - ring_radius - w, center_y - ring_radius - w,
                    center_x + ring_radius + w, center_y + ring_radius + w]
            draw.ellipse(bbox, outline=(150, 250, 255, ring_alpha))

        return img

    def animate(self):
        while self.running:
            self.phase += 0.07
            if self.phase > 2 * math.pi:
                self.phase = 0
            intensity = 0.7 + 0.3 * (0.5 + 0.5 * math.sin(self.phase))
            orb_img = self.create_orb_image(intensity)
            orb_photo = ImageTk.PhotoImage(orb_img)
            self.canvas.after(0, self.update_canvas, orb_photo)
            time.sleep(1/30)

    def update_canvas(self, photo):
        self.canvas.delete("orb")
        self.canvas.create_image(self.x, self.y, image=photo, tags="orb")
        self.canvas.image = photo

    def stop(self):
        self.running = False


class OrionAssistant:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Orion v2 Assistant")
        self.window.geometry("400x600")  # Match reference size
        self.window.resizable(False, False)
        self.window.configure(fg_color="#0f0f23")

        self.setup_ui()
        self.orb = None

    def setup_ui(self):
        # Main frame with centered layout
        main_frame = ctk.CTkFrame(self.window, fg_color="transparent")
        main_frame.pack(expand=True, fill="both", padx=20, pady=20)

        # Center everything with grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure((0, 1, 2, 3, 4), weight=0)
        main_frame.grid_rowconfigure(5, weight=1)  # Spacer

        # Orb canvas
        self.orb_canvas = tk.Canvas(
            main_frame,
            width=200,
            height=200,
            bg="#0f0f23",
            highlightthickness=0,
            bd=0
        )
        self.orb_canvas.grid(row=0, column=0, pady=(0, 20))

        self.orb = AnimatedOrb(self.orb_canvas, 100,
                               100)  # Centered at 100,100

        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="Orion v2 Assistant",
            font=ctk.CTkFont(size=24, weight="normal"),
            text_color="#ffffff"
        )
        title.grid(row=1, column=0, pady=(0, 10))

        # Status
        self.status = ctk.CTkLabel(
            main_frame,
            text="Status: Listening...",
            font=ctk.CTkFont(size=14),
            text_color="#c8c8c8"
        )
        self.status.grid(row=2, column=0, pady=(0, 20))

        # Spacer
        spacer = ctk.CTkFrame(main_frame, fg_color="transparent")
        spacer.grid(row=3, column=0, pady=10)

        # Input frame
        input_frame = ctk.CTkFrame(
            main_frame, fg_color="transparent", height=50)
        input_frame.grid(row=4, column=0, pady=(0, 20), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=0)

        # Text input
        self.text_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="",
            font=ctk.CTkFont(size=14),
            height=40,
            border_width=1,
            corner_radius=10,
            fg_color="#1e1e1e",
            border_color="#404040",
            placeholder_text_color="#999999",
            text_color="#ffffff"
        )
        self.text_input.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.text_input.bind("<Return>", self.send_message)

        # Send button
        self.send_button = ctk.CTkButton(
            input_frame,
            text="Send",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=40,
            width=70,
            corner_radius=10,
            fg_color="#1e1e1e",
            hover_color="#2e2e2e",
            border_width=1,
            border_color="#404040",
            command=self.send_message
        )
        self.send_button.grid(row=0, column=1, sticky="e")

        self.text_input.focus()

    def send_message(self, event=None):
        message = self.text_input.get().strip()
        if message:
            print(f"Sending message: {message}")
            self.status.configure(text="Status: Processing...")
            self.text_input.delete(0, 'end')
            self.window.after(2000, lambda: self.status.configure(
                text="Status: Listening..."))

    def run(self):
        try:
            self.window.mainloop()
        finally:
            if self.orb:
                self.orb.stop()


if __name__ == "__main__":
    app = OrionAssistant()
    app.run()
