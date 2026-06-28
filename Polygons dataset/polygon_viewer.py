import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk


DATA_PATH = Path("Polygons") / "polygons.json"
LOCAL_DATASET_PATH = Path("Polygons") / "local_dataset.json"
CANVAS_PADDING = 40


class PolygonViewer(tk.Tk):
    def __init__(self, data_path=LOCAL_DATASET_PATH):
        super().__init__()
        self.title("Polygon Dataset Viewer")
        self.geometry("980x680")
        self.minsize(760, 520)

        self.data_path = Path(data_path)
        self.polygons = self.load_polygons()
        self.selected_index = 0

        self.configure(bg="#f3f4f6")
        self.create_styles()
        self.create_layout()

        if self.polygons:
            self.listbox.selection_set(0)
            self.draw_selected_polygon()

    def create_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Main.TFrame", background="#f3f4f6")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#ffffff", foreground="#111827",
                        font=("Segoe UI", 16, "bold"))
        style.configure("Meta.TLabel", background="#ffffff", foreground="#4b5563",
                        font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 6))

    def create_layout(self):
        root = ttk.Frame(self, style="Main.TFrame", padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, style="Panel.TFrame", padding=12)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 16))

        ttk.Label(sidebar, text="Polygons", style="Title.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text=f"Loaded from {self.data_path}", style="Meta.TLabel").pack(anchor="w", pady=(2, 12))

        self.listbox = tk.Listbox(
            sidebar,
            width=24,
            height=24,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#e5e7eb",
            activestyle="none",
            font=("Segoe UI", 10),
            selectbackground="#2563eb",
            selectforeground="#ffffff"
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_polygon)

        for record in self.polygons:
            self.listbox.insert(tk.END, f"Polygon {record['id']}")

        controls = ttk.Frame(sidebar, style="Panel.TFrame")
        controls.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(controls, text="Previous", command=self.show_previous).pack(side=tk.LEFT)
        ttk.Button(controls, text="Next", command=self.show_next).pack(side=tk.RIGHT)

        main_panel = ttk.Frame(root, style="Panel.TFrame", padding=14)
        main_panel.grid(row=0, column=1, sticky="nsew")
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(1, weight=1)

        header = ttk.Frame(main_panel, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        header.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(header, text="Polygon", style="Title.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w")

        self.meta_label = ttk.Label(header, text="", style="Meta.TLabel")
        self.meta_label.grid(row=1, column=0, sticky="w", pady=(3, 0))

        self.canvas = tk.Canvas(
            main_panel,
            background="#f9fafb",
            highlightthickness=1,
            highlightbackground="#e5e7eb"
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.draw_selected_polygon())

    def load_polygons(self):
        if not self.data_path.exists():
            messagebox.showerror(
                "Missing dataset",
                f"Could not find {self.data_path}.\nRun generate_polygon_dataset.py first."
            )
            return []

        return json.loads(self.data_path.read_text(encoding="utf-8"))

    def on_select_polygon(self, _event):
        selection = self.listbox.curselection()
        if not selection:
            return

        self.selected_index = selection[0]
        self.draw_selected_polygon()

    def show_previous(self):
        if not self.polygons:
            return
        self.selected_index = (self.selected_index - 1) % len(self.polygons)
        self.select_current_index()

    def show_next(self):
        if not self.polygons:
            return
        self.selected_index = (self.selected_index + 1) % len(self.polygons)
        self.select_current_index()

    def select_current_index(self):
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(self.selected_index)
        self.listbox.see(self.selected_index)
        self.draw_selected_polygon()

    def draw_selected_polygon(self):
        self.canvas.delete("all")
        if not self.polygons:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="No polygons loaded",
                fill="#6b7280",
                font=("Segoe UI", 13)
            )
            return

        record = self.polygons[self.selected_index]
        outer_points = record["outer_points"]
        holes = record["holes"]

        self.title_label.configure(text=f"Polygon {record['id']}")
        self.meta_label.configure(
            text=f"{len(outer_points)} outer points  |  {len(holes)} holes  |  area {record['area']:.2f}"
        )

        all_points = outer_points + [point for hole in holes for point in hole]
        scaled_outer = self.scale_points(outer_points, all_points)

        self.canvas.create_polygon(
            scaled_outer,
            fill="#dbeafe",
            outline="#1d4ed8",
            width=3
        )

        for hole in holes:
            scaled_hole = self.scale_points(hole, all_points)
            self.canvas.create_polygon(
                scaled_hole,
                fill="#f9fafb",
                outline="#dc2626",
                width=2
            )

    def scale_points(self, points, all_points):
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        usable_width = max(canvas_width - CANVAS_PADDING * 2, 1)
        usable_height = max(canvas_height - CANVAS_PADDING * 2, 1)
        data_width = max(max_x - min_x, 1)
        data_height = max(max_y - min_y, 1)
        scale = min(usable_width / data_width, usable_height / data_height)

        drawn_width = data_width * scale
        drawn_height = data_height * scale
        offset_x = (canvas_width - drawn_width) / 2
        offset_y = (canvas_height - drawn_height) / 2

        scaled = []
        for x, y in points:
            scaled.append(offset_x + (x - min_x) * scale)
            scaled.append(offset_y + (y - min_y) * scale)
        return scaled


if __name__ == "__main__":
    app = PolygonViewer()
    app.mainloop()
