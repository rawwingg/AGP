import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from security import Guard, Security


BASE_DIR = Path(__file__).resolve().parent
LOCAL_DATASET_PATH = BASE_DIR / "Polygons" / "local_dataset.json"
CANVAS_PADDING = 44
REMOVE_RADIUS = 14


class PolygonGuardViewer(tk.Tk):
    def __init__(self, data_path=LOCAL_DATASET_PATH):
        super().__init__()
        self.title("Polygon Guard Viewer")
        self.geometry("1120x720")
        self.minsize(860, 560)

        self.data_path = Path(data_path)
        self.polygons = self.load_polygons()
        self.selected_index = 0
        self.mode = tk.StringVar(value="add")
        self.transform = None

        for record in self.polygons:
            record.setdefault("guards", [])

        self.configure(bg="#eef2f7")
        self.create_styles()
        self.create_layout()

        if self.polygons:
            self.listbox.selection_set(0)
            self.draw_selected_polygon()

    def create_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Main.TFrame", background="#eef2f7")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Toolbar.TFrame", background="#ffffff")
        style.configure(
            "Title.TLabel",
            background="#ffffff",
            foreground="#111827",
            font=("Segoe UI", 17, "bold"),
        )
        style.configure(
            "Meta.TLabel",
            background="#ffffff",
            foreground="#4b5563",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Status.TLabel",
            background="#ffffff",
            foreground="#374151",
            font=("Segoe UI", 10),
        )
        style.configure("TButton", font=("Segoe UI", 10), padding=(12, 7))
        style.configure("Toolbutton", font=("Segoe UI", 10), padding=(12, 7))
        style.map(
            "Toolbutton",
            background=[("selected", "#dbeafe")],
            foreground=[("selected", "#1d4ed8")],
        )

    def create_layout(self):
        root = ttk.Frame(self, style="Main.TFrame", padding=16)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(root, style="Panel.TFrame", padding=14)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 16))

        ttk.Label(sidebar, text="Dataset", style="Title.TLabel").pack(anchor="w")
        ttk.Label(sidebar, text=str(self.data_path), style="Meta.TLabel").pack(anchor="w", pady=(2, 12))

        self.listbox = tk.Listbox(
            sidebar,
            width=28,
            height=25,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#d1d5db",
            activestyle="none",
            font=("Segoe UI", 10),
            selectbackground="#2563eb",
            selectforeground="#ffffff",
        )
        self.listbox.pack(fill=tk.BOTH, expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_select_polygon)

        for record in self.polygons:
            difficulty = record.get("difficulty", "unknown")
            self.listbox.insert(tk.END, f"#{record['id']}  {difficulty}")

        nav = ttk.Frame(sidebar, style="Panel.TFrame")
        nav.pack(fill=tk.X, pady=(12, 0))
        ttk.Button(nav, text="Previous", command=self.show_previous).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next", command=self.show_next).pack(side=tk.RIGHT)

        main_panel = ttk.Frame(root, style="Panel.TFrame", padding=14)
        main_panel.grid(row=0, column=1, sticky="nsew")
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(2, weight=1)

        header = ttk.Frame(main_panel, style="Panel.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(header, text="Polygon", style="Title.TLabel")
        self.title_label.grid(row=0, column=0, sticky="w")

        self.meta_label = ttk.Label(header, text="", style="Meta.TLabel")
        self.meta_label.grid(row=1, column=0, sticky="w", pady=(3, 0))

        toolbar = ttk.Frame(main_panel, style="Toolbar.TFrame")
        toolbar.grid(row=1, column=0, sticky="ew", pady=(12, 10))

        ttk.Radiobutton(
            toolbar,
            text="Add guard",
            value="add",
            variable=self.mode,
            style="Toolbutton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Radiobutton(
            toolbar,
            text="Remove guard",
            value="remove",
            variable=self.mode,
            style="Toolbutton",
        ).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(toolbar, text="Reset guards", command=self.reset_guards).pack(side=tk.LEFT, padx=(8, 0))

        self.status_label = ttk.Label(
            toolbar,
            text="Click inside the polygon to add a guard.",
            style="Status.TLabel",
        )
        self.status_label.pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(
            main_panel,
            background="#f8fafc",
            highlightthickness=1,
            highlightbackground="#cbd5e1",
            cursor="crosshair",
        )
        self.canvas.grid(row=2, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self.draw_selected_polygon())
        self.canvas.bind("<Button-1>", self.on_canvas_click)

    def load_polygons(self):
        if not self.data_path.exists():
            messagebox.showerror(
                "Missing dataset",
                f"Could not find {self.data_path}.\nRun generate_polygon_dataset.py first.",
            )
            return []

        return json.loads(self.data_path.read_text(encoding="utf-8"))

    def on_select_polygon(self, _event):
        selection = self.listbox.curselection()
        if not selection:
            return

        self.selected_index = selection[0]
        self.set_status("Click inside the polygon to add a guard.")
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
        self.set_status("Click inside the polygon to add a guard.")
        self.draw_selected_polygon()

    def on_canvas_click(self, event):
        if not self.polygons or self.transform is None:
            return

        if self.mode.get() == "remove":
            self.remove_guard_at(event.x, event.y)
            return

        point = self.canvas_to_data(event.x, event.y)
        record = self.polygons[self.selected_index]
        security = Security(record)
        guard = Guard(*point)

        try:
            security.add_guard(guard)
        except ValueError:
            self.show_error("That point is outside the polygon or inside an obstacle.")
            return

        record["guards"].append([point[0], point[1]])
        self.set_status(f"Added guard at ({point[0]:.1f}, {point[1]:.1f}).")
        self.draw_selected_polygon()

    def remove_guard_at(self, canvas_x, canvas_y):
        record = self.polygons[self.selected_index]
        guards = record.get("guards", [])
        if not guards:
            self.show_error("There are no guards to remove.")
            return

        closest_index = None
        closest_distance = float("inf")
        for index, guard in enumerate(guards):
            gx, gy = self.data_to_canvas(guard[0], guard[1])
            distance = math.hypot(gx - canvas_x, gy - canvas_y)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = index

        if closest_index is None or closest_distance > REMOVE_RADIUS:
            self.show_error("Click closer to an existing guard to remove it.")
            return

        removed = guards.pop(closest_index)
        self.set_status(f"Removed guard at ({removed[0]:.1f}, {removed[1]:.1f}).")
        self.draw_selected_polygon()

    def reset_guards(self):
        if not self.polygons:
            return
        self.polygons[self.selected_index]["guards"] = []
        self.set_status("Removed all guards from this polygon.")
        self.draw_selected_polygon()

    def draw_selected_polygon(self):
        self.canvas.delete("all")
        if not self.polygons:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="No polygons loaded",
                fill="#6b7280",
                font=("Segoe UI", 13),
            )
            return

        record = self.polygons[self.selected_index]
        outer_points = record["outer_points"]
        holes = record["holes"]
        guards = record.get("guards", [])
        all_points = outer_points + [point for hole in holes for point in hole]
        self.update_transform(all_points)

        security = Security(record)
        for guard in guards:
            try:
                security.add_guard(Guard(guard[0], guard[1]))
            except ValueError:
                continue

        difficulty = record.get("difficulty", "unknown")
        self.title_label.configure(text=f"Polygon {record['id']}  |  {difficulty}")
        self.meta_label.configure(
            text=(
                f"{len(outer_points)} outer points  |  {len(holes)} holes  |  "
                f"area {record['area']:.2f}  |  guards {len(guards)}  |  "
                f"coverage {security.percent_coverage * 100:.2f}%"
            )
        )

        self.canvas.create_polygon(
            self.scale_points(outer_points),
            fill="#dbeafe",
            outline="#1d4ed8",
            width=3,
        )

        for guard in guards:
            self.draw_guard_coverage(Guard(guard[0], guard[1]), security)

        for hole in holes:
            self.canvas.create_polygon(
                self.scale_points(hole),
                fill="#f8fafc",
                outline="#dc2626",
                width=2,
            )

        for guard in guards:
            gx, gy = self.data_to_canvas(guard[0], guard[1])
            self.draw_point(gx, gy, size=12, color="#111827")

    def draw_guard_coverage(self, guard, security):
        coverage_polygon = security.get_area_coverage_of_guard(guard)
        if coverage_polygon is None or coverage_polygon.is_empty:
            return

        if coverage_polygon.geom_type == "Polygon":
            polygons = [coverage_polygon]
        elif coverage_polygon.geom_type.startswith("Multi") or coverage_polygon.geom_type == "GeometryCollection":
            polygons = [geom for geom in coverage_polygon.geoms if geom.geom_type == "Polygon"]
        else:
            return

        for polygon in polygons:
            self.canvas.create_polygon(
                self.scale_points(list(polygon.exterior.coords)),
                fill="#fde047",
                outline="",
                stipple="gray50",
            )

    def update_transform(self, all_points):
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

        self.transform = {
            "min_x": min_x,
            "min_y": min_y,
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
        }

    def scale_points(self, points):
        scaled = []
        for x, y in points:
            cx, cy = self.data_to_canvas(x, y)
            scaled.extend([cx, cy])
        return scaled

    def data_to_canvas(self, x, y):
        transform = self.transform
        return (
            transform["offset_x"] + (x - transform["min_x"]) * transform["scale"],
            transform["offset_y"] + (y - transform["min_y"]) * transform["scale"],
        )

    def canvas_to_data(self, x, y):
        transform = self.transform
        return (
            transform["min_x"] + (x - transform["offset_x"]) / transform["scale"],
            transform["min_y"] + (y - transform["offset_y"]) / transform["scale"],
        )

    def draw_point(self, x, y, size=5, color="#111827"):
        radius = size / 2
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill=color,
            outline="#ffffff",
            width=2,
        )

    def show_error(self, message):
        self.set_status(message)
        messagebox.showwarning("Invalid action", message)

    def set_status(self, message):
        self.status_label.configure(text=message)


if __name__ == "__main__":
    app = PolygonGuardViewer()
    app.mainloop()
