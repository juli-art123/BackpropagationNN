import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageFilter

import NeuralNetwork

class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Розпізнавач цифр")
        self.geometry("640x520")
        self.resizable(width=False, height=False)
        self.nn_manager = NeuralNetwork.NNManager("model.npz")
        self.nn = self.nn_manager.load()
        self._build_ui()
        self.update_idletasks()
        if self.nn is None:
            self._set_ui_state(enabled=False)
            dialog = TrainModelDialog(self)
            if dialog.result:
                self._start_training_thread()
            else:
                self.destroy()
        else:
            self._set_ui_state(enabled=True)
            self.epoch = self.nn.epoch
            self.status_label.config(text=f"Модель навчена за {self.epoch} епох на датасеті MNIST.")

    def _build_ui(self):
        tk.Label(
            text="Ця програма використовує штучну нейронну мережу із зворотнім поширенням помилки "
                 "для розпізнавання цифр від 0 до 9.",
            wraplength=600,
            justify="left",
            font=("Arial", 11)
        ).pack(padx=20, pady=(20, 10))
        self.status_label = tk.Label(
            self,
            text="Модель не налаштована.",
            anchor="nw"
        )
        self.status_label.pack(fill="x", padx=20)
        frame = tk.Frame(self)
        frame.pack(fill="x", padx=20, pady=20)
        self.canvas = Canvas(frame)
        self.canvas.pack(side="left")
        items_frame = tk.Frame(frame)
        items_frame.pack(side="left", fill="both", expand=True, padx=20)
        tk.Label(
            items_frame,
            text="Результати розпізнавання:",
            font=("Arial", 11),
            anchor="w"
        ).pack(fill="x")
        self.result_labels = []
        for i in range(10):
            lbl = tk.Label(items_frame, text=f"{i}: 0.0%", font=("Arial", 11), anchor="w")
            lbl.pack(fill="x", pady=2)
            self.result_labels.append(lbl)
        btn_frame = tk.Frame(items_frame)
        btn_frame.pack(side="bottom", fill="x")
        self.btn_predict = tk.Button(btn_frame, command=self._on_predict, text="Розпізнати")
        self.btn_predict.pack(side="right", padx=(40, 10))
        self.btn_clear = tk.Button(btn_frame, command=self._on_clear, text="Очистити")
        self.btn_clear.pack(side="right")

    def _set_ui_state(self, enabled):
        self.is_locked = not enabled
        state = tk.NORMAL if enabled else tk.DISABLED
        self.canvas.set_ui_state(enabled)
        self.btn_predict.config(state=state)
        self.btn_clear.config(state=state)

    def _start_training_thread(self):
        self.stop_event = threading.Event()
        self.status_label.config(text="Триває завантаження MNIST та навчання моделі...")
        self.progress_frame = ttk.Frame(self)
        self.progress_frame.pack(side="bottom", fill="x", padx=20, pady=20)
        self.progress = ttk.Progressbar(
            self.progress_frame,
            mode="indeterminate"
        )
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 20))
        self.progress.start()
        self.btn_cancel_training = tk.Button(
            self.progress_frame,
            text="Скасувати",
            command=self._cancel_training
        )
        self.btn_cancel_training.pack(side="right")
        thread = threading.Thread(target=self._training_thread, daemon=True)
        thread.start()

    def _cancel_training(self):
        self.stop_event.set()
        self.btn_cancel_training.config(state=tk.DISABLED, text="Скасування...")
        self.status_label.config(text="Зупинка процесу навчання...")

    def _training_thread(self):
        trained_nn = self.nn_manager.config(stop_event=self.stop_event)
        self.after(0, self._on_training_finished, trained_nn)

    def _on_training_finished(self, trained_nn):
        self.progress.stop()
        self.progress_frame.destroy()
        if trained_nn is None:
            self.destroy()
        else:
            self.nn = trained_nn
            self.epoch = trained_nn.epoch
            self.status_label.config(text=f"Модель навчена за {self.epoch} епох на датасеті MNIST.")
            self._set_ui_state(enabled=True)

    def _on_predict(self):
        if self.is_locked or self.nn is None:
            return
        processor = ImageProcessor()
        input_vector = processor.fetch_image(self.canvas.image)
        res, coef = self.nn.predict(input_vector)
        for i in range(10):
            if i == res:
                self.result_labels[i].config(
                    text=f"{i}: {coef[i]}%",
                    font=("Arial", 11, "bold")
                )
            else:
                self.result_labels[i].config(
                    text=f"{i}: {coef[i]}%",
                    font=("Arial", 11)
                )

    def _on_clear(self):
        if self.is_locked or self.nn is None:
            return
        self.canvas.clear()
        for i in range(10):
            self.result_labels[i].config(
                text=f"{i}: 0.0%",
                font=("Arial", 11)
            )

class TrainModelDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Файл не знайдно")
        self.geometry("400x120")
        self.resizable(width=False, height=False)
        self.transient(parent)
        self.grab_set()
        self.result = False
        tk.Label(
            self,
            text="Модель не налаштована. Почати навчання?",
            padx=20,
            pady=20,
            font=("Arial", 11)
        ).pack()
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=20)
        tk.Button(btn_frame, text="Навчати", command=self._on_confirm).pack(side="right", padx=20)
        tk.Button(btn_frame, text="Відмінити", command=self._on_cancel).pack(side="right")
        self._center_window(parent)
        parent.wait_window(self)

    def _on_confirm(self):
        self.result = True
        self.destroy()

    def _on_cancel(self):
        self.result = False
        self.destroy()

    def _center_window(self, parent):
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

class Canvas(tk.Canvas):
    def __init__(self, parent, size=280, brush_size=7):
        super().__init__(parent, width=size, height=size, bg="white", cursor="cross", relief=tk.SUNKEN, bd=2)
        self.size = size
        self.brush_size = brush_size
        self.enabled = True
        self.image = Image.new("RGB", (size, size), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        self.bind("<Button-1>", self._start_drawing)
        self.bind("<B1-Motion>", self._draw_line)
        self.bind("<ButtonRelease-1>", self._reset_coords)

    def set_ui_state(self, enabled):
        self.enabled = enabled
        self.config(
            cursor="cross" if enabled else "arrow",
            bg="white" if enabled else "#f0f0f0",
        )
        if not enabled:
            self.last_x, self.last_y = None, None

    def _start_drawing(self, event):
        if not self.enabled:
            return
        self.last_x, self.last_y = event.x, event.y
        self._draw_line(event)

    def _draw_line(self, event):
        if not self.enabled:
            return
        if self.last_x is not None or self.last_y is not None:
            self.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size, fill="black", capstyle=tk.ROUND, smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill="black", width=self.brush_size
            )
        self.last_x, self.last_y = event.x, event.y

    def _reset_coords(self, event):
        self.last_x, self.last_y = None, None

    def clear(self):
        self.delete("all")
        self.image = Image.new("RGB", (self.size, self.size), "white")
        self.draw = ImageDraw.Draw(self.image)

class ImageProcessor:
    def fetch_image(self, image):
        img = image.convert("L")
        img = ImageOps.invert(img)
        img = img.filter(ImageFilter.MaxFilter(9))
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        img.save("temp.png")
        img_arr = np.array(img) / 255.0
        return img_arr.reshape(784, 1)
