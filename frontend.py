import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from AppOpener import open as app_opener
import csv
import cv2
import random
import threading
import time
from backend import *

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("App Launcher")
        self.frame = None

        self.backend_thread = None
        self.random_number_interval = 10 

        self.app_data = []

        self.tabs = ttk.Notebook(root)
        self.home_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.home_tab, text="Home")
        self.create_home_tab()
        self.settings_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.settings_tab, text="Settings")
        self.create_settings_tab()

        self.tabs.pack(expand=1, fill="both")

        self.curr_prediction = -1
        self.confident_prediction = -1
        self.prediction_counter = 0
        self.prediction_threshold = 15


    def create_home_tab(self):
        self.label = tk.Label(self.home_tab)
        self.label.pack()

        # self.cap = cv2.VideoCapture(0)
        # self.show_frame()

        self.toggle_button = tk.Button(self.home_tab, text="Start", command=self.toggle_stream)
        self.toggle_button.pack()

    def show_frame(self):
        if self.cap.isOpened():
            _, self.frame = self.cap.read()
            # _,modified = preprocess(self.frame)
            new_prediction = predict(self.frame)
            print(new_prediction)
            if self.curr_prediction != new_prediction:
                self.curr_prediction = new_prediction
                self.confident_prediction = -1
                self.prediction_counter = 0
            else:
                if self.prediction_counter == self.prediction_threshold:
                    self.confident_prediction = self.curr_prediction
                else:
                    self.prediction_counter += 1
            cv2image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
        self.label.after(10, self.show_frame)

    def toggle_stream(self):
        if self.toggle_button["text"] == "Stop":
            self.toggle_button["text"] = "Start"
            self.cap.release()
            # self.backend_thread.close()
        else:
            self.toggle_button["text"] = "Stop"
            self.cap = cv2.VideoCapture(0)
            self.show_frame()
            self.backend_thread = threading.Thread(target=self.get_label, daemon=True)
            self.backend_thread.start()

    # def generate_random_number(self):
    #     time.sleep(self.random_number_interval)
    #     random_number = random.randint(1, 10)
    #     print(f"Received random number: {random_number}")
    #     self.open_app_based_on_number(random_number)

    def get_label(self):
        while self.confident_prediction == -1:
            time.sleep(0.2)
        print(f"Received Label: {self.confident_prediction}")
        self.open_app_based_on_number(self.confident_prediction)

    def create_settings_tab(self):
        self.table = ttk.Treeview(self.settings_tab, columns=("App ID", "App Name"), show="headings")
        self.table.heading("App ID", text="App ID")
        self.table.heading("App Name", text="App Name")
        self.table.column("App ID", width=100)
        self.table.column("App Name", width=500)
        self.load_data()
        self.table.pack()

        app_name_label = tk.Label(self.settings_tab, text="App Name")
        self.app_name_entry = tk.Entry(self.settings_tab)
        app_name_label.pack()
        self.app_name_entry.pack()

        update_button = tk.Button(self.settings_tab, text="Update", command=lambda: self.update_data(self.app_name_entry.get()))
        update_button.pack()

        self.table.bind("<<TreeviewSelect>>", lambda event: self.load_selected_data())

    def load_data(self):
        try:
            with open("app_data.csv", newline="") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    self.app_data.append((row[0], row[1]))
                    self.table.insert("", "end", values=(row[0], row[1]))
        except FileNotFoundError:
            pass

    def load_selected_data(self):
        item = self.table.selection()
        if item:
            values = self.table.item(item, "values")
            self.app_name_entry.delete(0, tk.END)
            self.app_name_entry.insert(0, values[1])

    def update_data(self, app_name):
        selected_item = self.table.selection()
        
        values = self.table.item(selected_item, "values")
        if selected_item:
            self.table.item(selected_item, values=(values[0], app_name))
        self.app_name_entry.delete(0, tk.END)

        for index, data in enumerate(self.app_data):
            if data[0] == values[0]:
                self.app_data[index] = (values[0], app_name)
        self.update_csv_file()
    
    def update_csv_file(self):
        with open("app_data.csv", mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["App ID", "App Name"])
            for data in self.app_data:
                writer.writerow(data)

    def open_app_based_on_number(self, number):
        for app_data in self.app_data:
            app_ID, app_name = app_data
            if int(app_ID[-1]) == number:
                self.cap.release()
                self.toggle_button["text"] = "Start"
                
                app_opener(app_name, match_closest=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
