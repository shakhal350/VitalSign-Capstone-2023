import tkinter as tk
from tkinter import ttk
from tkinter import CENTER
import time
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotting
import pandas as pd

def SplashScreen():
    splash_root = tk.Tk()
    splash_root.geometry("300x300")

    splash_background_image = tk.PhotoImage(file="background_splash.png")
    splash_label = ttk.Label(splash_root, text="Welcome", font=56, image=splash_background_image)
    splash_label.place(relx=0.5, rely=0.5, anchor=CENTER)
    splash_label.pack()
    return splash_root

class VitalSignsGUI:
    def __init__(self, root, splash_root):
        self.root = root
        self.root.title("Vital Signs Monitor")
        self.splash_root = splash_root
        self.userAge = tk.StringVar()
        self.userGender = tk.StringVar()
        self.userWeight = tk.StringVar()
        self.userHeight = tk.StringVar()
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.settings_frame = ttk.Frame(self.root, padding="10")
        self.startup = True

    def main(self):

        if self.startup == True:
            self.splash_root.destroy()
            self.startup = False

        self.settings_frame.pack_forget()
        self.main_frame.pack(side="top", fill="both", expand=True)

        # Heart rate section
        heart_rate_label = ttk.Label(self.main_frame, text="Heart Rate", font=("Arial", 20))
        heart_rate_value = ttk.Label(self.main_frame, text="80 bpm", font=("Arial", 24), foreground="red")
        
        # Retrieve the plot for heart rate from plotting module
        fig1, ax1, ax2, line1, line2 = plotting.setup_plots(1)
        heart_rate_plot = FigureCanvasTkAgg(fig1, master=self.main_frame)  # Embedding the plot in the Tkinter window

        heart_rate_label.grid(row=0, column=0, sticky="w")
        heart_rate_value.grid(row=1, column=0, sticky="w")
        heart_rate_plot.get_tk_widget().grid(row=2, column=0, sticky="ew")

        # Respiratory rate section
        respiratory_rate_label = ttk.Label(self.main_frame, text="Respiratory Rate", font=("Arial", 20))
        respiratory_rate_value = ttk.Label(self.main_frame, text="18 breaths/min", font=("Arial", 24), foreground="blue")
        
        # Retrieve the plot for respiratory rate from plotting module
        fig2, ax3, ax4, line3, line4 = plotting.setup_plots(2)
        respiratory_rate_plot = FigureCanvasTkAgg(fig2, master=self.main_frame)  # Embedding the plot in the Tkinter window

        respiratory_rate_label.grid(row=0, column=1, sticky="w")
        respiratory_rate_value.grid(row=1, column=1, sticky="w")
        respiratory_rate_plot.get_tk_widget().grid(row=2, column=1, sticky="ew")

        # View physiological history button
        view_history_button = ttk.Button(self.main_frame, text="View Physiological History")
        view_history_button.grid(row=3, column=0, columnspan=2)
        # Patient Info button
        view_settings_button = ttk.Button(self.main_frame, text="View Patient Information", command=self.settingsPage)
        view_settings_button.grid(row=4, column=0, columnspan=2)

        # You might want to run the animation as part of the GUI initialization
        # self.run_animation(data, samples_per_frame, fps, window_size, update_interval)
    def settingsPage(self):

        self.settings_frame.pack(expand=True)
        self.main_frame.pack_forget()

        age_label = ttk.Label(self.settings_frame, text="Age", font=("Arial", 20))
        sex_label = ttk.Label(self.settings_frame, text="Sex", font=("Arial", 20))
        weight_label = ttk.Label(self.settings_frame, text="Weight", font=("Arial", 20))
        height_label = ttk.Label(self.settings_frame, text="Height", font=("Arial", 20))

        options = ["", "Undisclosed", "Male", "Female"]

        age_entry = ttk.Entry(self.settings_frame, textvariable=self.userAge, font=("Arial", 14), state="disabled")
        sex_entry = ttk.OptionMenu(self.settings_frame, self.userGender, *options)
        #sex_entry = ttk.Entry(self.settings_frame, textvariable=self.userGender, font=("Arial", 14))
        weight_entry = ttk.Entry(self.settings_frame, textvariable=self.userWeight, font=("Arial", 14), state="disabled")
        height_entry = ttk.Entry(self.settings_frame, textvariable=self.userHeight, font=("Arial", 14), state="disabled")

        age_label.grid(row=0, column=0, sticky="w")
        sex_label.grid(row=1, column=0, sticky="w")
        weight_label.grid(row=2, column=0, sticky="w")
        height_label.grid(row=3, column=0, sticky="w")

        age_entry.grid(row=0, column=1, sticky="w")
        sex_entry.grid(row=1, column=1, sticky="w")
        weight_entry.grid(row=2, column=1, sticky="w")
        height_entry.grid(row=3, column=1, sticky="w")

        go_back_button = ttk.Button(self.settings_frame, text="Go Back", command=self.main)
        go_back_button.grid(row=1, column=2, columnspan=2)
        save_button = ttk.Button(self.settings_frame, text="Save", command=lambda: self.save(age_entry,sex_entry,weight_entry,height_entry))
        save_button.grid(row=2, column=2, columnspan=2)
        edit_button = ttk.Button(self.settings_frame, text="Edit",command=lambda: self.edit(age_entry, sex_entry, weight_entry, height_entry))
        edit_button.grid(row=3, column=2, columnspan=2)

    # You can create a method to run the animation if you want to start it with a button click, for example


    def save(self, age_entry, sex_entry, weight_entry, height_entry):

        self.userAge.set(age_entry.get())
        self.userWeight.set(weight_entry.get())
        self.userHeight.set(height_entry.get())

        age_entry.config(state="disabled")
        #sex_entry.config(state=DISABLED)
        weight_entry.config(state="disabled")
        height_entry.config(state="disabled")


    def edit(self, age_entry, sex_entry, weight_entry, height_entry):
        age_entry.config(state="enabled")
        weight_entry.config(state="enabled")
        height_entry.config(state="enabled")

    def run_animation(self, data, samples_per_frame, fps, window_size, update_interval):
        plotting.create_animation(data, samples_per_frame, fps, window_size, update_interval)

    def setAge(self, age):
        self.userAge = age
    def setGender(self, gender):
        self.userGender = gender
    def setWeight(self, weight):
        self.userWeight = weight
    def setHeight(self, height):
        self.userHeight = height

    def getAge(self):
        return self.userAge
    def getGender(self):
        return self.userGender
    def getWeight(self):
        return self.userWeight
    def getHeight(self):
        return self.userHeight



if __name__ == "__main__":
    splash = SplashScreen()
    root = tk.Tk()
    app = VitalSignsGUI(root, splash)
    splash.after(5000, app.main)
    root.mainloop()
