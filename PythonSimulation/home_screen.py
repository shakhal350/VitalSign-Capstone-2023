import tkinter as tk
from tkinter import font
from tkinter import ttk
from tkinter import CENTER
import time
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotting
import pandas as pd
import csv
import LoadingScreenGif as LSG
from GradientFrame import GradientFrame

class VitalSignsGUI:
    def __init__(self, root):
        self.root = root
        self.userAge = tk.StringVar()
        self.userGender = tk.StringVar()
        self.userWeight = tk.StringVar()
        self.userHeight = tk.StringVar()
        self.settings_frame = ttk.Frame(self.root, padding="10")
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.splashFrame = ttk.Frame(self.root)
        self.bg = ImageTk.PhotoImage(Image.open("back2.png"))
        self.gif = []
        for i in range(30):
            self.gif.append(ImageTk.PhotoImage(file="load.gif", height=50, width=50, format="gif -index %i" %(i)))
        self.frame = self.gif[0]
        self.startup = True
        self.loadingGif = LSG.LoadingScreenGif(self.splashFrame)

    def splashScreen(self):

        self.settings_frame.pack_forget()
        self.main_frame.pack_forget()
        self.splashFrame.pack()

        self.loadingGif.pack()
        self.splashFrame.after(0,self.loadingGif.update,self.splashFrame, 0)

    def main(self):

        if self.startup == True:
            self.splashFrame.destroy()
            self.startup = False

        self.settings_frame.pack_forget()
        self.main_frame.pack(side="top", fill="both", expand=True)

        heart_rate_label = ttk.Label(self.main_frame, text="Heart Rate", font=("Arial", 18))
        heart_rate_value = ttk.Label(self.main_frame, text="80 bpm", font=("Arial", 24), foreground="red")
        heart_rate_label.grid(row=0, column=0, sticky="w")
        heart_rate_value.grid(row=1, column=0, sticky="w")

        respiratory_rate_label = ttk.Label(self.main_frame, text="Respiratory Rate", font=("Arial", 18))
        respiratory_rate_value = ttk.Label(self.main_frame, text="18 breaths/min", font=("Arial", 24), foreground="blue")
        respiratory_rate_label.grid(row=0, column=1, sticky="w")
        respiratory_rate_value.grid(row=1, column=1, sticky="w")


        fig1, ax1, ax2, line1, line2 = plotting.setup_plots(1)
        plot1 = FigureCanvasTkAgg(fig1, master=self.main_frame)  # Embedding the plot in the Tkinter window
        plot1.get_tk_widget().grid(row=2, column=0, sticky="ew")

        fig2, ax3, ax4, line3, line4 = plotting.setup_plots(2)
        plot2 = FigureCanvasTkAgg(fig2, master=self.main_frame)  # Embedding the plot in the Tkinter window
        plot2.get_tk_widget().grid(row=3, column=0, sticky="ew")

        fig3, ax5, ax6, line5, line6 = plotting.setup_plots(3)
        plot3 = FigureCanvasTkAgg(fig3, master=self.main_frame)  # Embedding the plot in the Tkinter window
        plot3.get_tk_widget().grid(row=2, column=1, sticky="ew")

        fig4, ax7, ax8, line7, line8 = plotting.setup_plots(4)
        plot4 = FigureCanvasTkAgg(fig4, master=self.main_frame)  # Embedding the plot in the Tkinter window
        plot4.get_tk_widget().grid(row=3, column=1, sticky="ew")

        # View physiological history button
        view_history_button = ttk.Button(self.main_frame, text="View Physiological History")
        view_history_button.grid(row=4, column=0, columnspan=2)
        # Patient Info button
        view_settings_button = ttk.Button(self.main_frame, text="View Patient Information", command=self.settingsPage)
        view_settings_button.grid(row=5, column=0, columnspan=2)

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

        self.readValues(age_entry, sex_entry, weight_entry, height_entry)

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
        save_button = ttk.Button(self.settings_frame, text="Save", command=lambda: self.save(age_entry,weight_entry,height_entry))
        save_button.grid(row=2, column=2, columnspan=2)
        edit_button = ttk.Button(self.settings_frame, text="Edit", command=lambda: self.edit(age_entry, weight_entry, height_entry))
        edit_button.grid(row=3, column=2, columnspan=2)

    # You can create a method to run the animation if you want to start it with a button click, for example

    def readValues(self,age_entry, sex_entry, weight_entry, height_entry):
        with open("settingsinfo.csv","r") as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            for row in reader:
                if count == 2:
                    self.userAge.set(row[0])
                count = count + 1


    def save(self, age_entry, weight_entry, height_entry):

        fields = ["Age", "Sex", "Weight", "Height"]
        values = [age_entry.get(), self.userGender.get(), weight_entry.get(), height_entry.get()]
        with open("settingsinfo.csv", "w") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(fields)
            writer.writerow(values)

        age_entry.config(state="disabled")
        weight_entry.config(state="disabled")
        height_entry.config(state="disabled")


    def edit(self, age_entry, weight_entry, height_entry):
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

    def update(self,i,label):
        self.frame = self.gif[i]
        print(self.frame)
        if i == 29:
            i = 0
        else:
            i += 1
        label.config(image=self.frame)
        self.splashFrame.after(50, self.update, i, label)



if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1220x740")
    root.title("Vital Signs Monitor")
    app = VitalSignsGUI(root)
    root.after(0,app.splashScreen)
    root.after(6000, app.main)
    root.mainloop()
