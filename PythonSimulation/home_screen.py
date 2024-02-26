import csv
import tkinter as tk
from tkinter import CENTER
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
import plotting


def SplashScreen():
    splash_root = tk.Tk()
    splash_root.geometry("300x300")

    splash_label = ttk.Label(splash_root, text="Welcome", font=56)
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

        self.canvas = None
        self.animation1 = None
        self.animation2 = None

    def home_screen(self):
        if self.startup == True:
            self.splash_root.destroy()
            self.startup = False

        self.settings_frame.pack_forget()
        self.main_frame.pack(side="top", fill="both", expand=True)

        heart_rate_label = ttk.Label(
            self.main_frame, text="Heart Rate", font=("Arial", 18))
        heart_rate_value = ttk.Label(
            self.main_frame, text="80 bpm", font=("Arial", 24), foreground="red")
        heart_rate_label.grid(row=0, column=0, sticky="w")
        heart_rate_value.grid(row=1, column=0, sticky="w")

        respiratory_rate_label = ttk.Label(
            self.main_frame, text="Respiratory Rate", font=("Arial", 18))
        respiratory_rate_value = ttk.Label(
            self.main_frame, text="18 breaths/min", font=("Arial", 24), foreground="blue")
        respiratory_rate_label.grid(row=0, column=1, sticky="w")
        respiratory_rate_value.grid(row=1, column=1, sticky="w")

        fig1, ax1, ax2, line1, line2 = plotting.setup_plots(1)
        plot1 = FigureCanvasTkAgg(fig1, master=self.main_frame)
        plot1.get_tk_widget().grid(row=2, column=0, sticky="ew")

        fig2, ax3, ax4, line3, line4, ax5, ax6, line5, line6 = plotting.setup_plots(
            2)
        plot2 = FigureCanvasTkAgg(fig2, master=self.main_frame)
        plot2.get_tk_widget().grid(row=2, column=1, sticky="ew")

        view_history_button = ttk.Button(
            self.main_frame, text="View Physiological History")
        view_history_button.grid(row=3, column=0, columnspan=2)

        view_settings_button = ttk.Button(
            self.main_frame, text="View Patient Information", command=self.settingsPage)
        view_settings_button.grid(row=4, column=0, columnspan=2)

        self.run_animation(fig1, ax1, ax2, line1, line2, fig2, ax3, ax4, line3,
                           line4, ax5, ax6, line5, line6, heart_rate_value, respiratory_rate_value)

    def settingsPage(self):
        self.settings_frame.pack(expand=True)
        self.main_frame.pack_forget()

        age_label = ttk.Label(self.settings_frame,
                              text="Age", font=("Arial", 20))
        sex_label = ttk.Label(self.settings_frame,
                              text="Sex", font=("Arial", 20))
        weight_label = ttk.Label(
            self.settings_frame, text="Weight", font=("Arial", 20))
        height_label = ttk.Label(
            self.settings_frame, text="Height", font=("Arial", 20))

        options = ["", "Undisclosed", "Male", "Female"]

        age_entry = ttk.Entry(self.settings_frame, textvariable=self.userAge, font=(
            "Arial", 14), state="disabled")
        sex_entry = ttk.OptionMenu(
            self.settings_frame, self.userGender, *options)
        weight_entry = ttk.Entry(self.settings_frame, textvariable=self.userWeight, font=(
            "Arial", 14), state="disabled")
        height_entry = ttk.Entry(self.settings_frame, textvariable=self.userHeight, font=(
            "Arial", 14), state="disabled")

        self.readValues(age_entry, sex_entry, weight_entry, height_entry)

        age_label.grid(row=0, column=0, sticky="w")
        sex_label.grid(row=1, column=0, sticky="w")
        weight_label.grid(row=2, column=0, sticky="w")
        height_label.grid(row=3, column=0, sticky="w")

        age_entry.grid(row=0, column=1, sticky="w")
        sex_entry.grid(row=1, column=1, sticky="w")
        weight_entry.grid(row=2, column=1, sticky="w")
        height_entry.grid(row=3, column=1, sticky="w")

        go_back_button = ttk.Button(
            self.settings_frame, text="Go Back", command=self.home_screen)
        go_back_button.grid(row=1, column=2, columnspan=2)
        save_button = ttk.Button(self.settings_frame, text="Save", command=lambda: self.save(
            age_entry, weight_entry, height_entry))
        save_button.grid(row=2, column=2, columnspan=2)
        edit_button = ttk.Button(self.settings_frame, text="Edit", command=lambda: self.edit(
            age_entry, weight_entry, height_entry))
        edit_button.grid(row=3, column=2, columnspan=2)

    def readValues(self, age_entry, sex_entry, weight_entry, height_entry):
        with open("settingsinfo.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            count = 0
            for row in reader:
                if count == 2:
                    self.userAge.set(row[0])
                count = count + 1

    def save(self, age_entry, weight_entry, height_entry):
        fields = ["Age", "Sex", "Weight", "Height"]
        values = [age_entry.get(), self.userGender.get(),
                  weight_entry.get(), height_entry.get()]
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

    def run_animation(self, fig1, ax1, ax2, line1, line2, fig2, ax3, ax4, line3, line4, ax5, ax6, line5, line6, label1, label2):
        filename = r"C:\Users\Shaya\OneDrive - Concordia University - Canada\UNIVERSITY\CAPSTONE\Our Datasets (DCA1000EVM)\CSVFiles(RawData)\DCA1000EVM_shayan_normal_breathing.csv"

        data_Re, data_Im, radar_parameters = load_and_process_data(filename)

        animation_update_interval = 1

        data_Re = SVD_Matrix(data_Re, radar_parameters)
        data_Im = SVD_Matrix(data_Im, radar_parameters)
        plotting.create_animation(ax1, ax2, ax3, ax4, ax5, ax6, data_Re, data_Im, fig1, fig2, label1, label2, line1,
                                  line2, line3, line4, line5, line6, radar_parameters, animation_update_interval, timeWindowMultiplier=5)

    def create_plot(self, data_Re, data_Im, radar_parameters, update_interval, timeWindowMultiplier=1):
        self.fig, self.animation1, self.animation2 = create_animation(
            data_Re, data_Im, radar_parameters, update_interval, timeWindowMultiplier)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        widget = self.canvas.get_tk_widget()
        widget.pack()

        self.canvas.draw()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)

        widget = self.canvas.get_tk_widget()
        widget.pack()


if __name__ == "__main__":
    splash = SplashScreen()
    root = tk.Tk()
    app = VitalSignsGUI(root, splash)

    splash.after(5000, app.home_screen)
    root.mainloop()
