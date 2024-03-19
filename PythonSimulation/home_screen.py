import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from SVD_processing import SVD_Matrix
from data_processing import load_and_process_data
from TestData import setup_plots
import csv
import LoadingScreenGif as LSG
from process_raw_data import readDCA1000


class VitalSignsGUI:
    def __init__(self, root):
        self.root = root
        self.userAge = tk.StringVar()
        self.userGender = tk.StringVar()
        self.userWeight = tk.StringVar()
        self.userHeight = tk.StringVar()
        self.settings_frame = ttk.Frame(self.root, padding="10")
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_dev_frame = ttk.Frame(self.root, padding="10")
        self.splashFrame = ttk.Frame(self.root)
        self.startup = True
        self.loadingGif = LSG.LoadingScreenGif(self.splashFrame, self.startup)

    def splashScreen(self):
        self.root.geometry("1220x640")
        self.settings_frame.pack_forget()
        self.main_frame.pack_forget()
        self.main_dev_frame.pack_forget()
        self.splashFrame.pack()

        self.loadingGif.pack()
        if self.startup:
            self.splashFrame.after(0,self.loadingGif.update,self.splashFrame, 0)
            self.splashFrame.after(5000, self.loadingScreenUpdate)
        else:
            self.splashFrame.after(0,self.loadingScreenUpdate)

    def loadingScreenUpdate(self):

        if self.startup:
            self.splashFrame.after(1000)
            self.loadingGif.canvas.delete(self.loadingGif.final)
        self.Record = tk.Button(root, text="Record Vitals", background="#FFF8ED")
        self.Vital = tk.Button(root, text="Access Vital Signs", command=self.view_data, background="#FFF8ED")
        self.Settings = tk.Button(root, text="Settings", command=self.settingsPage, background="#FFF8ED")
        self.Quit = tk.Button(root, text="Quit", command=self.root.destroy, background="#FFF8ED")

        self.Record.config(width=15,padx=0,pady=0, bg="#7DC7F1")
        self.Settings.config(width=15,padx=0,pady=0, bg="#7DC7F1")
        self.Vital.config(width=15, padx=0, pady=0, bg="#7DC7F1")
        self.Quit.config(width=15, padx=0, pady=0, bg="#7DC7F1")

        self.RecordButton = self.loadingGif.canvas.create_window(610, 480, window=self.Record)
        self.VitalButton = self.loadingGif.canvas.create_window(610, 510, window=self.Vital)
        self.SettingsButton = self.loadingGif.canvas.create_window(610, 540, window=self.Settings)
        self.QuitButton = self.loadingGif.canvas.create_window(610, 570, window=self.Quit)

        self.loadingGif.secondTime()


    # def record_button(self):
    #     # Example call, replace paths with your desired file paths or use dialog to select files
    #     fileName = r"C:\ti\mmwave_studio_02_01_01_00\mmWaveStudio\PostProc\adc_data.bin"
    #     csvFileName = r"C:\Users\Shaya\PycharmProjects\VitalSign-Capstone-2023\DATASET\DCA1000EVM_Shayan_normal_upclose_60sec.csv"
    #     readDCA1000(fileName, csvFileName)

    def view_data(self):

        self.root.geometry("1620x670")

        if self.startup == True:
            self.startup = False

        self.splashFrame.pack_forget()
        self.settings_frame.pack_forget()
        self.main_dev_frame.pack_forget()
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

        # fig1, ax1, ax2, line1, line2 = plotting.setup_plots(1)
        fig, ax1, ax2, line1, line2 = setup_plots(1, r"C:\Users\Shaya\Downloads\DCA1000EVM_grace2_shallow_BR.csv")
        plot1 = FigureCanvasTkAgg(fig, master=self.main_frame) # Embedding the plot in the Tkinter window
        plot1.get_tk_widget().grid(row=2, column=0, sticky="ew")

        # fig2, ax3, ax4, line3, line4, ax5, ax6, line5, line6 = plotting.setup_plots(
        #     2)
        # plot2 = FigureCanvasTkAgg(fig, master=self.main_frame) # Embedding the plot in the Tkinter window
        # plot2.get_tk_widget().grid(row=2, column=1, sticky="ew")

        # View physiological history button
        view_history_button = ttk.Button(
            self.main_frame, text="Developer Mode", command=self.view_data_dev)
        view_history_button.config(width=20)
        view_history_button.grid(row=3, column=0, columnspan=2)

        # Patient Info button
        view_settings_button = ttk.Button(
            self.main_frame, text="Settings", command=self.settingsPage)
        view_settings_button.config(width=20)
        view_settings_button.grid(row=4, column=0, columnspan=2)

        back_home_button = ttk.Button(
            self.main_frame, text="Go Back Home", command=self.splashScreen)
        back_home_button.config(width=20)
        back_home_button.grid(row=5, column=0, columnspan=2)

        # You might want to run the animation as part of the GUI initialization

        #self.run_animation(fig1, ax1, ax2, line1, line2, fig2, ax3, ax4, line3,
                           #line4, ax5, ax6, line5, line6, heart_rate_value, respiratory_rate_value)

    def view_data_dev(self):

        self.root.geometry("1620x670")

        if self.startup == True:
            self.startup = False

        self.splashFrame.pack_forget()
        self.settings_frame.pack_forget()
        self.main_frame.pack_forget()
        self.main_dev_frame.pack(side="top", fill="both", expand=True)

        heart_rate_label = ttk.Label(
            self.main_dev_frame, text="Heart Rate", font=("Arial", 18))
        heart_rate_value = ttk.Label(
            self.main_dev_frame, text="80 bpm", font=("Arial", 24), foreground="red")
        heart_rate_label.grid(row=0, column=0, sticky="w")
        heart_rate_value.grid(row=1, column=0, sticky="w")

        respiratory_rate_label = ttk.Label(
            self.main_dev_frame, text="Respiratory Rate", font=("Arial", 18))
        respiratory_rate_value = ttk.Label(
            self.main_dev_frame, text="18 breaths/min", font=("Arial", 24), foreground="blue")
        respiratory_rate_label.grid(row=0, column=1, sticky="w")
        respiratory_rate_value.grid(row=1, column=1, sticky="w")

        # fig1, ax1, ax2, line1, line2 = plotting.setup_plots(1)
        # plot1 = FigureCanvasTkAgg(fig1, master=self.main_frame) # Embedding the plot in the Tkinter window
        # plot1.get_tk_widget().grid(row=2, column=0, sticky="ew")
        #
        # fig2, ax3, ax4, line3, line4, ax5, ax6, line5, line6 = plotting.setup_plots(
        #     2)
        # plot2 = FigureCanvasTkAgg(fig2, master=self.main_frame) # Embedding the plot in the Tkinter window
        # plot2.get_tk_widget().grid(row=2, column=1, sticky="ew")

        # View physiological history button
        view_history_button = ttk.Button(
            self.main_dev_frame, text="Default Mode", command=self.view_data)
        view_history_button.config(width=20)
        view_history_button.grid(row=3, column=0, columnspan=2)

        # Patient Info button
        view_settings_button = ttk.Button(
            self.main_dev_frame, text="Settings", command=self.settingsPage)
        view_settings_button.config(width=20)
        view_settings_button.grid(row=4, column=0, columnspan=2)

        back_home_button = ttk.Button(
            self.main_dev_frame, text="Go Back Home", command=self.splashScreen)
        back_home_button.config(width=20)
        back_home_button.grid(row=5, column=0, columnspan=2)

        # You might want to run the animation as part of the GUI initialization

        #self.run_animation(fig1, ax1, ax2, line1, line2, fig2, ax3, ax4, line3,
                           #line4, ax5, ax6, line5, line6, heart_rate_value, respiratory_rate_value)

    def settingsPage(self):

        self.root.geometry("540x320")

        if self.startup == True:
            self.startup = False

        self.splashFrame.pack_forget()
        self.main_frame.pack_forget()
        self.settings_frame.pack(expand=True)

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
            self.settings_frame, text="Back to Vitals", command=self.view_data)
        go_back_button.grid(row=0, column=2, columnspan=2)

        save_button = ttk.Button(self.settings_frame, text="Save", command=lambda: self.save(
            age_entry, weight_entry, height_entry))

        save_button.grid(row=1, column=2, columnspan=2)

        edit_button = ttk.Button(self.settings_frame, text="Edit", command=lambda: self.edit(
            age_entry, weight_entry, height_entry))
        edit_button.grid(row=2, column=2, columnspan=2)

        back_home_button = ttk.Button(
            self.settings_frame, text="Back Home", command=self.splashScreen)
        back_home_button.grid(row=3, column=2, columnspan=2)


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
        # filename = r"DCA1000EVM_shayan_normal_breathing.csv"
        data_Re, data_Im, radar_parameters = load_and_process_data(filename)

        animation_update_interval = 1

        data_Re = SVD_Matrix(data_Re, radar_parameters, 2)
        data_Im = SVD_Matrix(data_Im, radar_parameters, 2)

        # plotting.create_animation(fig1, ax1, ax2, line1, line2, fig2, ax3, ax4, line3, line4, ax5, ax6, line5, line6,
        #                           label1, label2, data_Re, data_Im, radar_parameters, animation_update_interval,
        #                           timeWindowMultiplier=5)

    def create_plot(self, data_Re, data_Im, radar_parameters, update_interval, timeWindowMultiplier=1):
        # self.fig, self.animation1, self.animation2 = create_animation(
        #     data_Re, data_Im, radar_parameters, update_interval, timeWindowMultiplier)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        widget = self.canvas.get_tk_widget()
        widget.pack()

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
    root.geometry("1220x640")
    root.title("Vital Signs Monitor")
    app = VitalSignsGUI(root)
    root.after(0,app.splashScreen)
    #root.after(6000, app.view_data)
    root.mainloop()
