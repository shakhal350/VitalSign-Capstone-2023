import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotting

class VitalSignsGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Vital Signs Monitor")

        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(side="top", fill="both", expand=True)

        # Heart rate section
        heart_rate_label = ttk.Label(main_frame, text="Heart Rate", font=("Arial", 14))
        heart_rate_value = ttk.Label(main_frame, text="80 bpm", font=("Arial", 18), foreground="red")
        
        # Retrieve the plot for heart rate from plotting module
        fig1, ax1, ax2, line1, line2 = plotting.setup_plots(1)
        heart_rate_plot = FigureCanvasTkAgg(fig1, master=main_frame)  # Embedding the plot in the Tkinter window

        heart_rate_label.grid(row=0, column=0, sticky="w")
        heart_rate_value.grid(row=1, column=0, sticky="w")
        heart_rate_plot.get_tk_widget().grid(row=2, column=0, sticky="ew")

        # Respiratory rate section
        respiratory_rate_label = ttk.Label(main_frame, text="Respiratory Rate", font=("Arial", 14))
        respiratory_rate_value = ttk.Label(main_frame, text="18 breaths/min", font=("Arial", 18), foreground="blue")
        
        # Retrieve the plot for respiratory rate from plotting module
        fig2, ax3, ax4, line3, line4 = plotting.setup_plots(2)
        respiratory_rate_plot = FigureCanvasTkAgg(fig2, master=main_frame)  # Embedding the plot in the Tkinter window

        respiratory_rate_label.grid(row=0, column=1, sticky="w")
        respiratory_rate_value.grid(row=1, column=1, sticky="w")
        respiratory_rate_plot.get_tk_widget().grid(row=2, column=1, sticky="ew")

        # View physiological history button
        view_history_button = ttk.Button(main_frame, text="View Physiological History")
        view_history_button.grid(row=3, column=0, columnspan=2)

        # You might want to run the animation as part of the GUI initialization
        # self.run_animation(data, samples_per_frame, fps, window_size, update_interval)

    # You can create a method to run the animation if you want to start it with a button click, for example
    def run_animation(self, data, samples_per_frame, fps, window_size, update_interval):
        plotting.create_animation(data, samples_per_frame, fps, window_size, update_interval)

if __name__ == "__main__":
    root = tk.Tk()
    app = VitalSignsGUI(root)
    root.mainloop()
