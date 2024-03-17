import tkinter as tk
from tkinter import ttk
from tkinter import CENTER
import time
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotting
import pandas as pd
import csv
from GradientFrame import GradientFrame

class LoadingScreenGif:
    def __init__(self, r, first):
        self.startup = first
        self.root = r
        self.canvas = GradientFrame(r, from_color="#FFF8ED", to_color="#6976DA", width=17000, height=700, borderwidth=0)

        self.img0 = Image.open("loadingGif/frame_00_delay-0.03s.gif").resize((72, 72))
        self.img1 = Image.open("loadingGif/frame_01_delay-0.03s.gif").resize((72, 72))
        self.img2 = Image.open("loadingGif/frame_02_delay-0.03s.gif").resize((72, 72))
        self.img3 = Image.open("loadingGif/frame_03_delay-0.03s.gif").resize((72, 72))
        self.img4 = Image.open("loadingGif/frame_04_delay-0.03s.gif").resize((72, 72))
        self.img5 = Image.open("loadingGif/frame_05_delay-0.03s.gif").resize((72, 72))
        self.img6 = Image.open("loadingGif/frame_06_delay-0.03s.gif").resize((72, 72))
        self.img7 = Image.open("loadingGif/frame_07_delay-0.03s.gif").resize((72, 72))
        self.img8 = Image.open("loadingGif/frame_08_delay-0.03s.gif").resize((72, 72))
        self.img9 = Image.open("loadingGif/frame_09_delay-0.03s.gif").resize((72, 72))
        self.img10 = Image.open("loadingGif/frame_10_delay-0.03s.gif").resize((72, 72))
        self.img11 = Image.open("loadingGif/frame_11_delay-0.03s.gif").resize((72, 72))
        self.img12 = Image.open("loadingGif/frame_12_delay-0.03s.gif").resize((72, 72))
        self.img13 = Image.open("loadingGif/frame_13_delay-0.03s.gif").resize((72, 72))
        self.img14 = Image.open("loadingGif/frame_14_delay-0.03s.gif").resize((72, 72))
        self.img15 = Image.open("loadingGif/frame_15_delay-0.03s.gif").resize((72, 72))
        self.img16 = Image.open("loadingGif/frame_16_delay-0.03s.gif").resize((72, 72))
        self.img17 = Image.open("loadingGif/frame_17_delay-0.03s.gif").resize((72, 72))
        self.img18 = Image.open("loadingGif/frame_18_delay-0.03s.gif").resize((72, 72))
        self.img19 = Image.open("loadingGif/frame_19_delay-0.03s.gif").resize((72, 72))
        self.img20 = Image.open("loadingGif/frame_20_delay-0.03s.gif").resize((72, 72))
        self.img21 = Image.open("loadingGif/frame_21_delay-0.03s.gif").resize((72, 72))
        self.img22 = Image.open("loadingGif/frame_22_delay-0.03s.gif").resize((72, 72))
        self.img23 = Image.open("loadingGif/frame_23_delay-0.03s.gif").resize((72, 72))
        self.img24 = Image.open("loadingGif/frame_24_delay-0.03s.gif").resize((72, 72))
        self.img25 = Image.open("loadingGif/frame_25_delay-0.03s.gif").resize((72, 72))
        self.img26 = Image.open("loadingGif/frame_26_delay-0.03s.gif").resize((72, 72))
        self.img27 = Image.open("loadingGif/frame_27_delay-0.03s.gif").resize((72, 72))
        self.img28 = Image.open("loadingGif/frame_28_delay-0.03s.gif").resize((72, 72))
        self.img29 = Image.open("loadingGif/frame_29_delay-0.03s.gif").resize((72, 72))

        self.gif0 = ImageTk.PhotoImage(self.img1)
        self.gif1 = ImageTk.PhotoImage(self.img2)
        self.gif2 = ImageTk.PhotoImage(self.img2)
        self.gif3 = ImageTk.PhotoImage(self.img3)
        self.gif4 = ImageTk.PhotoImage(self.img4)
        self.gif5 = ImageTk.PhotoImage(self.img5)
        self.gif6 = ImageTk.PhotoImage(self.img6)
        self.gif7 = ImageTk.PhotoImage(self.img7)
        self.gif8 = ImageTk.PhotoImage(self.img8)
        self.gif9 = ImageTk.PhotoImage(self.img9)
        self.gif10 = ImageTk.PhotoImage(self.img10)
        self.gif11 = ImageTk.PhotoImage(self.img11)
        self.gif12 = ImageTk.PhotoImage(self.img12)
        self.gif13 = ImageTk.PhotoImage(self.img13)
        self.gif14 = ImageTk.PhotoImage(self.img14)
        self.gif15 = ImageTk.PhotoImage(self.img15)
        self.gif16 = ImageTk.PhotoImage(self.img16)
        self.gif17 = ImageTk.PhotoImage(self.img17)
        self.gif18 = ImageTk.PhotoImage(self.img18)
        self.gif19 = ImageTk.PhotoImage(self.img19)
        self.gif20 = ImageTk.PhotoImage(self.img20)
        self.gif21 = ImageTk.PhotoImage(self.img21)
        self.gif22 = ImageTk.PhotoImage(self.img22)
        self.gif23 = ImageTk.PhotoImage(self.img23)
        self.gif24 = ImageTk.PhotoImage(self.img24)
        self.gif25 = ImageTk.PhotoImage(self.img25)
        self.gif26 = ImageTk.PhotoImage(self.img26)
        self.gif27 = ImageTk.PhotoImage(self.img27)
        self.gif28 = ImageTk.PhotoImage(self.img28)
        self.gif29 = ImageTk.PhotoImage(self.img29)

        self.gifs = [self.gif0, self.gif1, self.gif2, self.gif3, self.gif4, self.gif5, self.gif6, self.gif7, self.gif8,
                     self.gif9, self.gif10, self.gif11, self.gif12, self.gif13, self.gif14, self.gif15, self.gif16,
                     self.gif17, self.gif18, self.gif19,self.gif20, self.gif21, self.gif22, self.gif23, self.gif24,
                     self.gif25, self.gif26, self.gif27, self.gif28, self.gif29]

        self.logoImage = Image.open("icons8-proximity-sensor-80.png").resize((150,150))
        self.logoPic = ImageTk.PhotoImage(self.logoImage)


        self.text2 = self.canvas.create_text(610, 150, text="BIONEST", font=("Mokoto", 64), fill="#3A48B0")
        self.logo = self.canvas.create_image(610, 350, image=self.logoPic)
        if self.startup:
            self.final = self.canvas.create_image(610, 530, image=self.gif0)

    def pack(self):
        self.canvas.pack()

    def secondTime(self):
        self.startup = False

    def update(self, root, i):
        frame = self.gifs[i]
        if i == 29:
            i = 0
        else:
            i += 1
        self.canvas.itemconfig(self.final, image=frame)
        root.after(20, self.update, root, i)

    def select(self, root):
        self.canvas.delete(self.final)
        self.Vital = ttk.Button(root, text="Access Vital Signs")
        self.Settings = ttk.Button(root, text="Settings")

        #self.Vital.config(width=10, activebackground="#FFF8ED")
        #self.Settings.config(width=10, activebackground="#FFF8ED")

        self.VitalButton = self.canvas.create_window(610, 530,window=self.Vital)
        self.SettingsButton = self.canvas.create_window(610, 570, window=self.Settings)



#FOR TESTING

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1220x640")
    root.title("Vital Signs Monitor")
    frame = tk.Frame(root)
    frame.pack()
    giffy = LoadingScreenGif(frame)
    giffy.pack()
    root.after(0, giffy.update, root, 0)
    root.mainloop()
