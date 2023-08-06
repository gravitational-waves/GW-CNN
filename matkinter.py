import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# import pahila
# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.animation as animation
import time

root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# FIGURE OBJECTS ###############################################################

fig = Figure(figsize=(4, 3))
fig2 = Figure(figsize=(4, 3))
fig3 = Figure(figsize=(4, 3))
fig4 = Figure(figsize=(4, 3))

fig.suptitle("Accuracy")
fig2.suptitle("Strain")
fig3.suptitle("Loss")
fig4.suptitle("Animation")

# LISTS ###############################################################################

accuracy = [i**2 for i in range(24)]
# wave = pahila.get_wave()
wave = [1, 2, 5, 4, 3, 6, 7]
loss = [i ** -0.5 for i in range(1, 25)]

# FRAMES ###############################################################################

topframe = tkinter.Frame(root)
topframe.pack(side=tkinter.TOP, padx=30)
middleframe = tkinter.Frame(root)
middleframe.pack(fill=tkinter.BOTH, padx=30)
bottomframe = tkinter.Frame(root)
bottomframe.pack(side=tkinter.BOTTOM, fill=tkinter.X, padx=30)
subtopframe1 = tkinter.Frame(topframe)
subtopframe1.pack(side=tkinter.TOP)
subtopframe2 = tkinter.Frame(topframe)
subtopframe2.pack(side=tkinter.BOTTOM)


################################################################################

x = np.arange(0, 2*np.pi, 0.01)        # x-array


################################################################################


def signal():
    # s = e1.get()
    sine = [np.sin(i) for i in x]

    fig2.add_subplot(1, 1, 1).plot(wave)
    canvas1 = FigureCanvasTkAgg(fig2, master=middleframe)
    canvas1.flush_events()

    canvas1.draw()
    canvas1.get_tk_widget().pack()

    canvas3 = FigureCanvasTkAgg(fig4, master=bottomframe)
    canvas3.flush_events()
    canvas3.get_tk_widget().pack()
    xc, ny = [], []
    ax = fig4.add_subplot(111)

    for i in range(len(sine)):
        xc.append(i)
        ny.append(sine[i])
        ax.plot(xc, ny, "r")
        fig4.canvas.draw()
        time.sleep(0.000000000000000000000000000000000005)


def accurate():
    fig.add_subplot(1, 1, 1).plot(accuracy)
    canvas2 = FigureCanvasTkAgg(fig, master=middleframe)
    canvas2.flush_events()
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tkinter.TOP)


def lose():
    fig3.add_subplot(1, 1, 1).plot(loss)
    canvas4 = FigureCanvasTkAgg(fig3, master=middleframe)
    canvas4.flush_events()
    canvas4.draw()
    canvas4.get_tk_widget().pack(side=tkinter.TOP)


################################################################################


def buttons():
    accuracies = tkinter.Button(subtopframe2, text="Accuracy", command=accurate)
    accuracies.pack(side=tkinter.LEFT)

    losses = tkinter.Button(subtopframe2, text="Loss", command=lose)
    losses.pack(side=tkinter.LEFT)

    animate = tkinter.Button(subtopframe2, text="Animation", command=signal)
    animate.pack(side=tkinter.LEFT)

#################################################################################


l1 = tkinter.Label(subtopframe1, text="Filename :- ", font=30)
l1.pack(side=tkinter.LEFT)

e1 = tkinter.Entry(subtopframe1)
e1.pack(side=tkinter.LEFT)

b1 = tkinter.Button(subtopframe1, text="Submit", command=buttons, font=30, padx=20, anchor=tkinter.W)
b1.pack(side=tkinter.LEFT)

################################################################################

tkinter.mainloop()
