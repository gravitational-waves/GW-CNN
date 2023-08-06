import tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.animation as animation
import time
from pycbc_noise import create_noise
from pycbc.waveform import get_td_waveform


root = tkinter.Tk()
root.wm_title("Embedding in Tk")

# FIGURE OBJECTS ###############################################################

fig = Figure(figsize=(4, 3))
fig2 = Figure(figsize=(4, 3))
fig3 = Figure(figsize=(4, 3))
fig4 = Figure(figsize=(8, 6))

fig.suptitle("Accuracy")
fig2.suptitle("Strain")
fig3.suptitle("Loss")
fig4.suptitle("Visualization")

# LISTS ###############################################################################

accuracy = [0.5, 0.75, 0.833, 0.875, 0.899, 0.916, 0.928, 0.937, 0.944, 0.949, 0.954, 0.958, 0.961, 0.964, 0.966, 0.968,
            0.970, 0.972, 0.973, 0.975, 0.976, 0.977, 0.978, 0.979, 0.980, 0.980, 1.0]
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


def create_templates(m1, m2, noise_multiplier=1000):
    hp, hc = get_td_waveform(approximant="SEOBNRv4_opt",
                             mass1=m1,
                             mass2=m2,
                             delta_t=1.0 / 8192,
                             f_lower=20)

    noise, noise_psd = create_noise()
    noise *= noise_multiplier

    mid = int(len(noise) / 2)
    for i in range(len(hp)):
        noise[mid - i] += hp[len(hp) - i - 1]
    return hp.numpy(), noise.numpy()


def signal():
    m1 = e1.get()
    m2 = e2.get()
    hp, noise = create_templates(float(m1), float(m2))

    sine = [np.sin(i) for i in x]

    # fig2.add_subplot(1, 1, 1).plot(wave)
    # canvas1 = FigureCanvasTkAgg(fig2, master=bottomframe)
    # canvas1.flush_events()
    # canvas1.draw()
    # canvas1.get_tk_widget().pack(side=tkinter.LEFT)

    fig.add_subplot(1, 1, 1).plot(accuracy)
    canvas2 = FigureCanvasTkAgg(fig, master=middleframe)
    canvas2.flush_events()
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tkinter.RIGHT)

    fig3.add_subplot(1, 1, 1).plot(loss)
    canvas4 = FigureCanvasTkAgg(fig3, master=middleframe)
    canvas4.flush_events()    # s = e1.get()

    # fig2.add_subplot(1, 1, 1).plot(wave)
    # canvas1 = FigureCanvasTkAgg(fig2, master=bottomframe)
    # canvas1.flush_events()
    # canvas1.draw()
    #
    canvas4.draw()
    canvas4.get_tk_widget().pack(side=tkinter.LEFT)

    canvas3 = FigureCanvasTkAgg(fig4, master=bottomframe)
    canvas3.flush_events()
    canvas3.get_tk_widget().pack()
    xc, ny, y, slider = [], [], [], []
    ax = fig4.add_subplot(212)
    bx = fig4.add_subplot(212)    # s = e1.get()
    full_n = fig4.add_subplot(211)
    full_w = fig4.add_subplot(211)
    slide = fig4.add_subplot(211)

    full_n.plot([i for i in range(len(hp))], noise[:len(hp)], color='r')
    full_w.plot([i for i in range(len(hp))], hp, color="b")

    for i in range(len(hp)):
        xc.append(i)
        ny.append(noise[i])
        y.append(hp[i])

        ax.plot(xc, ny, "r")
        bx.plot(xc, y, "b")
        slide.plot([xc[i], xc[i] + 1], [-2 * pow(10, -18), -2 * pow(10, -18)], color='y')

        fig4.canvas.draw()

        ax.set_xlim(left=max(0, i - 50), right=i + 50)
        bx.set_xlim(left=max(0, i - 50), right=i + 50)

        time.sleep(1e-15)


# def accurate():
#     fig.add_subplot(1, 1, 1).plot(accuracy)
#     canvas2 = FigureCanvasTkAgg(fig, master=middleframe)
#     canvas2.flush_events()
#     canvas2.draw()
#     canvas2.get_tk_widget().pack(side=tkinter.TOP)
#
#
# def lose():
#     fig3.add_subplot(1, 1, 1).plot(loss)
#     canvas4 = FigureCanvasTkAgg(fig3, master=middleframe)
#     canvas4.flush_events()
#     canvas4.draw()
#     canvas4.get_tk_widget().pack(side=tkinter.TOP)


################################################################################


# def buttons():
#     accuracies = tkinter.Button(subtopframe2, text="Accuracy", command=accurate)
#     accuracies.pack(side=tkinter.LEFT)
#
#     losses = tkinter.Button(subtopframe2, text="Loss", command=lose)
#     losses.pack(side=tkinter.LEFT)
#
#     animate = tkinter.Button(subtopframe2, text="Animation", command=signal)
#     animate.pack(side=tkinter.LEFT)

#################################################################################


l1 = tkinter.Label(subtopframe1, text="Mass 1 :- ", font=30)
l1.pack(side=tkinter.LEFT)

e1 = tkinter.Entry(subtopframe1)
e1.pack(side=tkinter.LEFT)

l2 = tkinter.Label(subtopframe1, text="Mass 2 :- ", font=30)
l2.pack(side=tkinter.LEFT)

e2 = tkinter.Entry(subtopframe1)
e2.pack(side=tkinter.LEFT)

b1 = tkinter.Button(subtopframe1, text="Submit", command=signal, font=30, padx=20, anchor=tkinter.W)
b1.pack(side=tkinter.LEFT)

################################################################################

tkinter.mainloop()
