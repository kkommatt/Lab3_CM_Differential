from tkinter import ttk, Tk, Entry, Label, Button, Frame
import tkinter as tk  # Add this line
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def euler_method(dydx, x0, y0, xs, k, h, ys):
    x_vals = [x0]
    y_vals = [y0]

    while x_vals[-1] < xs:
        x = x_vals[-1]
        y = y_vals[-1]
        y_prime = dydx(x, y, k, ys)  # Include ys parameter
        y_new = y + h * y_prime
        x_vals.append(x + h)
        y_vals.append(y_new)

    return x_vals, y_vals


def runge_kutta_method(dydx, x0, y0, xs, k, h, ys):
    x_vals = [x0]
    y_vals = [y0]

    while x_vals[-1] < xs:
        x = x_vals[-1]
        y = y_vals[-1]

        k1 = h * dydx(x, y, k, ys)  # Include ys parameter
        k2 = h * dydx(x + h / 2, y + k1 / 2, k, ys)
        k3 = h * dydx(x + h / 2, y + k2 / 2, k, ys)
        k4 = h * dydx(x + h, y + k3, k, ys)

        y_new = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        x_vals.append(x + h)
        y_vals.append(y_new)

    return x_vals, y_vals


def dydx(x, y, k, ys):
    return -k * (y - ys)

def compute_and_plot():
    k = float(k_entry.get())
    x0 = float(x0_entry.get())
    y0 = float(y0_entry.get())
    ys = float(ys_entry.get())
    a, b = map(float, interval_entry.get().strip('[]').split(';'))
    h = float(h_entry.get())

    euler_x_vals, euler_y_vals = euler_method(dydx, x0, y0, b, k, h, ys)
    rk_x_vals, rk_y_vals = runge_kutta_method(dydx, x0, y0, b, k, h, ys)

    # Exact solution
    exact_solution = np.exp(-k * np.array(euler_x_vals)) + ys

    # Clear previous plots
    for widget in graph_frame.winfo_children():
        widget.destroy()

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(euler_x_vals, euler_y_vals, label="Euler's Method")
    ax.plot(rk_x_vals, rk_y_vals, label="Runge-Kutta Method")
    ax.plot(euler_x_vals, exact_solution, label="Exact Solution", linestyle='--', color='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)  # Use tk.TOP instead of ttk.TOP

    # Display the results in labels
    result_label = Label(inputs_frame, text=f"Results for interval [{a}, {b}]:")
    result_label.grid(row=7, column=0, columnspan=2, pady=10)

    euler_result_label = Label(inputs_frame, text=f"Euler's Method: {euler_y_vals[-1]:.4f}, Error: {np.abs(euler_y_vals[-1] - exact_solution[-1]):.4f}")
    euler_result_label.grid(row=8, column=0, columnspan=2)

    rk_result_label = Label(inputs_frame, text=f"Runge-Kutta Method: {rk_y_vals[-1]:.4f}, Error: {np.abs(rk_y_vals[-1] - exact_solution[-1]):.4f}")
    rk_result_label.grid(row=9, column=0, columnspan=2)




# GUI setup
root = Tk()
root.title("Cauchy Problem Solver")

# Input fields
inputs_frame = Frame(root, padx=10, pady=10)
inputs_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))


k_label = Label(inputs_frame, text="k:")
k_label.grid(row=0, column=0, sticky=tk.W)

k_entry = Entry(inputs_frame)
k_entry.grid(row=0, column=1)

x0_label = Label(inputs_frame, text="x0:")
x0_label.grid(row=1, column=0, sticky=tk.W)
x0_entry = Entry(inputs_frame)
x0_entry.grid(row=1, column=1)

y0_label = Label(inputs_frame, text="y0:")
y0_label.grid(row=2, column=0, sticky=tk.W)
y0_entry = Entry(inputs_frame)
y0_entry.grid(row=2, column=1)

ys_label = Label(inputs_frame, text="ys:")
ys_label.grid(row=3, column=0, sticky=tk.W)
ys_entry = Entry(inputs_frame)
ys_entry.grid(row=3, column=1)

interval_label = Label(inputs_frame, text="Interval [a;b]:")
interval_label.grid(row=4, column=0, sticky=tk.W)
interval_entry = Entry(inputs_frame)
interval_entry.grid(row=4, column=1)

h_label = Label(inputs_frame, text="Step Size (h):")
h_label.grid(row=5, column=0, sticky=tk.W)
h_entry = Entry(inputs_frame)
h_entry.grid(row=5, column=1)

# Compute button
compute_button = Button(inputs_frame, text="Compute", command=compute_and_plot)
compute_button.grid(row=6, column=0, columnspan=2, pady=10)

# Graphical representation
graph_frame = Frame(root)
graph_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

root.mainloop()
