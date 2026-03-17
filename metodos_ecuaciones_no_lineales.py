
import tkinter as tk
from tkinter import Toplevel, Label, Button, Text
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
x = symbols('x')

def Biseccion(Fx, a, b, tol=1e-6):
    while abs(b - a) > tol:
        c = (a + b) / 2
        if Fx(c) == 0:
            break
        elif Fx(a) * Fx(c) < 0:
            b = c
        else:
            a = c
    return c

def NewtonRaphson(Fx, derivada, xi, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        try:
            if abs(derivada(xi)) < 1e-10:
                raise ZeroDivisionError("La derivada se anuló en x = {:.6f}. El método no puede continuar.".format(xi))
            xi_new = xi - Fx(xi) / derivada(xi)
            if abs(xi_new - xi) < tol:
                return xi_new
            xi = xi_new
        except ZeroDivisionError as e:
            print(f"Error en Newton-Raphson: {e}")
            return None
    return xi

def Secante(Fx, Xn, Xni, tol=1e-6, max_iter=100):
    for _ in range(max_iter):
        try:
            if abs(Fx(Xni) - Fx(Xn)) < 1e-10:
                raise ZeroDivisionError("Diferencia demasiado pequeña entre F(Xi) y F(Xi-1).")
            Xnew = Xn - ((Fx(Xn) * (Xni - Xn)) / (Fx(Xni) - Fx(Xn)))
            if abs(Xnew - Xn) < tol:
                return Xnew
            Xni, Xn = Xn, Xnew
        except ZeroDivisionError as e:
            print(f"Error en el método de la secante: {e}")
            return None
    return Xn

def mostrar_resultado_con_grafica(Fx, a, b, raiz, metodo):
    """
    Crea una ventana donde se muestra la raíz calculada y la gráfica de la función.
    """
    ventana_resultado = Toplevel()
    ventana_resultado.title(f"Resultado y Gráfica - {metodo}")

    # Crear un frame para los resultados
    frame_texto = tk.Frame(ventana_resultado)
    frame_texto.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    Label(frame_texto, text=f"Método: {metodo}", font=("Arial", 12, "bold")).pack()

    text_area = Text(frame_texto, wrap="word", height=5, width=60)
    text_area.insert("1.0", f"Raíz aproximada: {raiz:.6f}")
    text_area.config(state="disabled")
    text_area.pack(pady=5, padx=5)

    # Crear la gráfica
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
    x_vals = np.linspace(a - 2, b + 2, 500)
    y_vals = np.array([Fx(x) for x in x_vals])
    ax.plot(x_vals, y_vals, label="Función")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(raiz, color="red", linewidth=0.8, linestyle="--", label="Raíz (aproximada)")
    ax.set_title(f"Método de {metodo}")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.legend()
    ax.grid(True)

    # Integrar la gráfica en la ventana de Tkinter
    canvas = FigureCanvasTkAgg(fig, master=ventana_resultado)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, ventana_resultado)
    toolbar.update()
    canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    Button(ventana_resultado, text="Cerrar", command=ventana_resultado.destroy).pack(pady=10)