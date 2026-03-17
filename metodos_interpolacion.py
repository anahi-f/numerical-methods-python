import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import Toplevel, Label, Text, Button, Scrollbar
from sympy import symbols, expand
from scipy.interpolate import interp1d

def mostrar_resultado_interpolacion(xz, y, x_val, y_val, metodo, polinomio_str=None):
    """
    Muestra los resultados de interpolación (Lagrange o Spline) en una ventana con la gráfica interactiva.
    """
    ventana_resultado = Toplevel()
    ventana_resultado.title(f"Resultados - {metodo}")

    # Crear un frame para los resultados
    frame_texto = tk.Frame(ventana_resultado)
    frame_texto.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    Label(frame_texto, text=f"Método: {metodo}", font=("Arial", 12, "bold")).pack()

    detalles_calculo = "📌 **Puntos de interpolación:**\n"
    for i in range(len(xz)):
        detalles_calculo += f"({xz[i]}, {y[i]})\n"

    if polinomio_str:
        detalles_calculo += f"\n📌 **Polinomio interpolante:**\n{polinomio_str}"

    text_area = Text(frame_texto, wrap="word", height=10, width=60)
    text_area.insert("1.0", detalles_calculo)
    text_area.config(state="disabled")
    text_area.pack(pady=5, padx=5)

    # Crear el frame donde irá la gráfica interactiva
    frame_grafica = tk.Frame(ventana_resultado)
    frame_grafica.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # Crear la gráfica
    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    ax.plot(xz, y, 'o', label="Puntos conocidos", markersize=8, color='red')
    ax.plot(x_val, y_val(x_val), '-', label=metodo, linewidth=2, color='blue')

    ax.set_title(f"Gráfica de {metodo}")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.legend()
    ax.grid(True)

    # Agregar la gráfica a la interfaz de Tkinter con interacción
    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Agregar barra de herramientas para zoom y navegación
    toolbar = NavigationToolbar2Tk(canvas, frame_grafica)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    Button(ventana_resultado, text="Cerrar", command=ventana_resultado.destroy).pack(pady=10)

def Lagrange(x_values, y_values):
    """
    Genera el polinomio interpolante de Lagrange y su representación simbólica.
    """
    x = symbols('x')
    n = len(x_values)
    polinomio = 0
    for i in range(n):
        term = y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - x_values[j]) / (x_values[i] - x_values[j])
        polinomio += term
    polinomio_simplificado = expand(polinomio)

    def polinomio_func(x_eval):
        result = 0
        for i in range(n):
            term = y_values[i]
            for j in range(n):
                if i != j:
                    term *= (x_eval - x_values[j]) / (x_values[i] - x_values[j])
            result += term
        return result

    return lambda x: polinomio_func(x), polinomio_simplificado

def spline_lineal(xz, y):
    """ Genera un Spline Lineal """
    return interp1d(xz, y, kind="linear")

def spline_cuadratico(xz, y):
    """ Genera un Spline Cuadrático """
    return interp1d(xz, y, kind="quadratic")

def spline_cubico(xz, y):
    """ Genera un Spline Cúbico """
    return interp1d(xz, y, kind="cubic")
