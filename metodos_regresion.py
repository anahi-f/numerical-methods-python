import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import Toplevel, Label, Text, Button, Scrollbar

def mostrar_resultado_regresion(n, Ex, Ey, Exy, Excuadr, xz, y, metodo, pendiente=None, corte=None, A0=None, A1=None, A2=None):
    """
    Muestra los resultados de regresión (lineal o polinomial) en una ventana con la gráfica interactiva.
    """
    ventana_resultado = Toplevel()
    ventana_resultado.title(f"Resultados - {metodo}")

    frame_texto = tk.Frame(ventana_resultado)
    frame_texto.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    Label(frame_texto, text=f"Método: {metodo}", font=("Arial", 12, "bold")).pack()

    detalles_calculo = f"""
    Sumatoria de X: {Ex:.4f}
    Sumatoria de Y: {Ey:.4f}
    Sumatoria de X^2: {Excuadr:.4f}
    Sumatoria de X*Y: {Exy:.4f}
    """

    if metodo == "Regresión Lineal":
        formula = f"y = {corte:.4f} + ({pendiente:.4f}) * x"
        detalles_calculo += f"\nEcuación de la regresión lineal:\n{formula}"
    elif metodo == "Regresión Polinomial":
        formula = f"y = {A0:.4f} + ({A1:.4f}) * x + ({A2:.4f}) * x^2"
        detalles_calculo += f"\nEcuación de la regresión polinomial:\n{formula}"

    text_area = Text(frame_texto, wrap="word", height=8, width=60)
    text_area.insert("1.0", detalles_calculo)
    text_area.config(state="disabled")
    text_area.pack(pady=5, padx=5)

    frame_grafica = tk.Frame(ventana_resultado)
    frame_grafica.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    x = np.linspace(min(xz) - 1, max(xz) + 1, 500)

    if metodo == "Regresión Lineal":
        yz = pendiente * x + corte
        ax.plot(x, yz, label="Línea de regresión", color="blue")

    elif metodo == "Regresión Polinomial":
        yz = A2 * x**2 + A1 * x + A0
        ax.plot(x, yz, label="Curva de regresión", color="blue")

    ax.scatter(xz, y, color="red", label="Datos originales")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"Gráfica de {metodo}")
    ax.set_xlabel("Eje x")
    ax.set_ylabel("Eje y")
    ax.legend()
    ax.grid(True)

    canvas = FigureCanvasTkAgg(fig, master=frame_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, frame_grafica)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    Button(ventana_resultado, text="Cerrar", command=ventana_resultado.destroy).pack(pady=10)

def Regresionlineal(n, Ex, Ey, Exy, Excuadr):
    pendiente = (n * Exy - Ex * Ey) / (n * Excuadr - Ex**2)
    corte = (Ey / n) - (pendiente * Ex) / n
    return pendiente, corte

def RegresionPolinomial(n, Ex, Ey, Exy, Excuadr, Excubo, Excuarta, Ex2y):
    A = np.array([[n, Ex, Excuadr], [Ex, Excuadr, Excubo], [Excuadr, Excubo, Excuarta]])
    b = np.array([Ey, Exy, Ex2y])
    Ab = np.linalg.solve(A, b)
    a0, a1, a2 = Ab[0], Ab[1], Ab[2]
    return a0, a1, a2

