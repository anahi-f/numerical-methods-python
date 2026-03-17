import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import Toplevel, Label, Text, Button, Scrollbar
from sympy import symbols, lambdify

x = symbols('x')

def mostrar_resultado_integracion(Fx, a, b, resultado, metodo):
    """
    Muestra los resultados de integración en una ventana con la gráfica interactiva.
    """
    ventana_resultado = Toplevel()
    ventana_resultado.title(f"Resultados - {metodo}")

    # Frame para mostrar texto
    frame_texto = tk.Frame(ventana_resultado)
    frame_texto.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    Label(frame_texto, text=f"Método: {metodo}", font=("Arial", 12, "bold")).pack()

    detalles_calculo = f"""
📌 **Límites de integración:**
- Inferior (a): {a}
- Superior (b): {b}

📌 **Resultado aproximado de la integral:**
{resultado:.6f}
"""

    text_area = Text(frame_texto, wrap="word", height=8, width=60)
    text_area.insert("1.0", detalles_calculo)
    text_area.config(state="disabled")
    text_area.pack(pady=5, padx=5)

    frame_grafica = tk.Frame(ventana_resultado)
    frame_grafica.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
    x_vals = np.linspace(a, b, 100)
    y_vals = np.array([Fx(x) for x in x_vals])

    ax.fill_between(x_vals, y_vals, alpha=0.3, label="Área bajo la curva")
    ax.plot(x_vals, y_vals, label="Función", linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(a, color="red", linestyle="--", label="Límite inferior")
    ax.axvline(b, color="blue", linestyle="--", label="Límite superior")

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

def Metodo_del_trapecio(Fx, a, b, n):
    """
    Método del Trapecio compuesto para aproximar integrales.
    """
    h = (b - a) / n
    sumatoria = sum(Fx(a + i * h) for i in range(1, n))
    resultado = (h / 2) * (Fx(a) + 2 * sumatoria + Fx(b))
    return resultado

def Metodo_de_simpson13(Fx, a, b, n):
    """
    Método de Simpson 1/3 para aproximar integrales.
    """
    h = (b - a) / n
    sumatoria1 = sum(Fx(a + i * h) for i in range(1, n, 2))
    sumatoria2 = sum(Fx(a + i * h) for i in range(2, n-1, 2))
    resultado = (h / 3) * (Fx(a) + 4 * sumatoria1 + 2 * sumatoria2 + Fx(b))
    return resultado

def Metodo_de_simpson38(Fx, a, b, n):
    """
    Método de Simpson 3/8 para aproximar integrales.
    """
    h = (b - a) / n
    sumatoria1 = sum(Fx(a + i * h) for i in range(1, n, 3))
    sumatoria2 = sum(Fx(a + i * h) for i in range(2, n-1, 3))
    resultado = ((3 * h) / 8) * (Fx(a) + 3 * sumatoria1 + 3 * sumatoria2 + Fx(b))
    return resultado

def Derivada_numericahacia(Fx, x, h, metodo):
    """
    Cálculo de derivadas numéricas con los métodos:
    - Adelante (4)
    - Atrás (5)
    - Centrada (6)
    """
    if metodo == 4:
        resultado = (Fx(x + h) - Fx(x)) / h
    elif metodo == 5:
        resultado = (Fx(x) - Fx(x - h)) / h
    elif metodo == 6:
        resultado = (Fx(x + h) - Fx(x - h)) / (2 * h)
    elif metodo == 7:
        resultado = Extrapolacion_Richardson(Fx, x, h)
    return resultado

def Extrapolacion_Richardson(Fx, x, h):
    D1 = (Fx(x + h) - Fx(x - h)) / (2 * h)
    D2 = (Fx(x + h/2) - Fx(x - h/2)) / h
    resultado = (4 * D2 - D1) / 3
    return resultado

def Runge_Kutta(Fxy, x0, y0, h, xn):
    if h == 0:
        raise ValueError("El paso h no puede ser 0.")

    n = int((xn - x0) / h)
    if n == 0:
        raise ValueError("El valor de n es 0, lo que indica que h es demasiado grande o x0 = xn.")

    x, y = x0, y0
    x_vals, y_vals = [x], [y]
    iteraciones = [(0, x, y)]  # Guardar la iteración inicial (0)

    for i in range(1, n + 1):  # Comenzamos desde 1 para que la numeración sea correcta
        k1 = h * Fxy(x, y)
        k2 = h * Fxy(x + h / 2, y + k1 / 2)
        k3 = h * Fxy(x + h / 2, y + k2 / 2)
        k4 = h * Fxy(x + h, y + k3)

        y += (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x += h

        x_vals.append(x)
        y_vals.append(y)
        iteraciones.append((i, x, y))  # Ahora se usa `i` en lugar de `n`

    return y, x_vals, y_vals, iteraciones
