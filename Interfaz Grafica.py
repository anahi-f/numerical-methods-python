import io
import sys
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, Toplevel, Entry, Label, Button, Text, Scrollbar
import numpy as np
from sympy import lambdify, diff, symbols, sympify
from metodos_ecuaciones_lineales import *
from metodos_ecuaciones_no_lineales import *
from metodos_regresion import *
from metodos_interpolacion import *
from metodos_integracion import *



try:
    from ttkthemes import ThemedTk
    themed_available = True
except ImportError:
    themed_available = False

root = ThemedTk(theme="arc") if themed_available else tk.Tk()
root.title("Métodos Numéricos")
root.geometry("700x500")
root.configure(bg="#f0f0f0")
    
selected_option = tk.StringVar()
selected_method = tk.StringVar()

options = [
    "Sistema de ecuaciones lineales",
    "Ecuaciones no lineales",
    "Regresión lineal y polinomial",
    "Métodos para la interpolación",
    "Cálculo de integrales o derivadas"
]

methods_dict = {
    "Sistema de ecuaciones lineales": ["Gauss sin Pivote", "Gauss con Pivote", "Gauss Seidel"],
    "Ecuaciones no lineales": ["Bisección", "Newton-Raphson", "Secante"],
    "Regresión lineal y polinomial": ["Regresión Lineal", "Regresión Polinomial"],
    "Métodos para la interpolación": ["Interpolación de Lagrange", "Interpolación Spline"],
    "Cálculo de integrales o derivadas": ["Método del Trapecio", "Método de Simpson 1/3", "Método de Simpson 3/8", "Derivación numérica hacia adelante", "Derivación numérica hacia atrás", "Derivación numérica centrada", "Extrapolación de Richardson", "Runge Kutta"]
}

def update_methods(event):
    selected = selected_option.get()
    method_menu["values"] = methods_dict.get(selected, [])
    selected_method.set("")
    feedback_label.config(text=f"Seleccionaste: {selected}", foreground="blue")
    check_enable_execute()

def check_enable_execute(*args):
    if selected_option.get() and selected_method.get():
        execute_button.config(state=tk.NORMAL)
    else:
        execute_button.config(state=tk.DISABLED)


def execute_method():
    option = selected_option.get()
    method = selected_method.get()
    x = symbols('x')

    if not option or not method:
        messagebox.showerror("Error", "Seleccione una opción y un método.")
        return
    
    feedback_label.config(text=f"Ejecutando {method} de {option}...", foreground="green")
    root.update_idletasks()

    if option == "Sistema de ecuaciones lineales":
        try:
            m = simpledialog.askinteger("Número de ecuaciones", "Ingrese la cantidad de ecuaciones (entre 2 y 5):", minvalue=2, maxvalue=5)
            if m is None:
                return
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido.")
            return

        # Crear una ventana para ingresar los coeficientes y términos independientes
        ventana = Toplevel()
        ventana.title("Ingresar coeficientes y términos independientes")

        entradas = []
        Label(ventana, text="Ingrese los coeficientes del sistema:").grid(row=0, column=0, columnspan=m, pady=5)

        for i in range(m):
            fila = []
            for j in range(m):
                entry = Entry(ventana, width=5)
                entry.grid(row=i + 1, column=j, padx=5, pady=5)
                fila.append(entry)
            entradas.append(fila)

        Label(ventana, text="Términos independientes:").grid(row=0, column=m, padx=10)

        valores_independientes = []
        for i in range(m):
            entry = Entry(ventana, width=5)
            entry.grid(row=i + 1, column=m, padx=5, pady=5)
            valores_independientes.append(entry)

        def obtener_datos():
            try:
                Ecuaciones = [[float(entrada.get()) for entrada in fila] for fila in entradas]
                ValorInd = [float(entry.get()) for entry in valores_independientes]
                ventana.destroy()

                # Barra de progreso
                progress["value"] = 0
                for i in range(1, 101, 10):
                    progress["value"] = i
                    root.update_idletasks()
                    root.after(100)
                
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()

                if method == "Gauss sin Pivote":
                    resultado = gauss_sin_pivote(Ecuaciones, ValorInd)
                elif method == "Gauss con Pivote":
                    resultado = gauss_con_pivote(Ecuaciones, ValorInd)
                elif method == "Gauss Seidel":
                    resultado = gauss_seidel(Ecuaciones, ValorInd)

                sys.stdout = old_stdout

                detalles_calculo = mystdout.getvalue()
                messagebox.showinfo("Solución", f"{detalles_calculo}\n\nSolución del sistema: {resultado}")

            except ValueError:
                messagebox.showerror("Error", "Ingrese valores numéricos válidos.")

        Button(ventana, text="Aceptar", command=obtener_datos).grid(row=m+1, column=0, columnspan=m+1, pady=10)
    
    if option == "Ecuaciones no lineales":
        func_str = simpledialog.askstring("Función", "Ingrese la función F(x):")
        if not func_str:
            return  
        
        try:

            expr = sympify(func_str, evaluate=True)

            if expr.is_Number:
                raise ValueError("La función ingresada no depende de x.")
        
            Fx = lambdify(x, expr, modules=["numpy"])
            derivadax = lambdify(x, diff(expr, x), modules=["numpy"])
        except Exception as e:
            messagebox.showerror("Error", f"La función ingresada no es válida.\nDetalles: {str(e)}")
            return
        
        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        try:
            if method == "Bisección":
                a = simpledialog.askfloat("Bisección", "Ingrese un punto menor:")
                b = simpledialog.askfloat("Bisección", "Ingrese un punto mayor:")
                if a is None or b is None:
                    return  
                raiz = Biseccion(Fx, a, b)

                mostrar_resultado_con_grafica(Fx, a, b, raiz, 'Bisección')

            elif method == "Newton-Raphson":
                xi = simpledialog.askfloat("Newton-Raphson", "Ingrese un punto inicial:")
                if xi is None:
                    return  
                raiz = NewtonRaphson(Fx, derivadax, xi)
                if raiz is None:
                    messagebox.showerror("Error", "Newton-Raphson falló debido a una división por cero.")
                    return
                mostrar_resultado_con_grafica(Fx, xi - 5, xi + 5, raiz, "Newton-Raphson")

            elif method == "Secante":
                Xn = simpledialog.askfloat("Secante", "Ingrese el punto Xi:")
                Xni = simpledialog.askfloat("Secante", "Ingrese el punto Xi-1:")
                if Xn is None or Xni is None:
                    return  
                raiz = Secante(Fx, Xn, Xni)
                if raiz is None:
                    messagebox.showerror("Error", "El método de la Secante falló debido a una división por cero.")
                    return
                mostrar_resultado_con_grafica(Fx, Xn - 5, Xn + 5, raiz, "Secante")
        except Exception as e:
            messagebox.showerror("Error", f"El método seleccionado no es válido.\nDetalles: {str(e)}")
            sys.stdout = old_stdout  
            return

    if option == "Regresión lineal y polinomial":
        method = selected_method.get()
    
        try:
            n = simpledialog.askinteger("Número de datos", "Ingrese la cantidad de términos:", minvalue=2)
            if n is None:
                return
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido.")
            return

        ventana_datos = Toplevel()
        ventana_datos.title("Ingresar datos X e Y")

        Label(ventana_datos, text="Ingrese los valores de X y Y").grid(row=0, column=0, columnspan=2, pady=5)

        entradas_x, entradas_y = [], []
        for i in range(n):
            Label(ventana_datos, text=f"x[{i}]:").grid(row=i+1, column=0)
            entrada_x = Entry(ventana_datos, width=10)
            entrada_x.grid(row=i+1, column=1)
            entradas_x.append(entrada_x)

            Label(ventana_datos, text=f"y[{i}]:").grid(row=i+1, column=2)
            entrada_y = Entry(ventana_datos, width=10)
            entrada_y.grid(row=i+1, column=3)
            entradas_y.append(entrada_y)

        def procesar_datos():
            try:
                xz = [float(entry.get()) for entry in entradas_x]
                y = [float(entry.get()) for entry in entradas_y]
                ventana_datos.destroy()

                # Calculando sumatorias
                Ex = sum(xz)
                Ey = sum(y)
                xcuadr = [x**2 for x in xz]
                xy = [x * y for x, y in zip(xz, y)]
                Excuadr = sum(xcuadr)
                Exy = sum(xy)

                if method == "Regresión Lineal":
                    pendiente, corte = Regresionlineal(n, Ex, Ey, Exy, Excuadr)
                    mostrar_resultado_regresion(n, Ex, Ey, Exy, Excuadr, xz, y, "Regresión Lineal", pendiente=pendiente, corte=corte)

                elif method == "Regresión Polinomial":
                    xcubo = [x**3 for x in xz]
                    xcuarta = [x**4 for x in xz]
                    x2y = [x**2 * y for x, y in zip(xz, y)]
                    Excubo = sum(xcubo)
                    Excuarta = sum(xcuarta)
                    Ex2y = sum(x2y)

                    A0, A1, A2 = RegresionPolinomial(n, Ex, Ey, Exy, Excuadr, Excubo, Excuarta, Ex2y)
                    mostrar_resultado_regresion(n, Ex, Ey, Exy, Excuadr, xz, y, "Regresión Polinomial", A0=A0, A1=A1, A2=A2)

            except ValueError:
                messagebox.showerror("Error", "Ingrese valores numéricos válidos.")

        Button(ventana_datos, text="Aceptar", command=procesar_datos).grid(row=n+1, column=0, columnspan=4, pady=10)

    if option == "Métodos para la interpolación":
        method = selected_method.get()

        try:
            n = simpledialog.askinteger("Número de datos", "Ingrese la cantidad de términos:", minvalue=2)
            if n is None:
                return  
        except ValueError:
            messagebox.showerror("Error", "Ingrese un número válido.")
            return

        ventana_datos = Toplevel()
        ventana_datos.title("Ingresar datos X e Y")

        Label(ventana_datos, text="Ingrese los valores de X y Y").grid(row=0, column=0, columnspan=2, pady=5)

        entradas_x, entradas_y = [], []
        for i in range(n):
            Label(ventana_datos, text=f"x[{i}]:").grid(row=i+1, column=0)
            entrada_x = Entry(ventana_datos, width=10)
            entrada_x.grid(row=i+1, column=1)
            entradas_x.append(entrada_x)

            Label(ventana_datos, text=f"y[{i}]:").grid(row=i+1, column=2)
            entrada_y = Entry(ventana_datos, width=10)
            entrada_y.grid(row=i+1, column=3)
            entradas_y.append(entrada_y)

        def procesar_datos():
            try:
                xz = [float(entry.get()) for entry in entradas_x]
                y = [float(entry.get()) for entry in entradas_y]
                ventana_datos.destroy()

                if method == "Interpolación de Lagrange":
                    polinomio_func, polinomio_str = Lagrange(xz, y)
                    x_val = np.linspace(min(xz) - 1, max(xz) + 1, 100)
                    mostrar_resultado_interpolacion(xz, y, x_val, polinomio_func, "Interpolación de Lagrange", polinomio_str)

                elif method == "Interpolación Spline":
                    spline_tipo = simpledialog.askinteger(
                        "Método Spline",
                        "Seleccione el tipo de interpolación spline:\n1. Lineal\n2. Cuadrática\n3. Cúbica\n4. Comparación de todas",
                        minvalue=1, maxvalue=4
                    )
                    if spline_tipo is None:
                        return

                    x_val = np.linspace(min(xz), max(xz), 100)

                    if spline_tipo == 1:
                        y_spline = spline_lineal(xz, y)
                        mostrar_resultado_interpolacion(xz, y, x_val, y_spline, "Spline Lineal")

                    elif spline_tipo == 2:
                        y_spline = spline_cuadratico(xz, y)
                        mostrar_resultado_interpolacion(xz, y, x_val, y_spline, "Spline Cuadrático")

                    elif spline_tipo == 3:
                        y_spline = spline_cubico(xz, y)
                        mostrar_resultado_interpolacion(xz, y, x_val, y_spline, "Spline Cúbico")

                    elif spline_tipo == 4:
                        y_lineal = spline_lineal(xz, y)
                        y_cuadratico = spline_cuadratico(xz, y)
                        y_cubico = spline_cubico(xz, y)

                        ventana_resultado = Toplevel()
                        ventana_resultado.title("Comparación de Splines")

                        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                        ax.plot(xz, y, 'o', label="Puntos conocidos")
                        ax.plot(x_val, y_lineal(x_val), '-', label="Spline Lineal")
                        ax.plot(x_val, y_cuadratico(x_val), '--', label="Spline Cuadrático")
                        ax.plot(x_val, y_cubico(x_val), '-.', label="Spline Cúbico")

                        ax.set_title("Comparación de Métodos Spline")
                        ax.set_xlabel("Eje x")
                        ax.set_ylabel("Eje y")
                        ax.legend()
                        ax.grid(True)

                        canvas = FigureCanvasTkAgg(fig, master=ventana_resultado)
                        canvas.draw()
                        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                        toolbar = NavigationToolbar2Tk(canvas, ventana_resultado)
                        toolbar.update()
                        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                        Button(ventana_resultado, text="Cerrar", command=ventana_resultado.destroy).pack(pady=10)

            except ValueError:
                messagebox.showerror("Error", "Ingrese valores numéricos válidos.")

        Button(ventana_datos, text="Aceptar", command=procesar_datos).grid(row=n+1, column=0, columnspan=4, pady=10)

    if option == "Cálculo de integrales o derivadas":
        method = selected_method.get()

        func_str = simpledialog.askstring("Función", "Ingrese la función F(x):")
        if not func_str:
            return
        try:
            x, y = symbols('x y')
            expr = sympify(func_str, evaluate=True)
            Fx = lambdify(x, expr, modules=["numpy"])
            Fxy = lambdify((x, y), expr, modules=["numpy"])
        except Exception as e:
            messagebox.showerror("Error", f"La función ingresada no es válida.\nDetalles: {str(e)}")
            return
        

        try:
            if method in ["Método del Trapecio", "Método de Simpson 1/3", "Método de Simpson 3/8"]:
                ventana_integral = Toplevel()
                ventana_integral.title("Límites de Integración")

                Label(ventana_integral, text="Límite inferior:").grid(row=0, column=0)
                entry_a = Entry(ventana_integral, width=10)
                entry_a.grid(row=0, column=1)

                Label(ventana_integral, text="Límite superior:").grid(row=1, column=0)
                entry_b = Entry(ventana_integral, width=10)
                entry_b.grid(row=1, column=1)

                Label(ventana_integral, text="Número de subdivisiones:").grid(row=2, column=0)
                entry_n = Entry(ventana_integral, width=10)
                entry_n.grid(row=2, column=1)

                def calcular_integral():
                    try:
                        a = float(entry_a.get())
                        b = float(entry_b.get())
                        n = int(entry_n.get())
                        if b <= a:
                            messagebox.showerror("Error", "Ingrese límites válidos donde b > a.")
                            return

                        if method == "Método del Trapecio":
                            resultado = Metodo_del_trapecio(Fx, a, b, n)

                        elif method == "Método de Simpson 1/3":
                            resultado = Metodo_de_simpson13(Fx, a, b, n)

                        elif method == "Método de Simpson 3/8":
                            resultado = Metodo_de_simpson38(Fx, a, b, n)
                        
                        mostrar_resultado_integracion(Fx, a, b, resultado, method)

                        ventana_integral.destroy()

                    except ValueError:
                        messagebox.showerror("Error", "Ingrese valores numéricos válidos.")

                Button(ventana_integral, text="Calcular", command=calcular_integral).grid(row=3, column=0, columnspan=2, pady=10)

            elif method in ["Derivación numérica hacia adelante", "Derivación numérica hacia atrás", "Derivación numérica centrada", "Extrapolación de Richardson"]:
                ventana_derivada = Toplevel()
                ventana_derivada.title(f"Cálculo de {method}")

                Label(ventana_derivada, text="📌 Punto de evaluación:").grid(row=0, column=0)
                entry_punto = Entry(ventana_derivada, width=10)
                entry_punto.grid(row=0, column=1)

                Label(ventana_derivada, text="📌 Distancia al punto (h):").grid(row=1, column=0)
                entry_h = Entry(ventana_derivada, width=10)
                entry_h.grid(row=1, column=1)

                metodo_codigo = {
                    "Derivación numérica hacia adelante": 4,
                    "Derivación numérica hacia atrás": 5,
                    "Derivación numérica centrada": 6,
                    "Extrapolación de Richardson": 7 
                }[method]

                def calcular_derivada():
                    try:
                        puntprinc = float(entry_punto.get())
                        h = float(entry_h.get())

                        resultado = Derivada_numericahacia(Fx, puntprinc, h, metodo_codigo)
                        ventana_derivada.destroy()

                        ventana_resultado = Toplevel()
                        ventana_resultado.title("Resultado de la Derivada")

                        Label(ventana_resultado, text=f"📌 Método: {method}", font=("Arial", 12, "bold")).pack(pady=5)
                        Label(ventana_resultado, text=f"📌 Punto de evaluación: {puntprinc}").pack()
                        Label(ventana_resultado, text=f"📌 Distancia h: {h}").pack()
                        Label(ventana_resultado, text=f"📌 Resultado aproximado de la derivada: {resultado:.6f}").pack(pady=10)

                        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                        x_vals = np.linspace(puntprinc - 2, puntprinc + 2, 100)
                        y_vals = np.array([Fx(x) for x in x_vals])
                        y_tangente = resultado * (x_vals - puntprinc) + Fx(puntprinc)

                        ax.plot(x_vals, y_vals, label="Función original")
                        ax.plot(x_vals, y_tangente, linestyle="--", color="red", label="Recta Tangente")
                        ax.scatter([puntprinc], [Fx(puntprinc)], color="black", label=f"Punto Evaluado ({puntprinc}, {Fx(puntprinc):.2f})")

                        ax.set_title(f"Gráfica de la derivada - {method}")
                        ax.set_xlabel("x")
                        ax.set_ylabel("y")
                        ax.legend()
                        ax.grid(True)

                        canvas = FigureCanvasTkAgg(fig, master=ventana_resultado)
                        canvas.draw()
                        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                        Button(ventana_resultado, text="Cerrar", command=ventana_resultado.destroy).pack(pady=5)

                    except ValueError:
                        messagebox.showerror("Error", "Ingrese valores numéricos válidos.")

                Button(ventana_derivada, text="Calcular", command=calcular_derivada).grid(row=2, column=0, columnspan=2, pady=10)

            if method == "Runge Kutta":
                try:
                    x, y = symbols('x y')
                    expr = sympify(func_str, evaluate=True)
                    Fxy = lambdify((x, y), expr, modules=["numpy"])  

                    ventana_runge = Toplevel()
                    ventana_runge.title("Cálculo con Runge-Kutta")

                    Label(ventana_runge, text="📌 Valor inicial de x0:").grid(row=0, column=0)
                    entry_x0 = Entry(ventana_runge, width=10)
                    entry_x0.grid(row=0, column=1)

                    Label(ventana_runge, text="📌 Valor inicial de y0:").grid(row=1, column=0)
                    entry_y0 = Entry(ventana_runge, width=10)
                    entry_y0.grid(row=1, column=1)

                    Label(ventana_runge, text="📌 Paso h:").grid(row=2, column=0)
                    entry_h = Entry(ventana_runge, width=10)
                    entry_h.grid(row=2, column=1)

                    Label(ventana_runge, text="📌 Valor final de x (xn):").grid(row=3, column=0)
                    entry_xn = Entry(ventana_runge, width=10)
                    entry_xn.grid(row=3, column=1)
                    
                    def calcular_runge_kutta():
                        try:
                            x0 = float(entry_x0.get())
                            y0 = float(entry_y0.get())
                            h = float(entry_h.get())
                            xn = float(entry_xn.get())

                            if h == 0:
                                messagebox.showerror("Error", "El valor de h no puede ser 0.")
                                return
                            if abs((xn - x0) / h - round((xn - x0) / h)) > 1e-6:
                                messagebox.showerror("Error", "El paso h no permite alcanzar xn exactamente. Ajuste los valores.")
                                return

                            resultado, x_vals, y_vals, iteraciones = Runge_Kutta(Fxy, x0, y0, h, xn)
                            ventana_runge.destroy()

                            ventana_resultado = Toplevel()
                            ventana_resultado.title("Resultado de Runge-Kutta")

                            Label(ventana_resultado, text=f"📌 Método: Runge-Kutta", font=("Arial", 12, "bold")).pack(pady=5)
                            Label(ventana_resultado, text=f"📌 x0: {x0}, y0: {y0}").pack()
                            Label(ventana_resultado, text=f"📌 Paso h: {h}").pack()
                            Label(ventana_resultado, text=f"📌 xn: {xn}").pack()
                            Label(ventana_resultado, text=f"📌 Resultado aproximado: {resultado:.6f}").pack(pady=10)

                            Label(ventana_resultado, text="📌 Iteraciones:").pack()
                            text_widget = Text(ventana_resultado, height=10, width=50)
                            scrollbar = Scrollbar(ventana_resultado, command=text_widget.yview)
                            text_widget.config(yscrollcommand=scrollbar.set)

                            for i, x_val, y_val in iteraciones:
                                text_widget.insert(tk.END, f"Iter {i}: x = {x_val:.4f}, y = {y_val:.6f}\n")

                            text_widget.pack()
                            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

                            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
                            ax.plot(x_vals, y_vals, marker='o', linestyle='-', color='blue', label="Solución Runge-Kutta")

                            for i, (x_val, y_val) in enumerate(zip(x_vals, y_vals)):
                                ax.annotate(f"[{i}]", (x_val, y_val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="red")

                            ax.set_title("Solución de Runge-Kutta con iteraciones")
                            ax.set_xlabel("x")
                            ax.set_ylabel("y")
                            ax.legend()
                            ax.grid(True)

                            canvas = FigureCanvasTkAgg(fig, master=ventana_resultado)
                            canvas.draw()
                            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

                            # Botón para cerrar la ventana
                            Button(ventana_resultado, text="Cerrar", command=ventana_resultado.destroy).pack(pady=5)


                        except ValueError:
                            messagebox.showerror("Error", "Ingrese valores numéricos válidos.")

                    Button(ventana_runge, text="Calcular", command=calcular_runge_kutta).grid(row=4, column=0, columnspan=2, pady=10)

                except Exception as e:
                    messagebox.showerror("Error", f"La función ingresada no es válida.\nDetalles: {str(e)}")
                    return
        except Exception as e:
            messagebox.showerror("Error", f"Error durante el cálculo: {str(e)}")
            return

style = ttk.Style()
style.configure("TFrame", background="#f0f0f0")
style.configure("TLabel", background="#f0f0f0", font=("Arial", 14))
style.configure("TButton", font=("Arial", 12), padding=10)
style.configure("TCombobox", font=("Arial", 12))

frame = ttk.Frame(root, padding="20")
frame.pack(fill=tk.BOTH, expand=True)

title_label = ttk.Label(frame, text="Métodos Numéricos", font=("Arial", 20, "bold"), foreground="#333")
title_label.pack(pady=15)


option_menu = ttk.Combobox(frame, textvariable=selected_option, values=options, state="readonly")
option_menu.set("Seleccione una opción")
option_menu.bind("<<ComboboxSelected>>", update_methods)
option_menu.pack(pady=10, fill=tk.X)

method_menu = ttk.Combobox(frame, textvariable=selected_method, state="readonly")
method_menu.pack(pady=10, fill=tk.X)

selected_option.trace_add("write", check_enable_execute)
selected_method.trace_add("write", check_enable_execute)

feedback_label = ttk.Label(frame, text="Seleccione una opción y método", font=("Arial", 12), foreground="red")
feedback_label.pack(pady=5)

progress = ttk.Progressbar(frame, length=200, mode="determinate")
progress.pack(pady=10, fill=tk.X)

execute_button = ttk.Button(frame, text="Ejecutar Método", command=execute_method, state=tk.DISABLED)
execute_button.pack(pady=20, fill=tk.X)


root.mainloop()
