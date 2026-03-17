import numpy as np

def gauss_sin_pivote(Ecuaciones, ValorInd):
    Ecuaciones = np.array(Ecuaciones, dtype=float)
    ValorInd = np.array(ValorInd, dtype=float).reshape(-1, 1)
    n = len(ValorInd)
    ampliada = np.hstack([Ecuaciones, ValorInd])
    print("Matriz ampliada inicial:")
    print(ampliada)
    for i in range(n):
        for j in range(i + 1, n):
            if ampliada[i, i] == 0:
                raise ValueError("El primer elemento es cero. Método sin pivoteo no funciona.")
            factor = ampliada[j, i] / ampliada[i, i]
            ampliada[j, i:] -= factor * ampliada[i, i:]
        print("Matriz ampliada después de pivotear elemento {}:".format(i))
        print(ampliada)
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (ampliada[i, -1] - np.dot(ampliada[i, i + 1:n], x[i + 1:])) / ampliada[i, i]
    
    return x


def gauss_con_pivote(Ecuaciones, ValorInd):
    Ecuaciones = np.array(Ecuaciones, dtype=float)
    ValorInd = np.array(ValorInd, dtype=float).reshape(-1, 1)
    n = len(ValorInd)
    ampliada = np.hstack([Ecuaciones, ValorInd])
    print("Matriz ampliada inicial:")
    print(ampliada)
    for i in range(n):
        max_row = np.argmax(abs(ampliada[i:, i])) + i
        if i != max_row:
            ampliada[[i, max_row]] = ampliada[[max_row, i]]
        
        for j in range(i + 1, n):
            if ampliada[i, i] == 0:
                raise ValueError("Error encontrado, no se puede continuar.")
            factor = ampliada[j, i] / ampliada[i, i]
            ampliada[j, i:] -= factor * ampliada[i, i:]
        print("Matriz ampliada después de pivotear elemento {}:".format(i))
        print(ampliada)             
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (ampliada[i, -1] - np.dot(ampliada[i, i + 1:n], x[i + 1:])) / ampliada[i, i]
    
    return x


def gauss_seidel(Ecuaciones, ValorInd, x0=None, tol=1e-6, max_iter=100):
    n = len(ValorInd)
    if x0 is None:
        x0 = np.zeros(n)
    
    x = x0.copy()
    
    print("Matriz de ecuaciones:")
    print(np.array(Ecuaciones, dtype=float))
    print("Vector de valores independientes:")
    print(np.array(ValorInd, dtype=float).reshape(-1, 1))

    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum1 = sum(Ecuaciones[i][j] * x[j] for j in range(i))
            sum2 = sum(Ecuaciones[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (ValorInd[i] - sum1 - sum2) / Ecuaciones[i][i]

        error = np.linalg.norm(x - x_old, ord=np.inf)

        if error < tol:
            print("\nConvergencia alcanzada.\n")
            break

    return x

