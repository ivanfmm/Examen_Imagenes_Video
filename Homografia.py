import numpy as np


# destino = H * origen


# Normalizacion de Hartley: centra los puntos y los escala
def normalizar(puntos):
    cx = np.mean(puntos[:, 0])
    cy = np.mean(puntos[:, 1])
    distancias = np.sqrt((puntos[:, 0] - cx) ** 2 + (puntos[:, 1] - cy) ** 2)
    promedio = np.mean(distancias)
    if promedio == 0:
        return None, None, None
    s = np.sqrt(2) / promedio

    T = np.array([[s, 0, -s * cx],
                  [0, s, -s * cy],
                  [0, 0, 1]], dtype=np.float64)
    T_inv = np.array([[1 / s, 0, cx],
                      [0, 1 / s, cy],
                      [0, 0, 1]], dtype=np.float64)

    normalizados = np.zeros_like(puntos, dtype=np.float64)
    normalizados[:, 0] = s * (puntos[:, 0] - cx)
    normalizados[:, 1] = s * (puntos[:, 1] - cy)
    return normalizados, T, T_inv


# Resuelve A x = b con eliminacion gaussiana
def resolver_sistema(A, b):
    A = A.astype(np.float64).copy()
    b = b.astype(np.float64).copy()
    n = len(b)

    for col in range(n):
        # Pivoteo
        fila_pivote = col + np.argmax(np.abs(A[col:, col]))
        if abs(A[fila_pivote, col]) < 1e-12:
            return None
        if fila_pivote != col:
            A[[col, fila_pivote]] = A[[fila_pivote, col]]
            b[[col, fila_pivote]] = b[[fila_pivote, col]]

        for fila in range(col + 1, n):
            factor = A[fila, col] / A[col, col]
            A[fila, col:] -= factor * A[col, col:]
            b[fila] -= factor * b[col]

    # Sustitucion hacia atras
    x = np.zeros(n)
    for fila in range(n - 1, -1, -1):
        x[fila] = (b[fila] - np.dot(A[fila, fila + 1:], x[fila + 1:])) / A[fila, fila]
    return x


# DLT con h33 = 1 (8 incognitas, 2 ecuaciones por pareja)
def estimar_homografia(origen, destino):
    if len(origen) < 4:
        return None

    o, T_o, _ = normalizar(origen)
    d, _, T_d_inv = normalizar(destino)
    if o is None or d is None:
        return None

    filas_A = []
    filas_b = []
    for (x, y), (xp, yp) in zip(o, d):
        filas_A.append([x, y, 1, 0, 0, 0, -x * xp, -y * xp])
        filas_b.append(xp)
        filas_A.append([0, 0, 0, x, y, 1, -x * yp, -y * yp])
        filas_b.append(yp)
    A = np.array(filas_A)
    b = np.array(filas_b)

    h = resolver_sistema(A.T @ A, A.T @ b)
    if h is None:
        return None

    H_norm = np.array([[h[0], h[1], h[2]],
                       [h[3], h[4], h[5]],
                       [h[6], h[7], 1.0]])

    # Deshacer la normalizacion
    H = T_d_inv @ H_norm @ T_o
    return H / H[2, 2]


# Aplica H a puntos N x 2
def aplicar_homografia(H, puntos):
    unos = np.ones((len(puntos), 1))
    homogeneos = np.hstack([puntos, unos])
    transformados = (H @ homogeneos.T).T
    w = transformados[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return transformados[:, :2] / w


# Error en pixeles de cada pareja
def error_reproyeccion(H, origen, destino):
    estimados = aplicar_homografia(H, origen)
    return np.sqrt(np.sum((estimados - destino) ** 2, axis=1))


# RANSAC: prueba H con 4 parejas al azar y se queda con la de mas inliers
def ransac(origen, destino, iteraciones=2000, umbral=3.0, semilla=0):
    origen = np.asarray(origen, dtype=np.float64)
    destino = np.asarray(destino, dtype=np.float64)
    n = len(origen)
    if n < 4:
        print("RANSAC: se necesitan al menos 4 parejas, hay", n)
        return None, np.zeros(n, dtype=bool)

    azar = np.random.RandomState(semilla)
    mejor_inliers = np.zeros(n, dtype=bool)

    for _ in range(iteraciones):
        muestra = azar.choice(n, 4, replace=False)
        H = estimar_homografia(origen[muestra], destino[muestra])
        if H is None:
            continue

        inliers = error_reproyeccion(H, origen, destino) < umbral
        if np.sum(inliers) > np.sum(mejor_inliers):
            mejor_inliers = inliers

    if np.sum(mejor_inliers) < 4:
        print("RANSAC: no se encontro una homografia valida")
        return None, mejor_inliers

    H_final = estimar_homografia(origen[mejor_inliers], destino[mejor_inliers])
    return H_final, mejor_inliers


# Ejemplo de uso: H lleva puntos de la imagen 2 al marco de la imagen 1
if __name__ == "__main__":
    import ORB
    import Hamming

    ruta1 = "IMG/img1.jpg"
    ruta2 = "IMG/img2.jpg"

    puntos1, desc1 = ORB.orb(ruta1)
    puntos2, desc2 = ORB.orb(ruta2)
    parejas = Hamming.emparejar(desc1, desc2)
    p1, p2 = Hamming.coordenadas_parejas(puntos1, puntos2, parejas)

    H, inliers = ransac(p2, p1)

    print("Parejas:", len(parejas))
    print("Inliers:", int(np.sum(inliers)))
    print("Matriz de homografia (img2 -> img1):")
    print(np.round(H, 4))
    if H is not None:
        errores = error_reproyeccion(H, p2[inliers], p1[inliers])
        print("Error promedio de inliers (px):", round(float(np.mean(errores)), 3))

    parejas_buenas = [p for p, ok in zip(parejas, inliers) if ok]
    Hamming.dibujar_parejas(ruta1, ruta2, puntos1, puntos2, parejas_buenas,
                            salida="ransac_inliers.png")
