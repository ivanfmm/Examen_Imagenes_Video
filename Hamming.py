from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# Distancia de Hamming entre dos descriptores BRIEF (arrays de 0s y 1s).
# Cuenta en cuantas posiciones los bits son diferentes.
def distancia_hamming(d1, d2):
    distancia = 0
    for i in range(len(d1)):
        if d1[i] != d2[i]:
            distancia += 1
    return distancia


# Para cada descriptor de la imagen 1 busca el mas parecido en la imagen 2.
# Se acepta la pareja solo si el mejor es claramente mejor que el segundo
# (prueba de proporcion): mejor < ratio * segundo.
def emparejar(desc1, desc2, ratio=0.8):
    parejas = []

    for i in range(len(desc1)):
        mejor_j = -1
        mejor = np.inf
        segundo = np.inf

        for j in range(len(desc2)):
            d = distancia_hamming(desc1[i], desc2[j])
            if d < mejor:
                segundo = mejor
                mejor = d
                mejor_j = j
            elif d < segundo:
                segundo = d

        if mejor_j != -1 and mejor < ratio * segundo:
            parejas.append((i, mejor_j, mejor))

    return parejas


# Entrada para la homografia: dos arreglos N x 2 con las coordenadas (x, y)
# de cada pareja. La fila k de p1 corresponde a la fila k de p2.
def coordenadas_parejas(puntos1, puntos2, parejas):
    p1 = np.array([[puntos1[i][0], puntos1[i][1]] for i, j, d in parejas], dtype=np.float64)
    p2 = np.array([[puntos2[j][0], puntos2[j][1]] for i, j, d in parejas], dtype=np.float64)
    return p1, p2


# Pone las dos imagenes lado a lado y une con una linea cada pareja.
def dibujar_parejas(ruta1, ruta2, puntos1, puntos2, parejas, salida="hamming_parejas.png"):
    img1 = np.array(Image.open(ruta1).convert("RGB"))
    img2 = np.array(Image.open(ruta2).convert("RGB"))

    alto = max(img1.shape[0], img2.shape[0])
    ancho1 = img1.shape[1]

    lienzo = np.zeros((alto, ancho1 + img2.shape[1], 3), dtype=np.uint8)
    lienzo[:img1.shape[0], :ancho1] = img1
    lienzo[:img2.shape[0], ancho1:] = img2

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.imshow(lienzo)

    for i, j, d in parejas:
        x1, y1 = puntos1[i][0], puntos1[i][1]
        x2, y2 = puntos2[j][0] + ancho1, puntos2[j][1]
        ax.plot([x1, x2], [y1, y2], "-", color="yellow", linewidth=0.8)
        ax.plot([x1, x2], [y1, y2], ".", color="lime", markersize=4)

    ax.axis("off")
    fig.savefig(salida, bbox_inches="tight", dpi=150)
    plt.close(fig)


# Ejemplo de uso
if __name__ == "__main__":
    import ORB

    ruta1 = "IMG/img1.jpg"
    ruta2 = "IMG/img2.jpg"

    puntos1, desc1 = ORB.orb(ruta1)
    puntos2, desc2 = ORB.orb(ruta2)

    parejas = emparejar(desc1, desc2)

    print("Puntos imagen 1:", len(puntos1))
    print("Puntos imagen 2:", len(puntos2))
    print("Parejas encontradas:", len(parejas))
    if len(parejas) > 0:
        print("Distancia Hamming promedio:", sum(p[2] for p in parejas) / len(parejas))

    dibujar_parejas(ruta1, ruta2, puntos1, puntos2, parejas)

    # Esto es lo que recibe la estimacion de homografia
    p1, p2 = coordenadas_parejas(puntos1, puntos2, parejas)
    print("Coordenadas para homografia:", p1.shape, p2.shape)
