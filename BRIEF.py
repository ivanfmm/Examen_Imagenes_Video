import numpy as np
from FAST import filtro_gauss

MITAD_PARCHE = 15   # parche de 31 x 31 píxeles alrededor del punto
MARGEN = 22         # distancia mínima al borde (15 * raíz de 2 ≈ 21.2, por la rotación)
N_BITS = 256        # tamaño del descriptor


#Aplica el filtro gaussiano a cada nivel de la pirámide. (Necesario a pesar de haber ya suavizado antes)
def suavizar_piramide(piramide, sigma=2):
    suaves = []
    for imagen_nivel in piramide:
        suaves.append(filtro_gauss(imagen_nivel, sigma))
    return suaves

def BRIEF(piramide_suave, keypoints, angulos=None):
    patron = generar_patron()
    puntos_validos = []
    descriptores = []

    for i, (x, y, nivel, score) in enumerate(keypoints):
        imagen_nivel = piramide_suave[nivel]
        alto, ancho = imagen_nivel.shape

        # Descartamos puntos muy cerca del borde (el parche se saldría)
        if x < MARGEN or y < MARGEN or x >= ancho - MARGEN or y >= alto - MARGEN:
            continue

        angulo = 0.0 if angulos is None else angulos[i]
        descriptores.append(descriptor_brief(imagen_nivel, x, y, patron, angulo))
        puntos_validos.append((x, y, nivel, score))

    return puntos_validos, np.array(descriptores)

# Parche de 31x31 alrededor del punto, con 256 parejas de píxeles (x1, y1, x2, y2) elegidas al azar.
def generar_patron(n_bits=N_BITS, mitad=MITAD_PARCHE, semilla=0):
    azar = np.random.RandomState(semilla)
    sigma = (2 * mitad + 1) / 5          # más parejas cerca del centro
    puntos = azar.normal(0, sigma, size=(n_bits, 4))
    puntos = np.round(puntos)
    puntos = np.clip(puntos, -mitad, mitad)
    return puntos.astype(int)

# Devuelve un array de 0s y 1s (uno por pareja). Bit = 1 si el primer píxel es más oscuro que el segundo.
def descriptor_brief(img, x, y, patron, angulo=0.0):
    bits = np.zeros(len(patron), dtype=np.uint8)

    for i, (x1, y1, x2, y2) in enumerate(patron):
        # Giramos la pareja según el ángulo del punto (con angulo=0 no cambia)
        dx1, dy1 = rotar(x1, y1, angulo)
        dx2, dy2 = rotar(x2, y2, angulo)

        pixel_1 = img[y + dy1, x + dx1]
        pixel_2 = img[y + dy2, x + dx2]

        if pixel_1 < pixel_2:
            bits[i] = 1

    return bits

# Rota el desplazamiento (dx, dy) alrededor del centro.
def rotar(dx, dy, angulo):
    c = np.cos(angulo)
    s = np.sin(angulo)
    nuevo_dx = int(round(c * dx - s * dy))
    nuevo_dy = int(round(s * dx + c * dy))
    return nuevo_dx, nuevo_dy


#se utiliza al comparar 2 imagenes
def distancia_hamming(d1, d2):
    return int(np.sum(d1 != d2))