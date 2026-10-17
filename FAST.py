import numpy as np
import matplotlib.pyplot as plt


# Parte 1 Fast
def crear_piramide(img, niveles=4, escala=1.2):
    piramide = [img.astype(np.float32)]
    for nivel in range(1, niveles):
        piramide.append(reducir_img(img, escala ** nivel))
    return piramide

#Reduce la imagen dividiendo su tamaño entre factor
def reducir_img(img, factor):
    # Suaviza la imagen antes de reducirla
    suave = filtro_gauss(img, sigma=0.5 * factor)
 
    alto = int(img.shape[0] / factor)
    ancho = int(img.shape[1] / factor)
 
    filas = (np.arange(alto) * factor).astype(int)
    cols = (np.arange(ancho) * factor).astype(int)
    return suave[np.ix_(filas, cols)]

def filtro_gauss(img, sigma):
    k = kernel_gauss(sigma)
    r = len(k) // 2
    img = img.astype(np.float32)
 
    p = np.pad(img, r, mode="edge")
 
    # Filtro horizontal
    h = np.zeros_like(img)
    for i, peso in enumerate(k):
        h += peso * p[r:r + img.shape[0], i:i + img.shape[1]]
 
    # Filtro vertical
    p = np.pad(h, r, mode="edge")
    salida = np.zeros_like(img)
    for i, peso in enumerate(k):
        salida += peso * p[i:i + img.shape[0], r:r + img.shape[1]]
 
    return salida

def kernel_gauss(sigma):
    radio = int(3 * sigma)
    x = np.arange(-radio, radio + 1)
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


#Parte 2 Fast (la piramide ya está creada)
# Encuentra esquinas en una imagen usando el detector FAST
def fast_una_imagen(img, umbral=20, n=9):
    circulo = crear_circulo()
    alto, ancho = img.shape
    esquinas = []
 
    for y in range(3, alto - 3):
        for x in range(3, ancho - 3):
            centro = img[y, x]
 
            # +1 si el vecino es más brillante, -1 si es más oscuro, 0 si es parecido
            estados = []
            for dx, dy in circulo:
                vecino = img[y + dy, x + dx]
                if vecino > centro + umbral:
                    estados.append(1)
                elif vecino < centro - umbral:
                    estados.append(-1)
                else:
                    estados.append(0)
 
            if hay_arco(estados, n):
                esquinas.append((x, y))
 
    return esquinas

def crear_circulo():
    return [
        (0, -3), (1, -3), (2, -2), (3, -1),
        (3, 0), (3, 1), (2, 2), (1, 3),
        (0, 3), (-1, 3), (-2, 2), (-3, 1),
        (-3, 0), (-3, -1), (-2, -2), (-1, -3),
    ]


def hay_arco(estados, n):
    estados = estados + estados[:n - 1]
    for signo in (1, -1):
        seguidos = 0
        for e in estados:
            seguidos = seguidos + 1 if e == signo else 0
            if seguidos >= n:
                return True
    return False







