import numpy as np

#regresa de toda la imagen
def gradientes(img):
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
 
    # img = img.astype(np.float32)
    p = np.pad(img, 1, mode="edge")
    alto, ancho = img.shape
 
    Ix = np.zeros_like(img)
    Iy = np.zeros_like(img)
 
    for i in range(3):
        for j in range(3):
            parte = p[i:i + alto, j:j + ancho]
            Ix += sobel_x[i, j] * parte
            Iy += sobel_y[i, j] * parte
 
    return Ix, Iy

# Calcula el score de Harris para un punto (x, y) dado los gradientes Ix e Iy.
def harris_score(Ix, Iy, x, y, radio=3, k=0.04):
    alto, ancho = Ix.shape
 
    y1, y2 = max(0, y - radio), min(alto, y + radio + 1)
    x1, x2 = max(0, x - radio), min(ancho, x + radio + 1)
 
    ix = Ix[y1:y2, x1:x2]
    iy = Iy[y1:y2, x1:x2]
 
    a = np.sum(ix * ix)
    b = np.sum(ix * iy)
    c = np.sum(iy * iy)
 
    determinante = a * c - b * b
    traza = a + c
 
    return determinante - k * traza ** 2
 
 #ordena los socres de mayor a menor y regresa los mejores
def mejores_puntos(puntos_con_score, cantidad):
    ordenados = sorted(puntos_con_score, key=lambda p: p[2], reverse=True)
    return ordenados[:cantidad]
