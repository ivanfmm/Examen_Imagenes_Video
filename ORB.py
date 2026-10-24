
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import FAST
import Harris
import BRIEF


def orb(image_path, umbral=20, n=9, niveles=4, escala=1.2, max_por_nivel=50, orientacion=True):
    imagen_pil = Image.open(image_path).convert("L")
    img_array = np.array(imagen_pil, dtype=np.float32)
 
    piramide = FAST.crear_piramide(img_array, niveles, escala)
 
    puntos = FAST_Harris(piramide, umbral, n, max_por_nivel)
 
    piramide_suave = BRIEF.suavizar_piramide(piramide)
    puntos, descriptores = BRIEF.BRIEF(piramide_suave, puntos, orientacion)
 
    return a_coordenadas_originales(puntos, escala), descriptores

#lista con los mejores puntos de todos los niveles, cada uno con su score y nivel de Harris.
def FAST_Harris(piramide, umbral=20, n=9, max_por_nivel=50):
    keypoints = []
 
    for nivel, imagen_nivel in enumerate(piramide):
        # 1) FAST propone candidatos
        candidatos = FAST.fast_una_imagen(imagen_nivel, umbral, n)
 
        # 2) Harris puntúa a cada candidato
        Ix, Iy = Harris.gradientes(imagen_nivel)
        con_score = []
        for x, y in candidatos:
            score = Harris.harris_score(Ix, Iy, x, y)
            con_score.append((x, y, score))
 
        # 3) Nos quedamos con los mejores de este nivel
        mejores = Harris.mejores_puntos(con_score, max_por_nivel)
 
        # 4) Guardamos los puntos (siguen en la escala de su nivel)
        for x, y, score in mejores:
            keypoints.append((x, y, nivel, score))
 
    return keypoints

def a_coordenadas_originales(puntos, escala=1.2):
    originales = []
    for x, y, nivel, score in puntos:
        factor = escala ** nivel
        originales.append((int(x * factor), int(y * factor), nivel, score))
    return originales


def dibujar_puntos(image_path, puntos, salida="orb_resultado.png", escala=1.2, radio_base=8):
    
    COLORES = ["red", "lime", "cyan", "yellow", "magenta", "orange"]
    imagen = Image.open(image_path).convert("RGB")
 
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(imagen)
 
    for x, y, nivel, score in puntos:
        color = COLORES[nivel % len(COLORES)]
        radio = radio_base * escala ** nivel
 
        ax.add_patch(Circle((x, y), radio, fill=False, color=color, linewidth=1.5))
        ax.plot(x, y, ".", color=color, markersize=3)
 
    ax.axis("off")
    fig.savefig(salida, bbox_inches="tight", dpi=150)
    plt.close(fig)


#Ejemplo de uso 
if __name__ == "__main__":
    image_path = "IMG/sol.jpg"
    puntos, descriptores = orb(image_path)

    dibujar_puntos(image_path, puntos, salida="orb_resultado_sol.png")