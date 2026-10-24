import os
import sys
import glob
import numpy as np
from PIL import Image, ImageDraw

from Panorama import interpolar


TOLERANCIA = 15        # diferencia minima contra el color de fondo
ANCHO_BUSQUEDA = 600   # ancho reducido para buscar el rectangulo (mas rapido)
ALTURA_MIN = 0.5       # columnas con menos de esta fraccion de la altura maxima se descartan
ANCHO_COLLAGE = 2000


# Paso 1: mascara de pixeles validos
# El fondo es el color de las esquinas (negro o blanco segun la imagen)
def mascara_valida(img):
    esquinas = np.array([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]], dtype=np.float32)
    fondo = np.median(esquinas, axis=0)
    diferencia = np.abs(img.astype(np.float32) - fondo).max(axis=2)
    return rellenar_huecos(diferencia > TOLERANCIA)


# Pixeles oscuros dentro de la foto (noche, sombras) quedan como "fondo".
# Solucion simple: por columna y por fila, todo lo que esta entre el primer
# y el ultimo pixel valido cuenta como valido. Se cruzan ambas mascaras.
def entre_extremos(mascara):
    hay_algo = mascara.any(axis=0)
    primero = np.argmax(mascara, axis=0)
    ultimo = mascara.shape[0] - 1 - np.argmax(mascara[::-1], axis=0)
    filas = np.arange(mascara.shape[0])[:, None]
    return (filas >= primero) & (filas <= ultimo) & hay_algo


def rellenar_huecos(mascara):
    por_columna = entre_extremos(mascara)
    por_fila = entre_extremos(mascara.T).T
    return por_columna & por_fila


# Rectangulo mas grande lleno de unos (algoritmo del histograma con pila)
# Regresa x, y, ancho, alto
def mayor_rectangulo(mascara):
    alto, ancho = mascara.shape
    alturas = np.zeros(ancho, dtype=int)
    mejor = (0, 0, 0, 0, 0)  # area, x, y, ancho, alto

    for fila in range(alto):
        # Cuantos unos seguidos hay hacia arriba en cada columna
        alturas = np.where(mascara[fila], alturas + 1, 0)
        pila = []  # (columna de inicio, altura)
        for x in range(ancho + 1):
            h = alturas[x] if x < ancho else 0
            inicio = x
            while pila and pila[-1][1] >= h:
                inicio, h_pila = pila.pop()
                area = h_pila * (x - inicio)
                if area > mejor[0]:
                    mejor = (area, inicio, fila - h_pila + 1, x - inicio, h_pila)
            pila.append((inicio, h))
    return mejor[1:]


# Opcion A: recortar el rectangulo mas grande (no deforma, pierde bordes)
def recortar(img, mascara):
    alto, ancho = mascara.shape
    escala = ancho / ANCHO_BUSQUEDA
    alto_chico = max(1, int(alto / escala))

    # Reducir la mascara; solo cuenta como valido si todo el bloque era valido
    chica = Image.fromarray((mascara * 255).astype(np.uint8))
    chica = np.array(chica.resize((ANCHO_BUSQUEDA, alto_chico), Image.BILINEAR)) > 250

    x, y, w, h = mayor_rectangulo(chica)

    # Regresar a la escala original con un margen de 2 px
    x0 = int(np.ceil(x * escala)) + 2
    y0 = int(np.ceil(y * escala)) + 2
    x1 = int((x + w) * escala) - 2
    y1 = int((y + h) * escala) - 2
    return img[y0:y1, x0:x1]


# Suaviza una curva: mediana (quita picos) y luego promedio (quita escalones)
def suavizar(valores, ventana=41):
    relleno = np.pad(valores.astype(np.float64), ventana // 2, mode="edge")
    mediana = np.median(np.lib.stride_tricks.sliding_window_view(relleno, ventana), axis=1)
    relleno = np.pad(mediana, ventana // 2, mode="edge")
    return np.convolve(relleno, np.ones(ventana) / ventana, mode="valid")


# Opcion B: estirar cada columna para llenar todo el alto (no pierde contenido, deforma)
def estirar(img, mascara):
    alto = mascara.shape[0]
    hay_algo = mascara.any(axis=0)
    arriba = np.argmax(mascara, axis=0)
    abajo = alto - 1 - np.argmax(mascara[::-1], axis=0)
    altura = np.where(hay_algo, abajo - arriba, 0)

    # Quedarse con el tramo de columnas suficientemente altas
    buenas = np.where(altura >= ALTURA_MIN * altura.max())[0]
    c0, c1 = buenas[0], buenas[-1] + 1

    # Suavizar los bordes para que las lineas no salgan onduladas
    ventana = max(5, (c1 - c0) // 40) | 1  # impar
    arriba = suavizar(arriba[c0:c1], ventana)
    abajo = suavizar(abajo[c0:c1], ventana)
    alto_salida = int(np.median(abajo - arriba))

    # Mapeo inverso: cada pixel de salida sabe de donde leer en la original
    t = np.linspace(0, 1, alto_salida)[:, None]
    ys = arriba[None, :] + t * (abajo - arriba)[None, :]
    xs = np.broadcast_to(np.arange(c0, c1)[None, :], ys.shape).astype(np.float64)

    colores, _ = interpolar(img.astype(np.float32), xs, ys, "bilineal")
    return np.clip(colores, 0, 255).astype(np.uint8)


# Paso final: juntar todos los resultados en una sola imagen (uno debajo del otro)
def collage(imagenes, nombres, separacion=20):
    filas = []
    for img, nombre in zip(imagenes, nombres):
        alto = int(img.shape[0] * ANCHO_COLLAGE / img.shape[1])
        fila = Image.fromarray(img).resize((ANCHO_COLLAGE, alto), Image.BILINEAR)
        ImageDraw.Draw(fila).text((15, 15), nombre, fill=(255, 255, 0))
        filas.append(fila)

    alto_total = sum(f.height for f in filas) + separacion * (len(filas) - 1)
    lienzo = Image.new("RGB", (ANCHO_COLLAGE, alto_total), (255, 255, 255))
    y = 0
    for fila in filas:
        lienzo.paste(fila, (0, y))
        y += fila.height + separacion
    return lienzo


# Igual que collage pero uno al lado del otro, todos con la misma altura
def collage_horizontal(imagenes, nombres, alto=800, separacion=20):
    columnas = []
    for img, nombre in zip(imagenes, nombres):
        ancho = int(img.shape[1] * alto / img.shape[0])
        col = Image.fromarray(img).resize((ancho, alto), Image.BILINEAR)
        ImageDraw.Draw(col).text((15, 15), nombre, fill=(255, 255, 0))
        columnas.append(col)

    ancho_total = sum(c.width for c in columnas) + separacion * (len(columnas) - 1)
    lienzo = Image.new("RGB", (ancho_total, alto), (255, 255, 255))
    x = 0
    for col in columnas:
        lienzo.paste(col, (x, 0))
        x += col.width + separacion
    return lienzo


# python Rectificar.py                       -> Resultados_Completos/* y resultados/panorama.jpg
# python Rectificar.py a.jpg b.jpg ...      -> solo esas
if __name__ == "__main__":
    if len(sys.argv) > 1:
        rutas = sys.argv[1:]
    else:
        rutas = sorted(glob.glob(os.path.join("Resultados_Completos", "*.jpg")))
        if os.path.exists(os.path.join("resultados", "panorama.jpg")):
            rutas.append(os.path.join("resultados", "panorama.jpg"))

    carpeta = "Resultados_Rectificados"
    os.makedirs(carpeta, exist_ok=True)

    recortes, estiradas, nombres = [], [], []
    for ruta in rutas:
        nombre = os.path.splitext(os.path.basename(ruta))[0]
        img = np.array(Image.open(ruta).convert("RGB"))
        mascara = mascara_valida(img)

        rec = recortar(img, mascara)
        est = estirar(img, mascara)
        Image.fromarray(rec).save(os.path.join(carpeta, nombre + "_recorte.jpg"), quality=95)
        Image.fromarray(est).save(os.path.join(carpeta, nombre + "_estirada.jpg"), quality=95)
        print("%s: original %s  recorte %s  estirada %s" % (nombre, img.shape[:2], rec.shape[:2], est.shape[:2]))

        recortes.append(rec)
        estiradas.append(est)
        nombres.append(nombre)

    collage(recortes, nombres).save(os.path.join(carpeta, "collage_recortes.jpg"), quality=90)
    collage(estiradas, nombres).save(os.path.join(carpeta, "collage_estiradas.jpg"), quality=90)
    collage_horizontal(recortes, nombres).save(os.path.join(carpeta, "tira_recortes.jpg"), quality=90)
    collage_horizontal(recortes, nombres, separacion=0).save(os.path.join(carpeta, "tira_recortes_pegada.jpg"), quality=90)
    print("Listo:", carpeta)
