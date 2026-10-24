import os
import sys
import numpy as np
from PIL import Image, ImageOps

import ORB
import Hamming
import Homografia


ANCHO_MAX = 1000
LIENZO_MAX = 8000


# Paso 1: endereza la foto (EXIF) y la reduce
def preparar_imagen(ruta, carpeta):
    img = Image.open(ruta)
    img = ImageOps.exif_transpose(img).convert("RGB")

    if img.width > ANCHO_MAX:
        alto = int(img.height * ANCHO_MAX / img.width)
        img = img.resize((ANCHO_MAX, alto), Image.BILINEAR)

    nombre = os.path.splitext(os.path.basename(ruta))[0] + ".jpg"
    salida = os.path.join(carpeta, "prep_" + nombre)
    img.save(salida, quality=95)
    return salida


# Inversa 3x3 con la adjunta
def invertir_3x3(M):
    a, b, c = M[0]
    d, e, f = M[1]
    g, h, i = M[2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        raise ValueError("La matriz no tiene inversa")
    adj = np.array([[e * i - f * h, c * h - b * i, b * f - c * e],
                    [f * g - d * i, a * i - c * g, c * d - a * f],
                    [d * h - e * g, b * g - a * h, a * e - b * d]])
    return adj / det


def traslacion(dx, dy):
    return np.array([[1, 0, dx],
                     [0, 1, dy],
                     [0, 0, 1]], dtype=np.float64)


# Tamano del lienzo a partir de las esquinas transformadas
def calcular_lienzo(imagenes, homografias):
    todas = []
    for img, H in zip(imagenes, homografias):
        alto, ancho = img.shape[:2]
        esquinas = np.array([[0, 0], [ancho - 1, 0],
                             [ancho - 1, alto - 1], [0, alto - 1]], dtype=np.float64)
        todas.append(Homografia.aplicar_homografia(H, esquinas))
    todas = np.vstack(todas)

    x_min, y_min = np.floor(todas.min(axis=0))
    x_max, y_max = np.ceil(todas.max(axis=0))
    ancho = int(x_max - x_min) + 1
    alto = int(y_max - y_min) + 1

    if ancho > LIENZO_MAX or alto > LIENZO_MAX:
        raise ValueError("Lienzo de %d x %d: alguna homografia salio mal "
                         "(revisa los inliers)" % (ancho, alto))

    return traslacion(-x_min, -y_min), ancho, alto


# Paso 7: interpolacion (vecino o bilineal)
def interpolar(img, xs, ys, metodo="bilineal"):
    alto, ancho = img.shape[:2]
    dentro = (xs >= 0) & (xs <= ancho - 1) & (ys >= 0) & (ys <= alto - 1)

    if metodo == "vecino":
        xi = np.clip(np.round(xs).astype(int), 0, ancho - 1)
        yi = np.clip(np.round(ys).astype(int), 0, alto - 1)
        return img[yi, xi], dentro

    x0 = np.clip(np.floor(xs).astype(int), 0, ancho - 1)
    y0 = np.clip(np.floor(ys).astype(int), 0, alto - 1)
    x1 = np.clip(x0 + 1, 0, ancho - 1)
    y1 = np.clip(y0 + 1, 0, alto - 1)

    wx = (xs - x0)[..., None]
    wy = (ys - y0)[..., None]

    arriba = (1 - wx) * img[y0, x0] + wx * img[y0, x1]
    abajo = (1 - wx) * img[y1, x0] + wx * img[y1, x1]
    return (1 - wy) * arriba + wy * abajo, dentro


# Paso 5: mapeo inverso con H inversa
def transformar_imagen(img, H, ancho, alto, metodo="bilineal"):
    H_inv = invertir_3x3(H)

    ys, xs = np.mgrid[0:alto, 0:ancho]
    puntos = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    origen = Homografia.aplicar_homografia(H_inv, puntos)
    xs_o = origen[:, 0].reshape(alto, ancho)
    ys_o = origen[:, 1].reshape(alto, ancho)

    colores, dentro = interpolar(img.astype(np.float32), xs_o, ys_o, metodo)
    colores[~dentro] = 0

    # Peso para la union: distancia al borde
    h, w = img.shape[:2]
    peso = np.minimum(np.minimum(xs_o, w - 1 - xs_o), np.minimum(ys_o, h - 1 - ys_o)) + 1
    peso[~dentro] = 0
    return colores, peso.astype(np.float32)


# Paso 6: union con promedio ponderado
def unir(transformadas, pesos):
    suma = np.zeros_like(transformadas[0], dtype=np.float32)
    total = np.zeros(pesos[0].shape, dtype=np.float32)
    for img, peso in zip(transformadas, pesos):
        suma += img * peso[..., None]
        total += peso
    total[total == 0] = 1
    return np.clip(suma / total[..., None], 0, 255).astype(np.uint8)


# Imagenes de izquierda a derecha, la del centro es la referencia
# Sin orientacion porque las fotos del panorama no rotan
def panorama(rutas, carpeta="resultados", metodo="bilineal", max_por_nivel=300,
             niveles=8, orientacion=False):
    os.makedirs(carpeta, exist_ok=True)
    n = len(rutas)
    if n < 2:
        raise ValueError("Se necesitan al menos 2 imagenes")

    # Preparacion
    rutas = [preparar_imagen(r, carpeta) for r in rutas]
    imagenes = [np.array(Image.open(r)) for r in rutas]

    # ORB
    puntos, descriptores = [], []
    for k, r in enumerate(rutas):
        p, d = ORB.orb(r, niveles=niveles, max_por_nivel=max_por_nivel, orientacion=orientacion)
        puntos.append(p)
        descriptores.append(d)
        ORB.dibujar_puntos(r, p, salida=os.path.join(carpeta, "puntos_%d.png" % (k + 1)))
        print("Imagen %d: %d puntos" % (k + 1, len(p)))

    # Asociacion y homografia entre vecinas (k+1 -> k)
    H_vecinas = []
    reporte = []
    for k in range(n - 1):
        parejas = Hamming.emparejar(descriptores[k], descriptores[k + 1])
        p_a, p_b = Hamming.coordenadas_parejas(puntos[k], puntos[k + 1], parejas)
        Hamming.dibujar_parejas(rutas[k], rutas[k + 1], puntos[k], puntos[k + 1], parejas,
                                salida=os.path.join(carpeta, "parejas_%d_%d.png" % (k + 1, k + 2)))

        H, inliers = Homografia.ransac(p_b, p_a)
        if H is None:
            raise ValueError("No se pudo unir la imagen %d con la %d" % (k + 1, k + 2))

        buenas = [p for p, ok in zip(parejas, inliers) if ok]
        Hamming.dibujar_parejas(rutas[k], rutas[k + 1], puntos[k], puntos[k + 1], buenas,
                                salida=os.path.join(carpeta, "inliers_%d_%d.png" % (k + 1, k + 2)))

        error = np.mean(Homografia.error_reproyeccion(H, p_b[inliers], p_a[inliers]))
        linea = "Imagenes %d-%d: parejas=%d inliers=%d error=%.2f px" % (
            k + 1, k + 2, len(parejas), int(np.sum(inliers)), error)
        print(linea)
        if np.sum(inliers) < 15:
            print("  OJO: pocos inliers, la union puede salir mal")
        reporte.append(linea + "\nH (%d -> %d):\n%s\n" % (k + 2, k + 1, np.array2string(H, precision=5)))
        H_vecinas.append(H)

    # Encadenar hacia la referencia
    ref = n // 2
    H_ref = [None] * n
    H_ref[ref] = np.eye(3)
    for k in range(ref + 1, n):
        H_ref[k] = H_ref[k - 1] @ H_vecinas[k - 1]
    for k in range(ref - 1, -1, -1):
        H_ref[k] = H_ref[k + 1] @ invertir_3x3(H_vecinas[k])

    # Transformacion e interpolacion
    T, ancho, alto = calcular_lienzo(imagenes, H_ref)
    print("Lienzo: %d x %d" % (ancho, alto))
    transformadas, pesos = [], []
    for k in range(n):
        H_final = T @ H_ref[k]
        img_t, peso = transformar_imagen(imagenes[k], H_final, ancho, alto, metodo)
        transformadas.append(img_t)
        pesos.append(peso)
        Image.fromarray(img_t.astype(np.uint8)).save(
            os.path.join(carpeta, "transformada_%d.jpg" % (k + 1)))
        reporte.append("H final imagen %d -> lienzo:\n%s\n" % (k + 1, np.array2string(H_final, precision=5)))

    # Union
    resultado = unir(transformadas, pesos)
    salida = os.path.join(carpeta, "panorama.jpg")
    Image.fromarray(resultado).save(salida, quality=95)

    with open(os.path.join(carpeta, "matrices.txt"), "w") as f:
        f.write("\n".join(reporte))

    print("Listo:", salida)
    return resultado


# python Panorama.py IMG/1.jpg IMG/2.jpg IMG/3.jpg
if __name__ == "__main__":
    if len(sys.argv) > 1:
        rutas = sys.argv[1:]
    else:
        extensiones = (".jpg", ".jpeg", ".png")
        rutas = sorted(os.path.join("IMG", f) for f in os.listdir("IMG")
                       if f.lower().endswith(extensiones))
    print("Imagenes:", rutas)
    panorama(rutas)
