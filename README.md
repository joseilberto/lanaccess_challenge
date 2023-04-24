# Lanaccess challenge

Para el ejercicio será necesario lo siguiente:

- Descargar del dataset mscoco imagenes de personas y motos, crear sets de training, validation y test (las anotaciones deben estar en formato yolo).
- Entrenar una red de detección para estas dos categorías, puedes usar el framework de la yolov8 (<https://github.com/ultralytics/ultralytics>) y usar un modelo preentrenado.
- Evaluar la red y detectar los problemas que tiene (en cuanto a precisión de las detecciones).
- Optimizar la red para mejorar los problemas encontrados.
- Sacar metricas de performance y hacer deployment de la red en local.
- Crear script para ejecutar la red con un video de entrada (las cajas deben tener la información de la confianza y la clase).

Es necesario entregar todos los scripts que crees para realizar el ejercicio.
No es necesario entrenar la red muchas épocas, con que dé un rendimiento aceptable nos vale.

## Code formatting, despliegue y versionamiento de datos y modelos

- Code formatting: Para formatear el código utilizando estándares de código a nivel de producción en Python, utilizamos aquí tres herramientas distintas:
  - [black](https://black.readthedocs.io/en/stable/): Se utiliza black con sus opciones por defecto y con el tamañno de la línea a 79 caracteres. El código se formatea automáticamente y nos permite mantener la estética del código ordenada en todos los archivos siguiendo el formato de PEP8 por defecto.
  - [pylint](https://www.pylint.org/): Pylint es un linter que hace respetar los estándares del PEP8 para el código, comprueba los imports y su órden en el código y nos permite, detecta posibles errores que se verían durante el runtime, contribuye para la refactoración del código reduciendo el riesgo de adoptar posibles malas prácticas.
  - [mypy](https://mypy-lang.org/): Es un optional static type checker para Python y nos permite comprobar los tipos de variables utilizados, reduciendo el riesgo de que funciones reciban y devuelvan parámetros de distintos tipos.

- Despliegue: Adoptamos [Docker](https://www.docker.com/) para la creación de contenedores que simplifiquen el uso del código y las dependencias que tenemos para el proyecto. Se enviará las imágenes de un video a la red neuronal entrenada que está en un contenedor Docker para que haga las detecciones y nos devuelva una imagen etiquetada con las categorías y confianza respectivas. Durante la creación del contenedor, leeremos el modelo del repo dvc.

- Versionamiento:
  - Código: Utilizamos git.
  - De datos y modelos: Utilizamos dvc configurado para google drive.

## Dataset MSCOCO con dos clases

Se ha descargado el MSCOCO con las figuras pertenecientes a dos categorias: personas y motocicletas. Para descargar exclusivamente esas categorias se ha utilizado la API de [fiftyone.zoo](<https://docs.voxel51.com/>)[]. He limitado el número de imágenes descargadas a 15000 en total. Una vez descargadas las imágenes, convertimos su formato de MSCOCO estándar a YOLO (con un archivo .txt con las etiquetas de cada una de las imágenes). La estructura de carpeta creadas nos permite utilizar los modelos de ultralytics a través de un archivo de configuración yaml. Vemos que hay un total de 15000 imágenes para el entrenamiento de la red neuronal y 2724 imágenes para en el dataset de validación.

Para el conjunto descargado, se nota un importante desbalance entre las clases de personas con más de 60000 instancias y motocicletas con menos de 10000 (resultados disponibles en labels.jpg en la carpeta Pretrained YOLOv8N two classes). También vemos que hay una distribución de objetos detectados por toda el area de la foto, con más probabilidad de que los objetos que buscamos estén cerca del centro de la imagen. También notamos que las cajas tienen una distribución de tamaños muy amplias, pero en general son más pequeñas que el 10% del tamaño total de la imagen.

## Entrenando la red neuronal

Utitilizamos los modelos preentrenados de YOLOv8N de [ultralytics](https://docs.ultralytics.com/modes/). Tenemos en cuenta que el modelo preentrenado ya posee un buen ajuste para detectar las clases de interés, pero lo refinaremos para la detección de las dos categorías del presente trabajo. La propuesta es entrentar el modelo durante 10 épocas y evaluar las siguientes métricas generadas por ultralytics:

- Confianza: mide la probabilidad de que el modelo asigne un objeto detectado a una clase
- Precisión: mide la probabilidad de que una detección sea correcta
- Exhaustividad: mide la probabilidad de que una detección correcta sea obtenida cuando buscamos por una clase en una imagen
- F1 score: mide la média armónica de la precisión y la exhaustividad. Nos permite determinar cuántos falso positivos medimos de una clase.

## Resultados y discusiones

Vemos que los resultados indican que todas las métricas: Precisión, Exhaustividad y F1 score son superiores al modelo naive para las imágenes que contienen personas. El mismo patrón no se observa para las imágenes conteniendo motos. Es muy probable que la causa principal sea el gran desbalance entre las clases en el dataset.

Para mejorar este resultado, tendríamos que hacer un mejor muestreo de las imágenes (probablemente reduciendo el número de imágenes de personas). Una otra alternativa sería descargar el dataset completo de MSCOCO para esas dos categorías y entonces hacer un muestreo. Teniendo en cuanta que hacer un oversampling en datasets de detección o segmentación [suele tener implicaciones en la distribución espacial de los objetos](https://arxiv.org/abs/1106.1813), la mejor estrategia sería realizar un undersampling para que las clases tuviesen aproximadamente el mismo número de instancias.

Una alternativa sería agregar algún modelo de regularización o cambiar la función de pérdida favoreciendo las detecciones de motocicletas en las imágenes. Cualquiera de esas soluciones, requiere el estudio debido del impacto de los cambios en los resultados. Debido a las limitaciones de tiempo del proyecto, no se ha realizado ningún cambio en la red neuronal y en el dataset con el objetivo de mejorar las detecciones y los resultados observados en las métricas obtenidas utilizando ultralytics.

## Despliegue y detecciones en videos y imágenes

El Dockerfile originalmente se propone copiar la dirección que contiene el archivo a ser etiquetado y la imagen docker posee un entrypoint para ejecutar la detección del archivo directamente. Por ejemplo, se puede ejecutar la imagen docker de la siguiente manera:

```console
docker run -v ./data/samples:/build/data/samples --gpus all --rm lanaccess --p="./data/samples/sample1.jpg"
```

El script `detect_objects.py` se encarga de etiquetar el archivo, copiar la imagen etiquetada al volumen compartido y ese se hace disponible localmente. Es importante comentar que en los tests locales, los archivos .mp4 no han funcionado para generar archivos etiquetados con la imagen Docker. Es probable que haya una incompatibilidad en el codec de video de openCV que guarda el video. Otra posibilidad es que al compartir un volumen entre el contenedor y el disco local, [se produzcan errores](https://forums.docker.com/t/volume-vs-mounting-a-host-folder-in-regards-to-file-corruption/14032). Sin embargo, siguiendo los pasos de la próxima sección, podemos etiquetar videos localmente.

## Instalación de requisitos y ejecución local

Localmente, he utilizado [conda](https://www.anaconda.com/download/) para la gestión de entornos. El primer paso es crear un entorno con las dependencias de cuda y las dependencias de python con poetry:

```console
conda create -n lanaccess python=3.8 cudatoolkit=11.8.0 -c conda-forge
conda activate lanacess
poetry install
```

Una vez instaladas las dependencias necesarias del proyecto, podemos ejecutar el script para descargar y entrenar la red neuronal:

```console
python train_yolo.py
```

Por fin, podemos ejecutar el script que etiqueta imágenes y videos:

```console
python detect_objects.py -p "./data/samples/video_sample3.mp4"
```

Al final de este proceso, obtenemos un video con cada frame etiquetado correctamente.

## Perspectivas y mejoras del proyecto

- Agregar más argumentos a los scripts o crear yaml files para configurar rutas de modelos y datos
- Utilizar el contenedor docker como un contenedor realmente y no como un entrypoint que depende un volumen compartido.
- Entrenar la red neuronal con un dataset balanceado
- Entrenar la red neuronal con más épocas y cambiando la learning rate.
