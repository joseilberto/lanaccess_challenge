# Lanaccess challenge

Para el ejercicio será necesario lo siguiente:
- Descargar del dataset mscoco imagenes de personas y motos, crear sets de training, validation y test (las anotaciones deben estar en formato yolo).
- Entrenar una red de detección para estas dos categorías, puedes usar el framework de la yolov8 (https://github.com/ultralytics/ultralytics) y usar un modelo preentrenado.
- Evaluar la red y detectar los problemas que tiene (en cuanto a precisión de las detecciones).
- Optimizar la red para mejorar los problemas encontrados.
- Sacar metricas de performance y hacer deployment de la red en local.
- Crear script para ejecutar la red con un video de entrada (las cajas deben tener la información de la confianza y la clase).

Es necesario entregar todos los scripts que crees para realizar el ejercicio.
No es necesario entrenar la red muchas épocas, con que dé un rendimiento aceptable nos vale.