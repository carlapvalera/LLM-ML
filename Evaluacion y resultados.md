## Evaluación del modelo

Para la evaluación del modelo se decidió utilizar tres métricas de comparación:

1. ROUGE (Suplente orientado a la recuperación para la evaluación de Gisting) es un conjunto de métricas que se utilizan para evaluar la calidad de la traducción de documentos y los modelos de resumen en el procesamiento del lenguaje natural. Mide la superposición entre un resumen o traducción generado por el sistema y un conjunto de traducciones o resúmenes de referencia creados por humanos. La puntuación de ROUGE varía de 0 a 1, y las puntuaciones más altas indican una mayor similitud entre el texto candidato y el de referencia.

2. BLEU (Suplente de evaluación bilingüe) es una métrica para evaluar la calidad del texto traducido automáticamente comparándolo con traducciones humanas de referencia. Varía de 0 a 1, y las puntuaciones más altas indican una mayor similitud entre el texto candidato y el de referencia.

3. METEOR (Métrica para la evaluación de la traducción con pedidos explícitos) es una métrica que se utiliza para evaluar la calidad del texto generado por máquina, particularmente en el contexto de la traducción automática. Está diseñado para abordar algunas de las limitaciones de la métrica BLEU más utilizada.

Se decidió comparar el modelo con hiperparámetros $\lambda=0.01,k=1$ y con sistema de recuperación de información basado en distancia coseno con respecto a un LLM normal.

Para evaluar se busco una conversación con el modelo de lenguaje donde se busca utilizarlo como entrenador físico personal que se adapte al usuario, escenario donde tener la mayor cantidad de información perdurable a traves del tiempo del usuario es crucial. Para esto se eligieron 15 pares de preguntas y respuestas con ambos modelos y luego otras 35 preguntas para su evaluación. Apéndice A contiene las preguntas realizadas tanto de la conversación en si como de las preguntas realizadas.

## Resultados:

A continuación se muestra los resultados:

## Discusión de los resultados

Se puede observar que el modelo salio considerablemente peor que el modelo de lenguaje. Sin embargo, observando los resultados concretos, se ve la explicación del suceso.

Por una parte, el modelo no parece estar lo suficientemente entrenado. Si bien las conversaciones se agrupan de forma conveniente en un principio, rápidamente podemos ver la perdida de información de los nodos resúmenes, indicando que falta entrenamiento del extractor del contexto y la necesidad de mejoría de ingeniería de prompt para resumir mejor a los nodos. Esto provocaba que mientras mas información había mas información se perdía, lo cual resultaba en que los nodos a menor profundidad tenían mas prioridad que los nodos a mayor profundidad. Sin embargo, tras la observación del árbol formado da grandes indicios de que el algoritmo tiene grandes posibilidades de mejorar sus resultados.

Por la otra parte, el modelo de lenguaje sin memoria a largo plazo, al carecer de contexto suficiente para dar una respuesta, sus alucinaciones le sirvieron para adivinar al menos parcialmente las respuestas planteadas, mientras que el modelo implementado no trataba nunca de "adivinar" la respuesta.

En conclusion, el modelo propuesto promete mucha mejoría al cambiar los hiperparámetros de su instancia probada, y da campo abierto a experimentar sus diferentes variedades.