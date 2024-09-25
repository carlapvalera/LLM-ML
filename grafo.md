Una forma comúnmente de proporcionar una memoria a largo plazo en los Modelos de Lenguaje Grande (LLMs) que supere las limitaciones del contexto temporal es la representación gráfica en lugar de vectorial. Este enfoque busca facilitar el razonamiento sostenido, el aprendizaje acumulativo y la interacción prolongada con los usuarios, aspectos que son esenciales para avanzar hacia sistemas de Inteligencia Artificial General (AGI). 

Sus principales puntos fuertes son:
- La extracción de conceptos clave y sus relaciones contextuales.
- El acceso eficiente y una actualización dinámica de la información.
- La relevancia temporal de los conceptos almacenados.
- El razonamiento y la generación de nuevo conocimiento

Sus principales aplicaciones son:
-Interacción prolongada con usuarios: Permite a los LLMs recordar información proporcionada por los usuarios a lo largo del tiempo, mejorando la personalización y relevancia en las respuestas.
-Razonamiento sostenido: La capacidad para actualizar creencias y mantener un entendimiento temporal facilita tareas complejas que requieren razonamiento continuo.
-Integración con bases de datos vectoriales: En las versiones híbrida, combina sus capacidades con bases de datos vectoriales para aprovechar lo mejor de ambos mundos, optimizando tanto la comprensión temporal como la recuperación general de información.

Entre los trabajos que se basan en la estructura de grafos esta:
- RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models: Su objetivo es desarrollar una arquitectura de memoria que permita a los LLMs recordar y recuperar información de manera más eficaz, imitando el proceso humano de recuerdo de memoria, lo que es esencial para mantener conversaciones coherentes y contextualmente relevantes.
- Graph Memory-based Editing for Large Language Models: El objetivo principal es desarrollar un enfoque que permita a los LLMs manejar información de manera más eficiente y efectiva utilizando estructuras de grafos como memoria externa. Esto es especialmente relevante para tareas que implican razonamiento complejo y recuperación de datos interconectados. 
- AriGraph: Learning Knowledge Graph World Models with Episodic Memory for LLM Agents: su objetivo principal es desarrollar un modelo que combine grafos de conocimiento y memoria episódica para permitir a los LLMs aprender y razonar sobre el mundo de manera más efectiva, facilitando la comprensión de contextos complejos y la ejecución de tareas que requieren múltiples pasos de razonamiento.


Este enfoque, aunque presenta innovaciones significativas en el ámbito de los modelos de lenguaje, también tiene varias desventajas que pueden limitar su efectividad y aplicabilidad. Las principales desventajas del modelo:
- Su complejidad computacional: El proceso de revisión del contexto es uno de los pasos más costosos computacionalmente dentro de sus mecanismos. Este proceso es necesario para mantener la relevancia y actualizar la información almacenada, lo que puede resultar en un impacto significativo en el rendimiento general del sistema, especialmente en aplicaciones que requieren respuestas rápidas.
- Dependen en gran medida de la calidad de la información proporcionada por los usuarios. Si los datos iniciales son incorrectos o imprecisos, esto puede llevar a una acumulación de errores en la memoria a largo plazo, afectando negativamente la precisión y confiabilidad de dichos modelos.
- Dificultades con el Olvido Catastrófico: Diferentes implementaciones tratan de resolver precisamente este problema, pero la actualización constante y la eliminación de información obsoleta pueden no ser suficientes para evitar que el modelo olvide información importante a medida que se añaden nuevos datos. Esto es un desafío común en cualquier sistema que utilice memoria dinámica.
- A pesar de su enfoque en la modelización de relaciones complejas a través de una base de datos en base a grafos, estas estructuras puede enfrentar dificultades al tratar con relaciones altamente interdependientes o contextos muy complejos. Esto podría limitar su capacidad para razonar sobre situaciones que requieren un entendimiento profundo y multifacético.
- La implementación y operación de este tipo de modelos, incluso mas en formas híbridas con bases de datos vectoriales, pueden requerir más recursos computacionales y almacenamiento en comparación con modelos más simples. Esto podría ser un obstáculo para su adopción en entornos con recursos limitados.

Debido a la alta complejidad y las grandes necesidades de recursos computacionales de estos modelos y en gran medida a su no trivial implementación, este tipo de estructura ha caído en desuso frente al modelo vectorial, el cual sigue siendo el mas profundizado entre la comunidad científica, relegando el uso de estructuras basadas en grafos a modelos híbridos.
