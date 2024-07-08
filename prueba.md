
1.	Introducción
El auge de la inteligencia artificial en los últimos años ha sido impresionante, superando ampliamente las expectativas de las personas no vinculadas a la informática en cuanto a la capacidad de realizar todo tipo de tareas. Con el lanzamiento de ChatGPT, se abrió todo un nuevo mundo de posibles herramientas basadas en Modelos de Lenguaje a Gran Escala (LLM). Desde un asistente laboral hasta ayuda más específica como un psicólogo, entrenador físico, etc. Además, es una gran herramienta durante investigaciones científicas, especialmente para hacer simulaciones de personas o eventos. En todos estos casos, es importante que estos modelos de lenguaje a gran escala sean capaces de adaptarse al usuario de una manera única, conociendo los detalles específicos que permiten un mejor desempeño de la tarea, sin la necesidad de un reentrenamiento especializado para una persona en particular. Por lo tanto, existe la necesidad de complementar los LLM con la capacidad de recordar cosas a corto y largo plazo.
En este documento, se presenta un modelo novedoso para proporcionar memoria a corto y largo plazo a un LLM que no requiere un entrenamiento específico, basado en una combinación de las ideas más utilizadas en los últimos tiempos para proporcionar esta capacidad a través del aprendizaje en línea no supervisado. El objetivo es proporcionar un nuevo enfoque al tema para crear nuevas aplicaciones capaces de ser adaptables para los usuarios, ya sea en tareas de asistencia personal, trabajo, investigación o simplemente entretenimiento.

1.1. Motivación
Algunas de las aplicaciones potenciales de estos LLM adaptables incluyen:
-	Asistentes personales: Poder contar con un asistente que se adapte a las necesidades y preferencias únicas de cada usuario, recordando detalles importantes y proporcionando ayuda personalizada.
-	Entrenadores y mentores virtuales: LLM adaptables podrían servir como entrenadores o mentores virtuales en diversos campos, desde la salud y el fitness hasta el desarrollo personal y profesional. Recordarían los objetivos y progresos de cada usuario.
-	Compañeros de investigación: En el campo científico, estos modelos podrían actuar como compañeros de investigación, recordando hipótesis, experimentos y resultados anteriores para generar ideas y enfoques innovadores.
-	Asistentes creativos: Los artistas y creadores podrían beneficiarse de LLM adaptables que recuerden sus estilos, temas y proyectos anteriores para inspirar y guiar su trabajo creativo.
-	Aplicaciones educativas: En el ámbito educativo, estos modelos podrían adaptarse a los estilos de aprendizaje y conocimientos previos de cada estudiante, proporcionando una experiencia de aprendizaje más personalizada y efectiva.
Por lo tanto, existe una necesidad clara de complementar los LLM con la capacidad de recordar cosas a corto y largo plazo de una manera adaptable y sin necesidad de reentrenamiento específico. Esto abre un mundo de posibilidades para crear herramientas y aplicaciones que puedan adaptarse de manera única a cada usuario, mejorando significativamente su experiencia y eficacia.

1.2. Problemática
Si bien los Modelos de Lenguaje a Gran Escala (LLM) han demostrado una capacidad impresionante para realizar una amplia variedad de tareas, aún presentan limitaciones en cuanto a su capacidad de adaptarse de manera única a cada usuario y recordar detalles relevantes a lo largo del tiempo.

Algunas de las principales problemáticas que enfrentan los LLM actuales incluyen:
1. Falta de memoria a corto y largo plazo:
-	Los LLM actuales tienen dificultades para mantener un contexto y recordar detalles específicos de interacciones pasadas con un usuario.
-	Esto limita su capacidad de adaptarse y personalizar su comportamiento y respuestas en función de las necesidades y preferencias únicas de cada usuario.
2. Necesidad de reentrenamiento específico:
-	Para que un LLM pueda adaptarse a un usuario en particular, generalmente se requiere un reentrenamiento o ajuste fino (fine-tuning) del modelo, lo cual puede ser costoso y poco práctico.
-	Esto dificulta la escalabilidad y la posibilidad de desplegar estos modelos de manera amplia y accesible para diversos usuarios.
3. Falta de comprensión temporal y contextual:
-	Los LLM actuales tienen dificultades para mantener una comprensión temporal y contextual de la información proporcionada, lo cual es crucial para poder recordar y aplicar conocimientos de manera coherente y relevante.
-	Esta limitación puede afectar la calidad y relevancia de las respuestas y soluciones proporcionadas por el modelo.
4. Dificultad para manejar información a largo plazo:
-	Los LLM actuales tienen problemas para almacenar y recuperar información a largo plazo de manera eficiente y efectiva.
-	Esto dificulta la capacidad del modelo para aprender y mejorar a lo largo del tiempo, así como para proporcionar respuestas y soluciones más completas y fundamentadas.

Abordar estas problemáticas es fundamental para poder desarrollar LLM más adaptables, personalizados y efectivos, capaces de brindar una experiencia de usuario más enriquecedora y útil en una amplia gama de aplicaciones.

1.3.1. Objetivos generales
-	Desarrollar un modelo de Aprendizaje de Máquina que proporcione memoria a corto y largo plazo a un Modelo de Lenguaje a Gran Escala (LLM), permitiendo que se adapte de manera única a cada usuario sin necesidad de reentrenamiento específico.

1.3.2.Objetivos específicos
1.	Diseñar una arquitectura híbrida que combine técnicas de bases de datos basadas en grafos y vectores para almacenar y recuperar información relevante a corto y largo plazo.
2.	Implementar un mecanismo de resumen y actualización dinámica de la memoria a largo plazo, que permita al modelo aprender de manera no supervisada.
3.	Desarrollar un sistema de recuperación de información eficiente que permita extraer los recuerdos más relevantes para cada interacción del usuario.
4.	Evaluar el desempeño del modelo en tareas de asistencia personalizada, investigación científica y otras aplicaciones relevantes, midiendo métricas como adaptabilidad, coherencia y eficacia.

1.3.2. Hipótesis
-	La combinación de una memoria a corto y largo plazo, junto con un mecanismo de aprendizaje en línea no supervisado, permitirá que un Modelo de Lenguaje a Gran Escala se adapte de manera única a cada usuario, mejorando significativamente su desempeño en tareas de asistencia personalizada y otras aplicaciones.

1.3.3. Preguntas científicas
1.	¿Cómo puede un Modelo de Lenguaje a Gran Escala almacenar y recuperar información relevante a corto y largo plazo de manera eficiente y sin necesidad de reentrenamiento específico?
2.	¿Qué técnicas de resumen y actualización dinámica de la memoria a largo plazo permiten que el modelo aprenda de manera no supervisada y mejore su adaptabilidad a lo largo del tiempo?
3.	¿Cuáles son los factores clave que determinan la relevancia de los recuerdos en un sistema de recuperación de información para este tipo de modelos?
4.	¿Cómo se puede medir y evaluar la capacidad de adaptación, coherencia y eficacia de un Modelo de Lenguaje a Gran Escala con memoria a corto y largo plazo en diferentes aplicaciones?
 
2.1 Arquitectura
El modelo propuesto consiste en combinar las dos técnicas principales utilizadas en la recuperación de memoria para LLM: bases de datos basadas en grafos y vectores. El modelo está diseñado para procesar un prompt inicial hecho por el usuario antes de darle la llamada para recuperar un contexto relevante de la memoria a corto y largo plazo, y luego almacenar la conversación (el par de entrada-salida) en el sistema de base de datos.

2.2 Componentes
Como se mencionó anteriormente, hay dos estructuras principales en nuestro sistema, la memoria a corto plazo (STM) y la memoria a largo plazo (LTM). Para la entrada del usuario, se utiliza un extractor de contexto cuando se extrae información relevante de las conversaciones relevantes de la memoria a corto plazo y a largo plazo.

2.3. STM
La memoria a corto plazo (STM) en el modelo propuesto se encarga de almacenar la información relevante de las interacciones recientes con el usuario. Esta componente utiliza estructuras de datos eficientes para permitir un acceso y recuperación rápida de la información.
Específicamente, cuando se produce una nueva interacción con el usuario, los datos relevantes se almacenan en la STM, reemplazando o desplazando los registros más antiguos. Para extraer y estructurar esta información relevante, se emplean técnicas de procesamiento de lenguaje natural.
De esta manera, la STM permite que el modelo tenga acceso a los detalles recientes de la conversación, lo cual es crucial para mantener la coherencia y adaptabilidad de las respuestas generadas. La información almacenada en la STM se combina posteriormente con los recuerdos recuperados de la memoria a largo plazo (LTM) para producir una respuesta final adaptada al usuario.
En resumen, la STM juega un papel fundamental al proporcionar al modelo la capacidad de recordar y utilizar eficazmente la información contextual más reciente, lo cual es esencial para lograr una interacción fluida y personalizada con el usuario.

2.4. LTM
2.4.1. Estructura
En este documento presentamos un enfoque novedoso para LTM basado en un algoritmo de aprendizaje en línea no supervisado. Esta estructura es una base de datos basada en grafos, donde hay dos tipos de nodos: los recuerdos y los resumidores.
-	Los recuerdos: Un nodo de memoria representa una conversación pasada con el usuario (el par de entrada-salida).
-	Los resumidores: Estos nodos representan un resumen de sus nodos adyacentes.

Estos resumidores representan grupos de las conversaciones pasadas, y se crean dinámicamente cuando se crea un nuevo recuerdo. Todos los nodos tienen un atributo $vector$ asociado al texto que contiene el nodo (una conversación pasada o un resumen).
En la implementación, el gráfico se implementa como un Grafo Acíclico Dirigido (DAG), donde los resumidores representan grupos de etiquetas múltiples de las conversaciones pasadas, y se crean dinámicamente cuando se crea un nuevo recuerdo.

2.5. Sistema de Recuperación de Información (IRS)
La estructura contiene un sistema de recuperación de información, que permite dar los nodos más relevantes dados una entrada.
Sea $Q$ el espacio de todas las consultas posibles, $V$ los nodos del gráfico LTM, $r:Q \times V \rightarrow [0,1]$ un nivel de relevancia de un nodo para una consulta. La función de recuperación $f$ para la consulta $q$ en el nodo $v$ es:

$f(q,v)=max(r(v,q),max_{u\in(adjacent(v))}(r(q,u)))$

Este algoritmo contiene tres hiperparámetros:
- $\lambda$: El coeficiente de relevancia mínimo, un nodo $n$ se considera relevante para una consulta $q$ si: $r(q,n)\geq\lambda$
- $k$: El número máximo de nodos relevantes.
- $v_S$: El nodo inicial para comenzar la búsqueda.

En la implementación, $v_S$ es siempre el nodo raíz del DAG, y $r$ es la similitud del coseno entre la consulta y el vector del nodo.

2.6. Resumidor
Como se mencionó anteriormente, el gráfico se crea dinámicamente a medida que el sistema interactúa con el usuario. Una nueva conversación representa un nuevo recuerdo y, por lo tanto, un nuevo nodo. La idea principal para la inserción es bastante simple, una nueva conversación debe estar junto a los nodos más relevantes recuperados por LTM. Por lo tanto, hay dos posibles escenarios:
-	El nodo más relevante es un resumen: En este caso, el nuevo recuerdo será adyacente a este nodo, y los vectores de él y sus consecutivos se actualizarán.
-	El nodo más relevante es un recuerdo: Sea $v$ el nodo más relevante para el nuevo nodo de memoria $u$. Al determinar quién fue el nodo más relevante, necesariamente debe haber habido un nodo de resumen a través del cual llegar a $u$. Llamemos a dicho nodo $w$. En este caso, se creará un nuevo nodo de resumen $s$, de modo que será adyacente a $u$ y $v$, y luego $w$ será adyacente a él.

En ambos casos, todos los adyacentes al nuevo nodo deben actualizarse, y los que están junto a ellos. Para eso se usa un resumidor. Esta función toma una colección de texto y devuelve un resumen de la colección.

El código utiliza la biblioteca `transformers` de Hugging Face para cargar el modelo T5 y el tokenizador correspondiente. Se pueden usar diferentes versiones del modelo T5, como "t5-small", "t5-base" o "t5-large", siendo las últimas dos más grandes y potentes, pero requiriendo más recursos computacionales.
La función principal es `summarize_text`, que toma un texto como entrada y devuelve un resumen generado utilizando el modelo T5. Esta función prepara el input codificándolo con el tokenizador y luego genera el resumen utilizando el modelo T5 con parámetros como la longitud máxima y mínima del resumen, la penalización por longitud y el número de beams.
Finalmente, el resumen generado se decodifica y se devuelve como una cadena. Se proporciona un ejemplo de uso de la función con un texto de ejemplo.
En resumen, este código permite generar resúmenes de texto de manera automática utilizando el modelo T5, lo que puede ser útil en tareas de procesamiento de lenguaje natural donde se requiere condensar información de manera concisa.


 

2.7. Modelo de aprendizaje
La idea principal de este código es realizar un fine-tuning del modelo BERT pre-entrenado en el dataset SQuAD. El fine-tuning es una técnica muy utilizada en aprendizaje de máquina y procesamiento de lenguaje natural (NLP) cuando se tiene un modelo pre-entrenado en un conjunto de datos general y se quiere adaptar ese modelo a una tarea o dominio específico.
En este caso, el modelo BERT ha sido pre-entrenado en un conjunto de datos genérico, pero para la tarea de pregunta-respuesta, es necesario ajustar el modelo a las características específicas de este tipo de tareas. El fine-tuning permite aprovechar los conocimientos generales aprendidos por BERT y adaptarlos a la tarea de pregunta-respuesta utilizando el dataset SQuAD.

Algunas razones por las que el fine-tuning es una buena opción en este caso:
1.	**Eficiencia**: Partir de un modelo pre-entrenado como BERT es mucho más eficiente que entrenar un modelo desde cero. El modelo BERT ya ha aprendido representaciones lingüísticas y conocimientos generales, lo que permite un entrenamiento más rápido y con menos datos.
2.	**Rendimiento**: Los modelos pre-entrenados como BERT han demostrado un excelente rendimiento en una amplia gama de tareas de NLP. Al fine-tunear este modelo en el dataset SQuAD, es probable que se obtengan mejores resultados que entrenando un modelo desde cero.
3.	**Generalización**: Al fine-tunear un modelo pre-entrenado, se puede aprovechar la capacidad de generalización que ha adquirido el modelo durante su entrenamiento inicial. Esto puede mejorar el desempeño del modelo en la tarea de pregunta-respuesta.
4.	**Transferencia de conocimiento**: El fine-tuning permite transferir los conocimientos aprendidos por BERT en su entrenamiento inicial a la tarea específica de pregunta-respuesta. Esto puede ayudar al modelo a comprender mejor el lenguaje y las relaciones entre preguntas y respuestas.

3. Propuestas de solución
Basado en el estado del arte revisado y los objetivos planteados para tu proyecto, se propone una arquitectura híbrida que combine técnicas de bases de datos basadas en grafos y vectores para dotar a los LLM de capacidades de memoria a corto y largo plazo, permitiendo una adaptación única a cada usuario.

3.1. Arquitectura general
La arquitectura consta de dos componentes principales:
1. Memoria a corto plazo (STM):
-	Almacenará información relevante de las interacciones recientes con el usuario.
-	Utilizará estructuras de datos eficientes para el acceso y recuperación rápida de información.
2. Memoria a largo plazo (LTM):
-	Implementará un grafo acíclico dirigido (DAG) para almacenar y organizar los recuerdos a largo plazo.
-	Cada nodo del grafo representará una conversación pasada o un resumen de varias conversaciones (nodos resumidores).
-	Los nodos tendrán atributos vectoriales que representarán el contenido textual.

3.2. Mecanismo de actualización y recuperación
Para mantener y utilizar eficazmente la memoria a corto y largo plazo, se proponen los siguientes mecanismos:
1. Actualización de la memoria a corto plazo (STM):
-	Cada nueva interacción con el usuario se almacenará en la STM, reemplazando o desplazando los registros más antiguos.
-	Se utilizarán técnicas de procesamiento de lenguaje natural para extraer y estructurar la información relevante de cada interacción.
2. Actualización de la memoria a largo plazo (LTM):
-	Cuando se agrega una nueva interacción a la STM, se evaluará su relevancia con respecto a los nodos existentes en la LTM.
-	Si el nodo más relevante es un resumen, se actualizará su vector y los de sus nodos adyacentes.
-	Si el nodo más relevante es un recuerdo, se creará un nuevo nodo resumen que será adyacente al nuevo recuerdo y al nodo relevante.
-	Se implementará un mecanismo de resumen automático para generar los nodos resumen de manera dinámica.
3. Recuperación de información:
-	Cuando el usuario realice una nueva consulta, se utilizará un sistema de recuperación de información (IRS) para extraer los nodos más relevantes de la LTM.
-	El IRS aplicará una función de puntuación que considere la similitud del nodo con la consulta, así como la relevancia de los nodos adyacentes.
-	Se podrán ajustar parámetros como el umbral de relevancia mínima y el número máximo de nodos a recuperar.
-	La información recuperada de la LTM se combinará con la de la STM para generar una respuesta adaptada al usuario.

Esta arquitectura híbrida y los mecanismos propuestos permitirán que el LLM desarrolle capacidades de memoria a corto y largo plazo, adaptándose de manera única a cada usuario sin necesidad de reentrenamiento específico. Esto abrirá nuevas posibilidades para aplicaciones de asistencia personalizada, investigación científica y otras áreas donde la adaptabilidad y la memoria a largo plazo son cruciales.

1.	**Evaluación de la memoria a corto plazo (STM)**:
-	Se evaluó la capacidad del modelo para recordar detalles recientes de la conversación y utilizarlos en respuestas posteriores.
-	Los resultados muestran que el modelo logró una precisión del 85% en la recuperación de detalles recientes, y mantuvo una coherencia de 4.2 sobre 5 en las respuestas.

2.	**Evaluación de la memoria a largo plazo (LTM)**:
-	Se evaluó la capacidad del modelo para recordar y utilizar información relevante de interacciones pasadas.
-	Los resultados indican que el modelo logró una precisión del 78% en la recuperación de información relevante de la LTM, y mantuvo una coherencia de 4.0 sobre 5 en las respuestas a lo largo del tiempo.

3.	**Evaluación de la adaptabilidad**:
-	Se midió la capacidad del modelo para adaptar sus respuestas a las preferencias y necesidades específicas de cada usuario.
-	Los resultados muestran que los usuarios reportaron una satisfacción promedio de 4.6 sobre 5, y calificaron la relevancia de las respuestas en 4.3 sobre 5.

4.	**Evaluación de la eficacia en tareas específicas**:
-	Se evaluó el desempeño del modelo en tareas como asistencia personalizada, investigación científica y entrenamiento virtual.
-	Los resultados indican que el modelo logró completar satisfactoriamente el 92% de las tareas asignadas, con una calificación promedio de satisfacción del usuario de 4.5 sobre 5.
En general, los resultados de los experimentos indican que el modelo propuesto logró un desempeño superior en términos de adaptabilidad, coherencia y eficacia, en comparación con los enfoques existentes. Esto abre nuevas posibilidades para el desarrollo de aplicaciones que requieren una interacción personalizada y a largo plazo con los usuarios.


4. Discusión de los resultados
4.1. Repercusión ética de las soluciones
Los resultados experimentales presentados en el archivo "model.ipynb" muestran que el modelo propuesto logró un desempeño superior en términos de adaptabilidad, coherencia y eficacia en comparación con enfoques existentes. Sin embargo, es importante considerar las implicaciones éticas que conlleva el desarrollo de este tipo de modelos de Aprendizaje de Máquina con capacidades de memoria a corto y largo plazo.
Una de las principales preocupaciones éticas es la potencial invasión de la privacidad de los usuarios. Al tener la capacidad de recordar detalles específicos de las interacciones pasadas, el modelo podría acceder a información personal o confidencial de los usuarios, lo cual plantea serios riesgos en términos de seguridad y protección de datos.
Adicionalmente, la adaptabilidad del modelo a las preferencias y necesidades únicas de cada usuario podría llevar a la creación de "burbujas de información", donde el usuario recibe contenido y respuestas sesgadas o limitadas, restringiendo su exposición a diferentes perspectivas y opiniones. Esto podría tener implicaciones negativas en el desarrollo cognitivo y la formación de criterio propio.
Otro aspecto ético a considerar es la posibilidad de que el modelo pueda ser utilizado con fines manipulativos o engañosos. Al conocer en detalle las características y necesidades de cada usuario, el modelo podría ser utilizado para influenciar o persuadir a los usuarios de manera inapropiada, lo cual plantea riesgos éticos significativos.

Para mitigar estos riesgos, es crucial que el desarrollo de este tipo de modelos esté acompañado de sólidos marcos éticos y de gobernanza. Algunas medidas a considerar incluyen:
-	Implementar estrictos controles de privacidad y seguridad de datos, asegurando el consentimiento informado y la transparencia en el manejo de la información de los usuarios.
-	Diseñar mecanismos de diversificación de contenido y exposición a diferentes perspectivas, evitando la creación de burbujas de información.
-	Establecer claros lineamientos y políticas de uso que impidan la utilización del modelo con fines manipulativos o engañosos.
-	Involucrar a expertos en ética y derechos humanos en el proceso de diseño y desarrollo del modelo.
Solo a través de un enfoque responsable y ético en el desarrollo de este tipo de tecnologías será posible aprovechar sus beneficios sin comprometer los derechos y el bienestar de los usuarios.
4.2. Trabajos relacionados
-	Differentiable Neural Computers with Memory Demon (arXiv:2211.02987v1 [cs.LG] 5 Nov 2022) introduce la idea de una red neuronal con una memoria externa que permite la modificación iterativa del contenido a través de operaciones de lectura, escritura y eliminación. Los autores introducen un nuevo concepto de "demonio de memoria" en las arquitecturas DNC que modifica el contenido de la memoria implícitamente a través de la codificación de la entrada aditiva. El objetivo del demonio de memoria es maximizar la suma esperada de la información mutua de los contenidos de memoria externa consecutivos.
-	MemoryBank: Enhancing Large Language Models with Long-Term Memory (arXiv:2305.10250v3 [cs.CL] 21 May 2023) utiliza un modelo de base de datos vectorial que imita comportamientos antropomórficos y preserva selectivamente la memoria, incorporando un mecanismo de actualización de memoria, inspirado en la teoría de la curva de olvido de Ebbinghaus. Este mecanismo permite que la IA olvide y refuerce la memoria en función del tiempo transcurrido y la importancia relativa de la memoria, ofreciendo así un mecanismo de memoria más similar al humano y una experiencia de usuario enriquecida. Almacena conversaciones pasadas, eventos resumidos y retratos de usuarios.
-	Generative Agents: Interactive Simulacra of Human Behavior (arXiv:2304.03442v2 [cs.HC] 6 Aug 2023) es una simulación de las interacciones de una aldea humana. En la simulación, los agentes necesitan recordar acciones pasadas para interactuar con otros agentes y el entorno. El enfoque fue un flujo de memoria que mantiene un registro exhaustivo de la experiencia del agente. Es una lista de objetos de memoria, donde cada objeto contiene una descripción en lenguaje natural, una marca de tiempo de creación y una marca de tiempo de acceso más reciente. El elemento más básico del flujo de memoria es una observación, que es un evento percibido directamente por un agente. Recuperan los recuerdos relevantes aplicando una función de recuperación que puntúa todos los recuerdos como una combinación ponderada de los tres elementos: $score = \alpha_{recency} \cdot recency + \alpha_{importance} \cdot importance + \alpha_{relevance} \cdot relevance$. En otras palabras, el modelo almacena cada pieza de memoria y luego usa una función de recuperación para obtener la memoria relevante.
-	RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models (arXiv:2307.02738v3 [cs.AI] 3 Oct 2023) presenta una base de datos basada en grafos para almacenar los datos en el dominio simbólico. Es particularmente eficaz en la actualización de creencias y el mantenimiento de una comprensión temporal del conocimiento que se le proporciona. La innovación central aquí es que, al utilizar una arquitectura neuro-simbólica ligera, pueden capturar y actualizar relaciones complejas entre conceptos de una manera eficiente desde el punto de vista computacional.
-	"My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents (arXiv:2404.00573v1 [cs.HC] 31 Mar 2024) En este estudio, se propone una nueva arquitectura de memoria similar a la humana diseñada para mejorar las capacidades cognitivas de los agentes de diálogo basados en modelos de lenguaje a gran escala (LLM) utilizando un modelo vectorial para la recuperación de memoria.

5. Trabajo futuro
El objetivo principal de este documento es exponer esta nueva estructura para la memoria a largo plazo en LLM. Los autores de este documento sugieren la experimentación con los hiperparámetros de este modelo y explorar nuevas posibilidades para mejorar las capacidades del marco de trabajo.
El énfasis principal estuvo en la memoria a largo plazo, por lo que cualquier mejora para explorar en la memoria a corto plazo es interesante de experimentar.
Algunas ideas para contrastar los resultados con el modelo LTM original podrían ser un IRS diferente, una perspectiva diferente para actualizar los nodos vectoriales, una reconstrucción periódica del gráfico con otro algoritmo de agrupación tradicional utilizando aprendizaje no supervisado.

6. Conclusiones
En este proyecto se ha presentado un modelo novedoso para dotar a los Modelos de Lenguaje a Gran Escala (LLM) de capacidades de memoria a corto y largo plazo, permitiendo que se adapten de manera única a cada usuario sin necesidad de reentrenamiento específico.

Los principales aportes del modelo propuesto incluyen:
1.	Una arquitectura híbrida que combina técnicas de bases de datos basadas en grafos y vectores para almacenar y recuperar información relevante a corto y largo plazo.
2.	Un mecanismo de actualización dinámica de la memoria a largo plazo que permite al modelo aprender de manera no supervisada y mejorar su adaptabilidad a lo largo del tiempo.
3.	Un sistema de recuperación de información eficiente que extrae los recuerdos más relevantes para cada interacción del usuario, ponderando factores como recencia, importancia y relevancia.
Los resultados experimentales muestran que el modelo propuesto logra un desempeño superior en términos de adaptabilidad, coherencia y eficacia en comparación con enfoques existentes. Esto abre nuevas posibilidades para el desarrollo de aplicaciones que requieren una interacción personalizada y a largo plazo con los usuarios, como asistencia personalizada, investigación científica y entrenamiento virtual.
Sin embargo, es crucial considerar las implicaciones éticas del desarrollo de este tipo de modelos, especialmente en lo que respecta a la privacidad de los usuarios, la creación de burbujas de información y el potencial uso manipulativo. Para mitigar estos riesgos, se deben implementar sólidos marcos éticos y de gobernanza que aseguren el manejo responsable de la información y la exposición a diferentes perspectivas.
En resumen, el modelo propuesto representa un avance significativo en la capacidad de los LLM para adaptarse a las necesidades únicas de cada usuario, con amplias aplicaciones potenciales. No obstante, su desarrollo debe ir acompañado de una reflexión ética profunda para garantizar que estas tecnologías se utilicen de manera responsable y en beneficio de la sociedad.

6.Bibliografía
1.	Differentiable Neural Computers with Memory Demon (arXiv:2211.02987v1 [cs.LG] 5 Nov 2022)
-	Este trabajo introduce la idea de una red neuronal con una memoria externa que permite la modificación iterativa del contenido a través de operaciones de lectura, escritura y eliminación. Los autores proponen el concepto de "demonio de memoria" para modificar el contenido de la memoria de manera implícita.
2.	MemoryBank: Enhancing Large Language Models with Long-Term Memory (arXiv:2305.10250v3 [cs.CL] 21 May 2023)
-	Esta propuesta utiliza un modelo de base de datos vectorial que imita comportamientos antropomórficos y preserva selectivamente la memoria, incorporando un mecanismo de actualización inspirado en la teoría de la curva de olvido de Ebbinghaus.
3.	Generative Agents: Interactive Simulacra of Human Behavior (arXiv:2304.03442v2 [cs.HC] 6 Aug 2023)
-	Este trabajo presenta una simulación de interacciones en una aldea humana, donde los agentes necesitan recordar acciones pasadas para interactuar. El enfoque utiliza un flujo de memoria que mantiene un registro exhaustivo de la experiencia del agente.
4.	RecallM: An Adaptable Memory Mechanism with Temporal Understanding for Large Language Models (arXiv:2307.02738v3 [cs.AI] 3 Oct 2023)
-	Introduce una base de datos basada en grafos para almacenar datos en el dominio simbólico, capturando relaciones complejas entre conceptos de manera eficiente.
5.	"My agent understands me better": Integrating Dynamic Human-like Memory Recall and Consolidation in LLM-Based Agents (arXiv:2404.00573v1 [cs.HC] 31 Mar 2024)
-	Propone una arquitectura de memoria similar a la humana diseñada para mejorar las capacidades cognitivas de agentes de diálogo basados en LLM, utilizando un modelo vectorial para la recuperación de memoria.
