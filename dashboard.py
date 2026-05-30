# ==========================================
# Dashboard Streamlit - Gastos Médicos
# ==========================================

# ==========================================
# Librerías
# ==========================================

import streamlit as st # Esta libreia se utiliza para poder crear aplicaciones web que va a ser interactiva de forma rápida y sencilla utilizando Python 
import numpy as np # Esta libreria se utiliza para realizar operaciones matemáticas como son los arreglos, matrices de forma eficiente
import pandas as pd # Esta libreia se utiliza para manipular, analzar y organizar datos en estructuras por medio de tablas y estructuras por medio de un tipo DataFrame  
import seaborn as sns # Esta libreria se utiliza para establecer las gráficas a nivel estadistico y generar visualización de esos datos de forma sencilla y actrativa para el usuario 
import matplotlib.pyplot as plt # Esta libreria se utiliza para poder establecer gráficas, diagramas y visualizaciones de datos de forma dinamica y personalizada para el usuario 

import plotly.express as px # Esta libreia se utiliza para poder establecer el dinamismo en las gráficos y interactivas de esos datos para el usuario 
import plotly.graph_objects as go # Esta libreria se utiliza para poder generar gráficos personalizados a nivel interactivo con mayor control y detalle para el usuario 

from sklearn.model_selection import train_test_split # Esta libreria se va a utilizar para poder dividir los datos en dos conjuntos que van a ser los datos de entrenamiento y los datos de prueba para poder ser aplicados posteriormente en modelos de Machine Learning 
from sklearn.linear_model import LinearRegression # Esta libreria se utiliza con el objetivo de poder crear modelos de regresión lineal y poder realizar la predicción de los valores a nivel númerico al utilizar está técnica de Machine Learning  
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
) # Esta libreria se utiliza para determinar los parametrosa a utilizar para poder evaluar el rendimiento y la precisión de los modelos aplicados de Machine Learning y poder determinar que tan bien se ajustan a los datos y su capacidad de predicción 

from sklearn.preprocessing import StandardScaler # Esta libreria se utiliza para poder estandarizar los datos, con el objetivo de ajustar los valores a una escala con media 0 y desviación estándar 1, se utiliza para poder mejorar el rendimiento de los modelos de Machine Learning 
import time # Se utiliza esta libreria para poder aplicar tiempos, pausas y medición de como se aplica la duración de la ejecución de ciertas partes del código, para poder mejorar la experiencia del usuario

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
) # Esta libreria se utiliza para poder adquirir herramientas de ReportLab ya se va a utilizar para crear y estructurar el documento de PDF que va a tener como objetivo monstrar el reporte inteligente, esto indicando el texto, espacios y saltos de página para poder organizar el contenido del PDF para el usuario de forma profesional y atractiva 
from reportlab.lib.styles import getSampleStyleSheet # Se utiliza esta libreria que va a permitir obtener estilos ya establecidos y poder darle formato al contenido de documentos PDF, para poder mejorar la presentación y legibilidad del reporte inteligente de negocios y salud que se va a generar para el usuario 
from reportlab.lib.pagesizes import letter # Esta libreia se va a utilizar para poder establecer el tamaño que va a tener cadapágina del documento PDF que va a ser del tamaño carta
from reportlab.pdfbase import pdfmetrics # Esta libreria se utiliza para poder llevar a cabo las funciones de gestión y configuración a nivel tipografico en el documento PDF
from reportlab.lib import colors # Esta libreria se va a utilizar para poder establecer y manejar los colores que se van a aplicar en el documento PDF, enfocado principalmente en las gráficas adjuntadas en el documento
from io import BytesIO # Esta libreria se utiliza para poder manejar los datos para poder establecer en la memoria como si fueran archivos, sin requerir guardarlos de forma física 

from reportlab.lib.styles import ParagraphStyle # Se utiliza esta libreria para poder establecer y poder realizar la personalización aplicada a los estilos de los textos en los párrafos que se conforman en el documetno PDF  
from reportlab.platypus import Image, Table, TableStyle # Se utiliza esta libreria para poder agregar imágenes, tabalas y estilos interactivos aplicados en el documento PDF 

# Importación de la API de Groq
try: # Esta linea se va a utilizar para determinar que suceda esto
    from groq import Groq # Se va a importar la clase Grop desde la librería de groq que permite llamar las funcionales de la inteligencia artifical por medio de la clave key que proporciona Groq
except ModuleNotFoundError: # Esta linea se va a utilizar es para determinar si sucede un error, debido a que la libreria no está instalada por lo tanto se va a ejecutar el comando except
    st.error(
        "La librería groq no está instalada. "
        "Ejecute: pip install groq"
    ) # Esta linea se utiliza con el objetivo de poder monstrarle al usuario que va a existir un error en la funcionalidad que se aplica para generar el reporte inteligente de negocio en Streamlit 
    st.stop() # Esta linea se utiliza para poder deterner de forma completa la ejecución de esa aplicación

# ==========================================
# Configuración Dashboard
# ==========================================

st.set_page_config( # Estos comandos se utilizan para determinar las propiedades de la página de Streamlit
    page_title="Dashboard Inteligente de Gastos Médicos", # Esta linea que van a establecer el rítulo de la pagina, pero de la pastaña que va a visualizar el usuario
    page_icon="📊", # Esta linea va a permitir definir el ícono que va a visualizar el usuario en la pestaña de la pagina que va a visualizar el usuario
    layout="wide" # Esta línea se establece un diseño de una página en modo ancho para que el usuario pueda visualizar mejor la información el usuario
)

# ==========================================
# Estilos CSS
# ==========================================

st.markdown(""" # Esta serie de comandos permite insertar código de tipo CSS y HTML que van a permitir pernozaliar la aplicación de Streamlit 
<style> # Permite esta linea establecer el diseño del CCS que va a tener la pagina de Streamlit

.main { # Permite cambiar el color del fondo de la pagina principal de la aplicación
    background-color: #0f172a;
}

h1, h2, h3 { # Permite cambiar el color de los títulos h1, h2 y h3 a un color blanco
    color: white !important;
}

p { # Permite cambiar el color de los parrafos a un color blanco
    color: white;
}

.stMetric { # Permite establecer las metricas que va a llevar el diseño de la aplicación desarrollada en Streamlit
    background-color: #111827; # Permite definir el color del fondo que van a tener las tarjetas métricas que van a ser evidenciadas en la aplicación
    padding: 20px; # Permite definir ese espacio interno que va a tener cada una de las tarjetas métricas 
    border-radius: 15px; # Permite aplicarle a las tarjetas un borde redondeado para darle un estilo visual y atractivo para el usuario
    border: 1px solid #374151; # Permite establecer un borde de forma sutil en las tarjetas métricas para darle un estilo visual para el usuario aplicando un color gris oscuro
    text-align: center; # Permite centrar el contenido que va a tener cada una de las tarjetas métricas para mejorar la presentación visual para el usuario
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3); # Permite agregar una pequeña sombra a la tarjeta métrica para darle un atractivo más visual produciendo un efecto de profundidad a la tarjeta
}

.stMetric label { # Permite perzonalizar ese texto que va a ser contenido esa tarjeta métrica de las etiquetas definidas por medio de las metricas 
    color: white !important;
    font-size: 18px !important;
}

.stMetric div { # Permite personalizar ese valor a nivel númerico permite evidenciar por medio de la aplicación de las metricas definidas 
    color: #00FFAA !important;
    font-size: 28px !important;
    font-weight: bold;
}

section[data-testid="stSidebar"] { # Permite cambiar el color del fondo de la barra lateral de la navegación que va a permitr visualizar al usuario las diferentes secciones del dashboard
    background-color: #1e293b;
}

</style>
""", unsafe_allow_html=True) # Esta declaración que va a permitir establecer en el Streamlit poder llevar a cabo la interpretación del lenguaje del CSS y HTML

# ==========================================
# Título
# ==========================================

st.title("Dashboard Inteligente de Gastos Médicos") # Esta linea permite evidenciar de forma visual el título principal de la aplicación que va a ser visualizada por el usuario
st.markdown("### Machine Learning + Analítico + Simulación") # Esta linea permite evidenciar de forma visual un subtítulo aplicando un formato Mardown

# ==========================================
# Sección de estados de modelos
# ==========================================

if 'modelo1_generado' not in st.session_state: # Esta linea permite verificar que la variable que representa el modelo 1 que es "modelo1_generado" no exista en el estado de session_state
    st.session_state.modelo1_generado = False # Si la variable "modelo1_generado" no existe, entonces la crea y permite asignar a esa variable el valor de False, esto con el objetivo de poder controlar la generación del modelo 1

if 'modelo2_generado' not in st.session_state: # Esta linea permite verificar que la variable que representa el modelo 2 que es "modelo2_generado" no existe en el estado de session_state
    st.session_state.modelo2_generado = False # Si la variable "modelo2_generado" no existe, entonces la crea y permite asignar a esa variable el valor de False, esto con el objetivo de poder controlar la generación del modelo 2 

if 'modelo3_generado' not in st.session_state: # Esta linea permite verificar que la variable que representa el modelo 3 que es "modelo3_generado" no existe en el estado de session_state
    st.session_state.modelo3_generado = False # Si la variable "modelo3_generado" no existe, entonces la crea y permite asignar a esa variable el valor de False, esto con el objetivo de poder controlar la generación del modelo 3

# ==========================================
# Carga de datos
# ==========================================

archivo = st.file_uploader(
    "Cargue el archivo Excel",
    type=["xlsx"]
) # Esta linea permite establecer el poder cargar los archivos pero solo de tipo Excel por medio de establecer la extensión de .xlsx, esto con el objetivo de poder realizar el procesamiento, análisis y conclusiones de esos datos subministrados por el usuario

# ==========================================
# Si existe el archivo
# ==========================================

if archivo is not None: # Esta linea permite aplicar una condición que va a obedecer si el usuario ha ingresado un archivo, esto va a tener como objetivo poder ejecutar el proceso de procesamiento de esos datos, análisis para su posterior aplicación de los modelos de Machine Learning y la generación de un reporte inteligente de negocio

    datos = pd.read_excel(archivo) # Se establece una variable que va a guardar el contenido del archivo de tipo Excel con la extensión de .xlsx, esto con el objetivo de poder visualizar, manipular y analizar esos datos que contiene ese archivo

    # ==========================================
    # Sección de estados de notificaciones
    # ==========================================

    if 'notificacion_carga_mostrada' not in st.session_state: # Esta linea permite aplicar una condición que va obedecer que si la variable "notificacion_carga_mostrada" sino existe en el estado de session_state, entonces procede a crearse esa variable y a asignarle el valor de False, esto se realiza con el objetivo de controlar cuando se crea la notificación de forma exitosa y que la visualice el usuario 
        st.session_state.notificacion_carga_mostrada = False # Si la variable "notificacion_carga_mostrada" no existe, entonces la crea y permite asignar a esa variable de False, esto con el objetivo de poder controlar la generación la notificación de forma exitosa y que la visualice el usuario esa notificación de forma única, evitando que se muestre de forma repetida cada vez que se ejecute el proceso de carga de datos en la aplicación 

    # ==========================================
    # Notificación de que se cargo de forma exitosa
    # ==========================================

    if not st.session_state.notificacion_carga_mostrada: # Esta linea permite aplicar una condición que va a obedecer a la verificación de que si la notificación de carga aún no ha sido mostrada, entonces se declara como un valor de False 

        with st.spinner("Subiendo base de datos..."): # Esto permite monstrar una animación de carga al usuario evidenciado al usuario que se esta cargando la base de datos
            time.sleep(2) # Esta linea permite pausar la ejecución del programa durante dos segundos, para poder llevar a cabo la simulación de procesar o cargar los datos

        st.toast(
            "La base de datos se cargó correctamente",
            icon="📁"
        ) # Se utiliza esta linea para poder monstrar la notificación de tipo toast que es la que se muestrta en la esquina superior del lado izquierdo de la pantalla, indicando al usuario que se cargo de forma exitosa la base de datos, y se agrega un icono que representa el archivo o base de datos cargada por el usuario 

        st.session_state.notificacion_carga_mostrada = True # Esta linea permite marcar la notificación para mostrarla solo una vez, esto con el objetivo de evitar repeticiones

    st.success("Archivo cargado correctamente") # Esta linea permite notificarle al usuario que ya se cargo el archivo de forma exitosa y que lo visualice el usuario por medio de una notificación de tipo success que se va a mostrar en la pantalla

    # ==========================================
    # Limpieza de datos
    # ==========================================

    datos.drop_duplicates(inplace=True) # Esta linea va a permitir eliminar las filas duplicadas en el DataFrame para poder evitar registros duplicados, esto por medio del inplace=True que va a realizar el cambio que se va a aplicar de forma directa en el DataFrame original

    # ==========================================
    # Variable objetivo global
    # ==========================================
    y = datos['gastos'] # Se establece una variable que va a representar a esa variable objetivo que se va a utilizar para poder aplicar los modelos de Machine Learning, esto para poder predecir los gastos médicos, esta variable se va a establecer de forma global para luego ser utilizado posteriormente en cada uno de los modelos

    # ==========================================
    # Variables categóricas
    # ==========================================

    datos['sexo'] = datos['sexo'].astype('category') # Esta linea permite convertir la variable que va a denominarse como "sexo" que va a ser una variable de tipo categorica, esto con el objetivo de poder llevar a cabo un proceso de optimización en el manejo de esa variable, esto con el objetivo de mejorar el rendimiento del procesamiento de esa variable, para luego ser aplicada en los Modelos de Machine Learning
    datos['fumador'] = datos['fumador'].astype('category') # Esta linea permite convertir la variable que va a denominarse como "fumador" que va ser una variable de tipo categorica, esto con el objetivo de poder llevar a cabo un proceso de optimización en el manejo de esa variable, esto con el objetivo de mejorar el rendimiento del procesamiento de esa variable, para luego ser aplicada en los Modelos de Machine Learning
    datos['region'] = datos['region'].astype('category') # Esta linea permite convertir la variable que va a denominarse como "region" que va a ser una variable de tipo categorica, esto con el objetivo de poder llevar a cbo un proceso de optimización en el manejo de esa variable, esto con el objetivo de mejorar el rendimiento del procesamiento de esa variable, para luego ser aplicada en los Modelos de Machine Learning

    # ==========================================
    # Variable IMC
    # ==========================================

    datos['imc_cat'] = np.where(
        datos['imc'] >= 30,
        'Obeso (IMC>=30)',
        'No Obeso'
    ) # Esta linea va a permitir crear una variable nueva que va a denominarse como "imc_cat" que va a ser una variable de tipo categorica, esa variable se va a crear por medio de la función np.where() que va a permitir determinar el valor que va a tomar esa variable que va a ser "imc_cat" que va a depender del valor que tenga la variable para poder establecerla en una categoria de "Obeso (IMC debe ser mayor o igual a 30)" y de lo contrario se establece de que el usuario "No es obeso"

    # ==========================================
    # Barra Lateral de Navegación
    # ==========================================

    st.sidebar.title("Navegación") # Esta linea va a permitir visualizar el titulo de esa sección que va a estar ubicada en la barra lateral de navegación

    opcion = st.sidebar.radio(
        "Seleccione una sección",
        [
            "EDA",
            "Correlaciones",
            "Modelo 1",
            "Modelo 2",
            "Modelo 3",
            "Comparación",
            "Simulador",
            "Reporte"
        ]
    ) # Esta linea va a permitir definir las secciones que va a tener la barra lateral de navegación, para poder que el usuario visualice y se pueda transladar a las otras secciones definidas en la aplicación 

    # ==========================================
    # Avance
    # ==========================================

    st.subheader("Vista previa") # Esta linea permite establecer un subtítulo que va a poder visualizar el usario en la sección princiapl que va a determinar la funcionalidad de la aplicación

    st.dataframe(datos.head()) # Esta linea va a permitir visualizar los primeros datos de las filas del DataFrame para que el usuario pueda visualizar los datos que el cargo 

    # ==========================================
    # Calidad de datos
    # ==========================================

    st.subheader("Calidad de los Datos") # Esta linea permite mostrar al usuario un subtitulo para poder establecer al usario que esta en la sección de análisis de calidad en los datos cargados por el usuario

    col1, col2, col3, col4 = st.columns(4) # Esta linea permite definir las cuatro columnas que van a tener como objetivo organizar las métricas relevantes de forma grafica y visual para el usuario 

    with col1: # Esta linea permite definir la primera métrica que va a ser la cantidad de filas que tiene el dataset que cargo el usuario
        st.metric(
            label="Filas",
            value=f"{datos.shape[0]:,}"
        )

    with col2: # Esta linea permite definir la segunda métrica que va a ser la cantidad de columnas que va a tener el dataset que cargo el usuario
        st.metric(
            label="Columnas",
            value=datos.shape[1]
        )

    with col3: # Esta linea permite definir la tercera métrica que va a ser el determinar el número de datos que se hayan registrado duplicados en el dataset cargado por el usuario
        st.metric(
            label="Duplicados",
            value=datos.duplicated().sum()
        )

    with col4: # Esta linea permite definir la cuarta métrica que va a ser el promedio que se registra de la columna "gastos" del dataset que cargo rl usuario
        st.metric(
            label="Gasto Promedio",
            value=f"${datos['gastos'].mean():,.0f}"
        )

    # ==========================================
    # Función Evaluación
    # ==========================================

    def evaluacion(y_true, y_pred, model_name): # Esta linea va a permitir definir una función que se va a aplicar en cada modelo de Machine Learnig que se aplique en la aplicación, que va a tener como parametros y_true, y_pred, modelo_name que va a poder definir la estructura general que se va a aplicar a los modelos de Machine Learning

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        ) # Esta linea va a establecer el calculo de una de las metricas que determina si el modelo es el mejor con respecto a los otros que es el error cuadratico medio que permite evaluar la precisión del modelo que se va a utilizar la función mean_squared_error y que va a tener como parametros las variables y_true y y_pred 

        mape = mean_absolute_percentage_error(
            y_true,
            y_pred
        ) # Esta linea va a establecer el calculo de otra métrica fundamental para determinar si el modelo es el mejor con respecto a los otros que es el error porcentual absoluto medio que va a medir la precisión del modelo pronóstico que va a utilizar la función mean_absolute_percentage_error y que va a tener como parametros las variables y_true y y_pred   

        r2 = r2_score(
            y_true,
            y_pred
        ) # Esta linea va a establecer el calculo de otra metrica importante para determinar si el modelo es el mejor con respecto a los otros modelos que es el coeficiente de determinación que va a ser una métrica que permite evauluar la precisión del modelo de regresión que se va a utilizar la función r2_score y que va a tener como parametros las variables y_true y y_pred  

        return {
            'Modelo': model_name,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        } # Esta linea permite retronar los valores del nombre del modelo que se esta aplicando, el rmse que es el error cuadrado medio, el mape que es el error porcentual medio absoluto y el r2 que es el coeficiente de determinación esto evaluaco en cada uno de los modelo aplicados

    # ==========================================
    # Modelo 1
    # ==========================================

    if not st.session_state.modelo1_generado: # Esta linea aplica una condicional que va a permitir verificar si el modelo 1 no ha sido generado de forma previa en la aplicación en el momento que se haya conectado el usuario

        with st.spinner("Generando Modelo 1..."): # Esta linea permite evidenciar de forma grafica por medio de una animación que se esta cargando, que esto se traduce en que se esta entrenando el modelo 

            time.sleep(5) # Esta linea permite aplicar una simulación de tiempo para que se procese el entrenamiento de los datos adjuntados por el usuario

            x1 = datos[['edad', 'imc', 'hijos']] # Esta linea determina la selección de las variables que van a actuar de forma independiente del modelo

            X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
                x1,
                y,
                test_size=0.2,
                random_state=42
            ) # Esta linea permite dividir los datos del entrenamiento y los datos de prueba que se van a aplicar en el modelo, esto estableciendo una semilla que en este caso adquiere el valor de 42 y que va a tener como parametros la variable y que es la variable objetivo global y x1 que es la variable que acumula las variables que actuan de forma independiente al modelo 1

            modelo1 = LinearRegression() # Esta linea permite crear un modelo de regresión lineal como el primer modelo sugerido para estos datos

            modelo1.fit(
                X_train_1,
                y_train_1
            ) # Esta linea permite entrenar los datos de entrenamieno por medio del modelo planteado que en este caso es el de regresión lineal que exige los parametros X_train_1 y y_train_1 que son los datos de entrenamiento

            y_pred_1 = modelo1.predict(X_test_1) # Esta linea aplica la predicción sobre los datos de prueba, pero siguiendo el modelo 1 que es de regresión lineal que exige un parametro que es X_test_1 que va a representar los datos de prueba 

            res_1 = evaluacion(
                y_test_1,
                y_pred_1,
                'Modelo 1'
            ) # Esta linea permite llamar a la función declarada como "evaluacion" para poder determinar las metricas para poder determinar como se comporta el modelo para ello se exige los parametros de y_test_1 que representan los datos de prueba, la variable y_pred_1 que son los datos que se aplico el modelo de Machine Learning y por ultimo el nombre del modelo que en este caso es Modelo 1

            st.session_state.modelo1 = modelo1 # Esta linea permite guardar el modelo de regresión lineal entrenado en la sesión que ingreso el usuario para poder ser reutilizado 
            st.session_state.res_1 = res_1 # Esta linea permite guardar las métricas que permiten evaluar el modelo 1 como lo son: rmse, mape y r^2
            st.session_state.x1 = x1 # Esta linea permite guardar las variables que son independientes y que fueron utilizadas en el Modelo 1
            st.session_state.y_pred_1 = y_pred_1 # Esta linea permite guardar esas predicciones realizadas por medio del modelo sobre los datos de prueba 
            st.session_state.y_test_1 = y_test_1 # Esta linea permite guarda los valores que son reales que se ubicaron en el conjunto de prueba, con el objetivo de ser comparados con los datos predichos por el modelo de regresión lineal aplicado 

            st.session_state.modelo1_generado = True # Esta linea permite establecer una marca en el modelo se le indica al usuario que ya se a generado el modelo y esto evita el volver a recalcular el modelo 1

        st.toast(
            "Modelo 1 generado correctamente",
            icon="📊"
        ) # Se utiliza esta linea para poder monstrar la notificación de tipo toast que es la que se muestrta en la esquina superior del lado izquierdo de la pantalla, indicando al usuario que a terminado de generarse el modelo 1 de regresión lineal, y se agrega un icono que representa que termino el modelo de realizarse para que pueda ser visualizado por el usuario

    modelo1 = st.session_state.modelo1 # Esto permite recuperar el modelo que se entreno desde el área de session_state, con el objetivo de poder utilizarlo sin volver a entrenar el modelo 
    res_1 = st.session_state.res_1 # Esto permite recuperar los resultados que determinan la evaluación del modelo 1 de regresión lineal que son: rmse, mape y r^2  
    x1 = st.session_state.x1 # Esto permite recuperar las variables que son independientes del Modelo 1 de regresión lineal
    y_pred_1 = st.session_state.y_pred_1 # Esto permite recuperar las predicciones realizadas a traves del modelo que se utiliza regresión lineal aplicado a el conjunto de datos de prueba 
    y_test_1 = st.session_state.y_test_1 # Esto permite recuperar los valores reales del conjunto de prueba, con el objetivo de poder compararlos con las predicciones realizadas por el modelo 1 que aplica regresión lineal  

    # ==========================================
    # Modelo 2
    # ==========================================

    if not st.session_state.modelo2_generado: # Esta linea aplica una condicional que va a permitir verificar si el modelo 2 no ha sido generado de forma previa en la aplicación en el momento que se haya conectado el usuario 

        with st.spinner("Generando Modelo 2..."): # Esta linea permite evidenciar de forma grafica por medio de una animación que se esta cargando, que esto se traduce en que se esta entrenando el Modelo 2 

            time.sleep(5) # Esta linea permite aplicar una simulación de tiempo para que se procese el entrenamiento de los datos adjuntados por el usuario

            df_encoded = pd.get_dummies(
                datos,
                drop_first=True
            ) # En esta linea permite realizar el proceso de conversión de las variables categóricas en variables numéricas por medio de aplicar el metodo one-hot encoding que tiene como dos parametros el datos y el parametro drop_first que se establece con el valor de True paea indicar que se elimine la primera categoria para evitar multicolinealidad en el modelo 2 

            columnas_a_borrar = ['gastos', 'riesgo_modelo1', 'riesgo_modelo2', 'riesgo'] # Esta linea crea una lista de las columnas que se desean eliminar si existen dentro del dataset cargado por el usuario 
            columnas_existentes = [col for col in columnas_a_borrar if col in df_encoded.columns] # Esta linea permite establecer un filtro que va a tener como objetivo el filtrar solo las columnas que existan realmente en el DataFrame cargado por el usuario 
            
            x2 = df_encoded.drop(
                columnas_existentes,
                axis=1
            ) # Esta linea permite eliminar las columnas que fueron seleccionadas para establecer las variables independientes del modelo, que va a tener como parametro la variable columnas_existentes que es la variable que determina las columnas que existen y axis que va a adquirir el valor de 1 que va a indicar que la operación se va a aplicar a lo largo de las columnas 

            X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
                x2,
                y,
                test_size=0.2,
                random_state=42
            ) # Esta linea permite dividir los datos del entrenamiento y los datos de prueba que se van a aplicar en el modelo, esto estableciendo una semilla que en este caso adquiere el valor de 42 y que va a tener como parametros la variable y que es la variable objetivo global y x2 que es la variable que acumula las variables que actuan de forma independiente al modelo 2 

            modelo2 = LinearRegression() # Esta linea permite crear un modelo de regresión lineal como el segundo modelo sugerido para estos datos 

            modelo2.fit(
                X_train_2,
                y_train_2
            ) # Esta linea permite entrenar los datos de entrenamieno por medio del modelo planteado que en este caso es el de regresión lineal que exige los parametros X_train_2 y y_train_2 que son los datos de entrenamiento

            y_pred_2 = modelo2.predict(X_test_2) # Esta linea aplica la predicción sobre los datos de prueba, pero siguiendo el modelo 2 que es de regresión lineal que exige un parametro que es X_test_2 que va a representar los datos de prueba  

            res_2 = evaluacion(
                y_test_2,
                y_pred_2,
                'Modelo 2'
            ) # Esta linea permite llamar a la función declarada como "evaluacion" para poder determinar las metricas para poder determinar como se comporta el modelo para ello se exige los parametros de y_test_2 que representan los datos de prueba, la variable y_pred_2 que son los datos que se aplico el modelo de Machine Learning y por ultimo el nombre del modelo que en este caso es Modelo 2

            st.session_state.modelo2 = modelo2 # Esta linea permite guardar el modelo de regresión lineal entrenado en la sesión que ingreso el usuario para poder ser reutilizado 
            st.session_state.res_2 = res_2 # Esta linea permite guardar las métricas que permiten evaluar el Modelo 2 como lo son: rmse, mape y r^2
            st.session_state.x2 = x2 # Esto permite recuperar las variables que son independientes del Modelo 2 de regresión lineal
            st.session_state.y_pred_2 = y_pred_2 # Esto permite recuperar las predicciones realizadas a traves del modelo que se utiliza regresión lineal aplicado a el conjunto de datos de prueba  
            st.session_state.y_test_2 = y_test_2 # Esta linea permite guarda los valores que son reales que se ubicaron en el conjunto de prueba, con el objetivo de ser comparados con los datos predichos por el modelo de regresión lineal aplicado 

            st.session_state.modelo2_generado = True # Esta linea permite establecer una marca en el modelo se le indica al usuario que ya se a generado el modelo y esto evita el volver a recalcular el modelo 2 

        st.toast(
            "Modelo 2 generado correctamente",
            icon="📊"
        ) # Se utiliza esta linea para poder monstrar la notificación de tipo toast que es la que se muestrta en la esquina superior del lado izquierdo de la pantalla, indicando al usuario que a terminado de generarse el Modelo 2 de regresión lineal, y se agrega un icono que representa que termino el modelo de realizarse para que pueda ser visualizado por el usuario

    modelo2 = st.session_state.modelo2 # Esto permite recuperar el modelo que se entreno desde el área de session_state, con el objetivo de poder utilizarlo sin volver a entrenar el modelo 
    res_2 = st.session_state.res_2 # Esto permite recuperar los resultados que determinan la evaluación del Modelo 2 de regresión lineal que son: rmse, mape y r^2 
    x2 = st.session_state.x2 # Esto permite recuperar las variables que son independientes del Modelo 2 de regresión lineal
    y_pred_2 = st.session_state.y_pred_2 # Esto permite recuperar las predicciones realizadas a traves del modelo que se utiliza regresión lineal aplicado a el conjunto de datos de prueba 
    y_test_2 = st.session_state.y_test_2 # Esto permite recuperar los valores reales del conjunto de prueba, con el objetivo de poder compararlos con las predicciones realizadas por el Modelo 2 que aplica regresión lineal

    # ==========================================
    # Modelo 3
    # ==========================================

    if not st.session_state.modelo3_generado: # Esta linea aplica una condicional que va a permitir verificar si el Modelo 3 no ha sido generado de forma previa en la aplicación en el momento que se haya conectado el usuario  

        with st.spinner("Generando Modelo 3..."): # Esta linea permite evidenciar de forma grafica por medio de una animación que se esta cargando, que esto se traduce en que se esta entrenando el Modelo 3  

            time.sleep(5) # Esta linea permite aplicar una simulación de tiempo para que se procese el entrenamiento de los datos adjuntados por el usuario

            datos3 = datos.copy() # Esta linea permite establecer una copia del dataset original subido por el usuario para proceder a realizar transformaciones al aplicar el modelo de Machine Learning

            datos3['imc_alto'] = (
                datos3['imc'] >= 30
            ).astype(int) # Esta linea permite crear una variable binaria que permita indicar si el IMC es alto o igual a 30, para luego ser transformado a un dato de tipo int osea entero

            datos3['fumador_bool'] = (
                datos3['fumador'] == 'si'
            ).astype(int) # Esta linea permite convertir la variable categórica a una variabnle binaria que indica que 0 es una persona que no fuma y 1 es una persona que fuma

            datos3['fumador_imc'] = (
                datos3['fumador_bool'] * datos3['imc']
            ) # Esta linea permite establecer una variable que va a permitir realizar la interacción entre la variable fumador de tipo binaria y la variable IMC de tipo entera que va a ser la operación matematica entre estas dos variables 

            datos3['edad2'] = datos3['edad'] ** 2 # Esta linea permite indicar que esta variable se comporta de manera cuadratica de la edad para poder capturar esa relación no lineal que existen en los datos de edad

            datos3['edad_fumador'] = (
                datos3['edad'] * datos3['fumador_bool']
            ) # Esta linea permite establecer una variable que va a permitir realizar la interacción entre la variable fumador de tipo binaria y la variable edad de tipo entera que va a ser la operación matematica entre estas dos variables 

            datos3['riesgo_metabolico'] = (
                datos3['imc'] * datos3['edad']
            ) / 100 # Esta linea permite determinar el calculo que va a apoder establecer el indicador de riesgo a nivel metabolico esto teniendo en cuenta los factores determinantes que son la edasd y el IMC

            columnas_a_borrar_m3 = ['fumador', 'gastos', 'riesgo_modelo1', 'riesgo_modelo2', 'riesgo'] # Esta linea permite determinar una lista de columnas que se deben eliminar si continuan existiendo 
            columnas_existentes_m3 = [col for col in columnas_a_borrar_m3 if col in datos3.columns] # Esta linea permite establecer un filtro que va a tener como objetivo el filtrar solo las columnas que existan realmente en el DataFrame cargado por el usuario  

            datos3 = pd.get_dummies(
                datos3.drop(columnas_existentes_m3, axis=1),
                drop_first=True
            ) # Esta linea permite eliminar las columnas que sean irrelevantes para el Modelo 3 y aplicar el metodo de one-hot encoding que permite realizar el preprocesamiento de datos que van a convertir esas variables categóricas a un formato numérico

            x3 = datos3 # En esta linea se permite definir cuales van a ser las variables independientes del Modelo 3 

            X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(
                x3,
                y,
                test_size=0.2,
                random_state=42
            ) # Esta linea permite dividir los datos del entrenamiento y los datos de prueba que se van a aplicar en el modelo, esto estableciendo una semilla que en este caso adquiere el valor de 42 y que va a tener como parametros la variable y que es la variable objetivo global y x3 que es la variable que acumula las variables que actuan de forma independiente al Modelo 3

            scaler = StandardScaler() # Esta linea permite establecer un proceso de estandarización de esos datos para poder mejorar el rendimiento del Modelo 3

            X_train3_scaled = scaler.fit_transform(X_train_3) # Esta linea permite realizar un proceso de ajuste a nivel escalar aplicado a ese conjunto de datos de entrenamiento establecidos por medio del parametro determinado por la variable X_train_3 

            X_test3_scaled = scaler.transform(X_test_3) # Esta linea permite aplicar un proceso de transformación de ajuste a nivel escalar aplicado a el conjunto de datos de prueba establecido por medio del parametro determinado por la variable X_test_3

            modelo3 = LinearRegression() # Esta linea permite crear un modelo de regresión lineal como el tercer modelo sugerido para estos datos 

            modelo3.fit(
                X_train3_scaled,
                y_train_3
            ) # Esta linea permite entrenar los datos de entrenamieno por medio del modelo planteado que en este caso es el de regresión lineal que exige los parametros X_train_3 y y_train_3 que son los datos de entrenamiento

            y_pred_3 = modelo3.predict(X_test3_scaled)# Esta linea aplica la predicción sobre los datos de prueba, pero siguiendo el modelo 3 que es de regresión lineal que exige un parametro que es X_test_3 que va a representar los datos de prueba  

            res_3 = evaluacion(
                y_test_3,
                y_pred_3,
                'Modelo 3'
            ) # Esta linea permite llamar a la función declarada como "evaluacion" para poder determinar las metricas para poder determinar como se comporta el modelo para ello se exige los parametros de y_test_3 que representan los datos de prueba, la variable y_pred_3 que son los datos que se aplico el modelo de Machine Learning y por ultimo el nombre del modelo que en este caso es Modelo 3

            st.session_state.modelo3 = modelo3 # Esta linea permite guardar el Modelo 3 permite entrenar por medio la aplicación de la región session_states para poder reutilizar sin generar nuevamente el proceso de entrenamiento  
            st.session_state.res_3 = res_3 # Esta linea permite guardar las métricas que permiten evaluar el Modelo 3 como lo son: rmse, mape y r^2
            st.session_state.x3 = x3 # Esto permite recuperar las variables que son independientes del Modelo 3 de regresión lineal
            st.session_state.y_pred_3 = y_pred_3 # Esto permite recuperar las predicciones realizadas a traves del modelo que se utiliza regresión lineal aplicado a el conjunto de datos de prueba  
            st.session_state.y_test_3 = y_test_3 # Esta linea permite guarda los valores que son reales que se ubicaron en el conjunto de prueba, con el objetivo de ser comparados con los datos predichos por el modelo de regresión lineal aplicado 
            st.session_state.scaler = scaler # Esta linea permite realizar un proceso de guardar un objeto de tipo StandardScaler para proceder a ser usado para aplicar en los datos que se van a utilizar en el Modelo 3

            st.session_state.modelo3_generado = True # Esta linea permite establecer una marca en el modelo se le indica al usuario que ya se a generado el modelo y esto evita el volver a recalcular el modelo 3

        st.toast(
            "✅ Modelo 3 generado correctamente",
            icon="📊"
        ) # Se utiliza esta linea para poder monstrar la notificación de tipo toast que es la que se muestrta en la esquina superior del lado izquierdo de la pantalla, indicando al usuario que a terminado de generarse el Modelo 3 de regresión lineal, y se agrega un icono que representa que termino el modelo de realizarse para que pueda ser visualizado por el usuario

    modelo3 = st.session_state.modelo3 # Esto permite recuperar el modelo que se entreno desde el área de session_state, con el objetivo de poder utilizarlo sin volver a entrenar el modelo
    res_3 = st.session_state.res_3 # Esto permite recuperar los resultados que determinan la evaluación del Modelo 2 de regresión lineal que son: rmse, mape y r^2
    x3 = st.session_state.x3 # Esto permite recuperar las variables que son independientes del Modelo 2 de regresión lineal
    y_pred_3 = st.session_state.y_pred_3 # Esto permite recuperar las predicciones realizadas a traves del modelo que se utiliza regresión lineal aplicado a el conjunto de datos de prueba 
    y_test_3 = st.session_state.y_test_3 # Esto permite recuperar los valores reales del conjunto de prueba, con el objetivo de poder compararlos con las predicciones realizadas por el Modelo 3 que aplica regresión lineal
    scaler = st.session_state.scaler # Esto permite recuperar el objeto de tipo StandardScaler que se va a aplicar un tipo escalar que se va a emplear en los datos subministrados por el usuario 

    # ==========================================
    # EDA
    # ==========================================

    if opcion == "EDA": # Esta linea va a permite verificar que el usuario haya escogido la opción de "EDA"

        st.header("Análisis Exploratorio de Datos") # Esta linea va a evidenciar al usuario un titulo de la sección wue va a realizar el análisis exploratorio de datos aplicado a los datos cargados por el usuario 

        fig = px.histogram( # Esta linea permite crear un grafico de tipo histograma que va a ser interactivo para el usuario efocado en la variable "gastos" 
            datos, # Esta linea permite definer el DataFrame que se va a utilizar para generar la grafica 
            x='gastos', # Esta linea va a permitir generar una variable que va a representar el eje X que va a ser los "gastos"
            nbins=40, # Esta linea permite definir el número de divisiones o barras que van a tener la grafica del histograma 
            marginal='box', # Esta linea permite definir el proceso de agregar un boxplot de forma adicional para poder establecer la visualización, distribución y outliers que va a evidenciarse en la grafica
            color_discrete_sequence=['#3b82f6'], # Esta linea define el color que se va a aplicar a la grafica
            title='Distribución de Gastos Médicos' # Esta linea permite definir el titulo que va a llevar esta grafica
        )

        fig.update_layout( # Esta linea va a permitir poder personalizar el diseño que va a tener la grafica
            template='plotly_dark', # Esta linea va a poder aplicar un tema oscuro a la grafica para poder ser más atractiva para el usuario 
            height=500 # Esta linea va a poder definir esa altura que va a aplicarse a la grafica
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación  

        fig_box = px.box( # Esta linea permite poder realizar el proceso de creación del un boxplot que va a tener como objetivo poder analizar el impacto del tabaquismo con respecto a los gastos medicos 
            datos, # Esta linea permite definer el DataFrame que se va a utilizar para generar la grafica 
            x='fumador', # Esta linea va a permitir generar una variable que va a representar el eje X que va a ser el "fumador"
            y='gastos', # Esta linea va a permitir generar una variable que va a representar el eje Y que va a ser el "gasto"
            color='fumador', # Esta linea va a permitir colorear las cajas según la categoría del fumador
            points='all', # Esta linea va a permitir visualizar todos los puntos de los datos individuales 
            title='Impacto del Tabaquismo', # Esta linea permite definir el titulo que va a llevar esta grafica 
            color_discrete_sequence=['#ef4444', '#10b981'] # Esta linea va a poder permitir definir los colores que van a personalizar la grafica para que se pueda ver atractivamente la grafica para el usario
        )

        fig_box.update_layout( # Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en la grafica de boxplot
            template='plotly_dark', # Esta linea permite aplicar un color oscuro a la grafica para ser entretenida para el usuario
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico 
        )

        st.plotly_chart(
            fig_box,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación   

    # ==========================================
    # Correlaciones
    # ==========================================

    elif opcion == "Correlaciones": # Esta linea va a permite verificar que el usuario haya escogido la opción de "Correlaciones" 

        st.header("Matriz de Correlación") # Esta linea va a evidenciar al usuario un titulo de la sección wue va a realizar la correlaciones que se van a aplicar a las diferentes variables que influyen en los modelos de Machine Learning a los datos cargados por el usuario

        datos_corr = datos.copy() # Esta linea va a ser el poder crear una copia del DataFrame original subministrado por el usuario esto con el objetivo de poder evitar que se modifique los datos originales 

        if 'fumador' in datos_corr.columns and datos_corr['fumador'].dtype.name == 'category': # Esta linea se aplica una condicional que va a permitir verificar si la columna que contiene la variable "fumador" existe o no y si existe se va a declarar como una variable de tipo categorica
            datos_corr['fumador'] = datos_corr['fumador'].map({ # En esta linea se aplica un proceso de conversión de esa variable categoria "fumador" en valores de tipo numéricos para determinar si es fumador es 1 y sino es fumador es 0
                'si': 1,
                'no': 0
            })

        corr = datos_corr.select_dtypes(
            include=[np.number]
        ).corr() # Esta linea permite realizar un proceso de selección de manera unica entre las columnas numéricas y posteriormente se calcula la matriza de correlacción que va a permitir correlacionar las diferentes variables que influyen en los Modelos de Machine Learning

        fig_corr = px.imshow( # Esta linea permite realizar un proceso de creación de un grafico de tipo mapa de calor que va a ser interactivo para el usuario y va a tener como objetivo correlacionar las variables que influyen en los modelos de Machine Learning
            corr, # Esta linea permite definer la variable que va a representar los datos que van a conformar la matriz de correlacción que se calculo 
            text_auto=True, # Esta linea permite mostrar de forma automatica esos valores numéricos en cada celda que se ubicara en la matriz de correlación 
            color_continuous_scale='RdBu_r', # Esta linea permite establecer la definición de la escala de los colores que se van a aplicar a la grafica del mapa de calor
            title='Matriz de Correlación' # Esta linea permite definir el titulo que va a llevar esta grafica  
        )

        fig_corr.update_layout( # Esta linea permite poder realizar una personalización en el diseño de la grafica
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario 
            height=700 # Esta linea permite definir la altura donde se va a ubicar la grafico   
        )

        st.plotly_chart(
            fig_corr,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación    

    # ==========================================
    # Modelo 1
    # ==========================================

    elif opcion == "Modelo 1": # Esta linea va a permite verificar que el usuario haya escogido la opción de "Modelo 1" 

        st.header("Modelo 1") # Esta linea va a permitir establecer un titulo para esa sección que se va a ubicarse el usuario

        st.dataframe(pd.DataFrame([res_1])) # Esta linea va a permitir mostrar las métricas que van a permitir evaluar el modelo en forma de una tabla que condense esa información del Modelo 1

        st.subheader("Coeficientes del Modelo") # Esta linea va a permitir establecer un subtitulo en esa sección que va a demostrarle al usuario los coeficientes

        coef1 = pd.DataFrame({
            "Variable": x1.columns,
            "Coeficiente": modelo1.coef_
        }) # Esta linea va a poder realizar el proceso de creación de un DataFrame con las variables y coeficientes que definen ese Modelo 1

        st.dataframe(coef1) # Esta linea va a poder mostrar esa tabla que define los coeficientes de ese Modelo 1

        fig_pred1 = px.scatter( # Esta linea permite realizar un proceso de creación de la grafica de dispersión que va a tener como objetivo comparar los valores reales con respecto a los valores predichos por medio de aplicar el Modelo 1
            x=y_test_1, # Esta linea permite establecer los valores reales de los gastos medicos con respecto a los datos subministrados por el usuario
            y=y_pred_1, # Esta linea permite establecer los valores predichos por el modelo 1 
            labels={ # Esta linea va a permitir establecer las etiquetas que van a tener los ejes en la grafica creada
                'x': 'Gastos Reales',
                'y': 'Predicción'
            },
            title='Modelo 1 - Reales vs Predichos' # Esta linea permite definir el titulo que va a llevar esta grafica 
        )

        fig_pred1.add_trace( # Esta linea permite agregar un línea diagonal en la grafica de los datos reales con respecto a los datos predichos que va a representar la predicción perfecta
            go.Scatter( 
                x=[y_test_1.min(), y_test_1.max()], # Esta linea permite generar los valores mínimos y máximos que se van a establecer en el eje X
                y=[y_test_1.min(), y_test_1.max()], # Esta linea permite generar los valores mínimos y máximos que se van a establecer en el eje Y
                mode='lines', # Esta linea permite establecer el tipo de grafica que se va a someter que en este caso es una grafica de línea
                name='Predicción Perfecta' # Esta linea permite definir el titulo que va a llevar esta grafica
            )
        )

        fig_pred1.update_layout( # Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en el diseño de la grafica 
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario 
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico
        )

        st.plotly_chart(
            fig_pred1,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación

        # Segmentación Riesgo - Modelo 1
        gastos_predichos_1 = modelo1.predict(x1) # Esta linea permite realizar un proceso de generación de la predicciones enfocadas en los gastos que va a ser utilizadas todas las variables que van a influir en el Modelo 1

        datos['riesgo_modelo1'] = pd.qcut( # Esta linea va a generar una clasificación de aquellos clientes que se van a clasificar en niveles de riesgo según los gastos que fueron predichos por el Modelo 1
            gastos_predichos_1, # Esta linea permite utilizar los datos que se realizo un proceso de segmentación 
            q=[0, 0.4, 0.8, 1.0], # Esta linea permite establecer los cuartiles o divisiones porcentuales de los grupos que van a aplicarse en la grafica
            labels=[ # Esta linea permite realizar la aplicación del etiquetamiento de cada segmento que se analiza en la grafica que se genera
                'Bajo Riesgo',
                'Riesgo Medio',
                'Alto Riesgo'
            ]
        )

        fig_riesgo1 = px.violin( # Esta linea permite realizar un proceso de creación de una grafica de violin que va a tener como objetivo visualizar esa distribución de los gastos por nivel de riesgo que tiene los usuarios que adquieren un gasto medico 
            datos, # Esta linea permite utilizar los datos del DataFrame original
            x='riesgo_modelo1', # Esta linea define la variable que va a ser de tipo categórica que va a aplicarse en el eje X de la grafica
            y='gastos', # Esta line define la variable que va a ser de tipo numérica que va a aplicarse en el eje Y de la grafica
            color='riesgo_modelo1', # Esta linea va a permitir colorear esta grafica de violin teniendo en cuenta según el nivel de riesgo determinado 
            box=True, # Esta linea va a poder mostrar de manera interna el boxplot generado en la grafica
            points='all', # Esta linea va a permitir visualizar todos los puntos de los datos individuales
            title='Segmentación Inteligente de Riesgo - Modelo 1' # Esta linea permite definir el titulo que va a llevar esta grafica
        )

        fig_riesgo1.update_layout( # Esta linea permite poder realizar una personalización en el diseño de la grafica
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico
        )

        st.plotly_chart(
            fig_riesgo1,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación

    # ==========================================
    # Modelo 2
    # ==========================================

    elif opcion == "Modelo 2": # Esta linea va a permite verificar que el usuario haya escogido la opción de "Modelo 2"

        st.header("Modelo 2") # Esta linea va a permitir establecer un titulo para esa sección que se va a ubicarse el usuario

        st.dataframe(pd.DataFrame([res_2])) # Esta linea va a permitir mostrar las métricas que van a permitir evaluar el modelo en forma de una tabla que condense esa información del Modelo 2

        st.subheader("Coeficientes del Modelo") # Esta linea va a permitir establecer un subtitulo en esa sección que va a demostrarle al usuario los coeficientes 

        coef2 = pd.DataFrame({
            "Variable": x2.columns,
            "Coeficiente": modelo2.coef_
        }) # Esta linea va a poder realizar el proceso de creación de un DataFrame con las variables y coeficientes que definen ese Modelo 2

        st.dataframe(coef2) # Esta linea va a poder mostrar esa tabla que define los coeficientes de ese Modelo 2

        fig_pred2 = px.scatter( # Esta linea permite realizar un proceso de creación de la grafica de dispersión que va a tener como objetivo comparar los valores reales con respecto a los valores predichos por medio de aplicar el Modelo 2
            x=y_test_2, # Esta linea permite establecer los valores reales de los gastos medicos con respecto a los datos subministrados por el usuario
            y=y_pred_2, # Esta linea permite establecer los valores predichos por el modelo 2
            labels={ # Esta linea va a permitir establecer las etiquetas que van a tener los ejes en la grafica creada 
                'x': 'Gastos Reales',
                'y': 'Predicción'
            },
            title='Modelo 2 - Reales vs Predichos' # Esta linea permite definir el titulo que va a llevar esta grafica 
        )

        fig_pred2.add_trace( # Esta linea permite agregar un línea diagonal en la grafica de los datos reales con respecto a los datos predichos que va a representar la predicción perfecta
            go.Scatter(
                x=[y_test_2.min(), y_test_2.max()], # Esta linea permite generar los valores mínimos y máximos que se van a establecer en el eje X
                y=[y_test_2.min(), y_test_2.max()], # Esta linea permite generar los valores mínimos y máximos que se van a establecer en el eje Y
                mode='lines', # Esta linea permite establecer el tipo de grafica que se va a someter que en este caso es una grafica de línea
                name='Predicción Perfecta' # Esta linea permite definir el titulo que va a llevar esta grafica
            )
        )

        fig_pred2.update_layout( # Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en el diseño de la grafica
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario 
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico 
        )

        st.plotly_chart(
            fig_pred2,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación

        # Segmentación Riesgo - Modelo 2
        gastos_predichos_2 = modelo2.predict(x2) # Esta linea permite realizar un proceso de generación de la predicciones enfocadas en los gastos que va a ser utilizadas todas las variables que van a influir en el Modelo 2

        datos['riesgo_modelo2'] = pd.qcut(# Esta linea va a generar una clasificación de aquellos clientes que se van a clasificar en niveles de riesgo según los gastos que fueron predichos por el Modelo 2
            gastos_predichos_2, # Esta linea permite utilizar los datos que se realizo un proceso de segmentación 
            q=[0, 0.4, 0.8, 1.0], # Esta linea permite establecer los cuartiles o divisiones porcentuales de los grupos que van a aplicarse en la grafica
            labels=[ # Esta linea permite realizar la aplicación del etiquetamiento de cada segmento que se analiza en la grafica que se genera
                'Bajo Riesgo',
                'Riesgo Medio',
                'Alto Riesgo'
            ]
        )

        fig_riesgo2 = px.violin( # Esta linea permite realizar un proceso de creación de una grafica de violin que va a tener como objetivo visualizar esa distribución de los gastos por nivel de riesgo que tiene los usuarios que adquieren un gasto medico 
            datos, # Esta linea permite utilizar los datos del DataFrame original
            x='riesgo_modelo2', # Esta linea define la variable que va a ser de tipo categórica que va a aplicarse en el eje X de la grafica
            y='gastos', # Esta line define la variable que va a ser de tipo numérica que va a aplicarse en el eje Y de la grafica
            color='riesgo_modelo2', # Esta linea va a permitir colorear esta grafica de violin teniendo en cuenta según el nivel de riesgo determinado 
            box=True, # Esta linea va a poder mostrar de manera interna el boxplot generado en la grafica
            points='all', # Esta linea va a permitir visualizar todos los puntos de los datos individuales
            title='Segmentación Inteligente de Riesgo - Modelo 2' # Esta linea permite definir el titulo que va a llevar esta grafica
        )

        fig_riesgo2.update_layout( # Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en el diseño de la grafica 
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario 
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico 
        )

        st.plotly_chart(
            fig_riesgo2,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación

    # ==========================================
    # Modelo 3
    # ==========================================

    elif opcion == "Modelo 3": # Esta linea va a permite verificar que el usuario haya escogido la opción de "Modelo 3" 

        st.header("Modelo 3")  # Esta linea va a permitir establecer un titulo para esa sección que se va a ubicarse el usuario

        st.dataframe(pd.DataFrame([res_3])) # Esta linea va a permitir mostrar las métricas que van a permitir evaluar el modelo en forma de una tabla que condense esa información del Modelo 3

        st.subheader("Coeficientes del Modelo") # Esta linea va a permitir establecer un subtitulo en esa sección que va a demostrarle al usuario los coeficientes

        coef3 = pd.DataFrame({
            "Variable": x3.columns,
            "Coeficiente": modelo3.coef_
        }) # Esta linea va a poder realizar el proceso de creación de un DataFrame con las variables y coeficientes que definen ese Modelo 3

        st.dataframe(coef3) # Esta linea va a poder mostrar esa tabla que define los coeficientes de ese Modelo 3

        fig_pred3 = px.scatter(# Esta linea permite realizar un proceso de creación de la grafica de dispersión que va a tener como objetivo comparar los valores reales con respecto a los valores predichos por medio de aplicar el Modelo 3
            x=y_test_3, # Esta linea permite establecer los valores reales de los gastos medicos con respecto a los datos subministrados por el usuario
            y=y_pred_3, # Esta linea permite establecer los valores predichos por el modelo 3 
            labels={ # Esta linea va a permitir establecer las etiquetas que van a tener los ejes en la grafica creada
                'x': 'Gastos Reales',
                'y': 'Predicción'
            },
            title='Modelo 3 - Reales vs Predichos' # Esta linea permite definir el titulo que va a llevar esta grafica 
        )

        fig_pred3.add_trace( # Esta linea permite agregar un línea diagonal en la grafica de los datos reales con respecto a los datos predichos que va a representar la predicción perfecta
            go.Scatter(
                x=[y_test_3.min(), y_test_3.max()], # Esta linea permite generar los valores mínimos y máximos que se van a establecer en el eje X
                y=[y_test_3.min(), y_test_3.max()], # Esta linea permite generar los valores mínimos y máximos que se van a establecer en el eje Y
                mode='lines', # Esta linea permite establecer el tipo de grafica que se va a someter que en este caso es una grafica de línea 
                name='Predicción Perfecta' # Esta linea permite definir el titulo que va a llevar esta grafica
            )
        )

        fig_pred3.update_layout( # Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en el diseño de la grafica
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario 
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico 
        )

        st.plotly_chart(
            fig_pred3,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación 

        # Segmentación Riesgo - Modelo 3
        gastos_predichos = modelo3.predict(
            scaler.transform(x3)
        ) # Esta linea permite generar el proceso de predicciones de los gastos medicos aplicando el Modelo 3 esto teniendo en cuenta los datos que se transformaron a tipo StandardScaler

        datos['riesgo'] = pd.qcut( # Esta linea va a generar una clasificación de aquellos clientes que se van a clasificar en niveles de riesgo según los gastos que fueron predichos por el Modelo 3
            gastos_predichos, # Esta linea permite utilizar los datos que se realizo un proceso de segmentación determinados por aplicar el tipo StandardScaler 
            q=[0, 0.4, 0.8, 1.0], # Esta linea permite establecer los cuartiles o divisiones porcentuales de los grupos que van a aplicarse en la grafica
            labels=[ # Esta linea permite realizar la aplicación del etiquetamiento de cada segmento que se analiza en la grafica que se genera
                'Bajo Riesgo',
                'Riesgo Medio',
                'Alto Riesgo'
            ]
        )

        fig_riesgo = px.violin(# Esta linea permite realizar un proceso de creación de una grafica de violin que va a tener como objetivo visualizar esa distribución de los gastos por nivel de riesgo que tiene los usuarios que adquieren un gasto medico 
            datos, # Esta linea permite utilizar los datos del DataFrame original
            x='riesgo', # Esta linea define la variable que va a ser de tipo categórica que va a aplicarse en el eje X de la grafica
            y='gastos', # Esta line define la variable que va a ser de tipo numérica que va a aplicarse en el eje Y de la grafica
            color='riesgo', # Esta linea va a permitir colorear esta grafica de violin teniendo en cuenta según el nivel de riesgo determinado 
            box=True, # Esta linea va a poder mostrar de manera interna el boxplot generado en la grafica
            points='all', # Esta linea va a permitir visualizar todos los puntos de los datos individuales
            title='Segmentación Inteligente de Riesgo - Modelo 3' # Esta linea permite definir el titulo que va a llevar esta grafica
        )

        fig_riesgo.update_layout(# Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en el diseño de la grafica 
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario
            height=600 # Esta linea permite definir la altura donde se va a ubicar la grafico 
        )

        st.plotly_chart(
            fig_riesgo,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación 

    # ==========================================
    # Comparación
    # ==========================================

    elif opcion == "Comparación": # Esta linea va a permite verificar que el usuario haya escogido la opción de "Comparación"  

        st.header("Comparación de Modelos") # Esta linea va a permitir establecer un titulo para esa sección que se va a ubicarse el usuario

        datos_resultados = pd.DataFrame([ # Esta linea va a permitir realizar un proceso de creación de un DataFrame que indica los resultados obtenidos por medio de la evaluación de los tres modelos realizados
            res_1,
            res_2,
            res_3
        ])

        st.dataframe(datos_resultados) # Esta linea va a permitir mostrar la tabla con las metricas de los diferentes modelos de Machine Learning utilizados

        fig_bar = px.bar( # Esta linea permitre realizar un proceso de creación de un grafico de barras que tendran como objetivo comparar el valor de R^2 de cada modelo realizado
            datos_resultados, # Esta linea permite aplicar el DataFrame que indica los resultados obtenidos en los modelos de Machine Learning 
            x='Modelo', # Esta linea permite generar una variable que va a ser monstrada en el eje X de la grafica
            y='R2', # Esta linea permite generar una variable que va a ser monstrada en el eje Y de la grafica
            color='Modelo', # Esta linea genrea colores en la barra según el modelo de Machine Learning aplicado en la aplicación
            text='R2', # Esta linea permite monstrar el valor de R^2 sobre encima de cada barra que representa el modelo de Machine Learning aplicado en esta aplicación
            title='Comparación de R²' # Esta linea permite definir el titulo que va a llevar esta grafica
        )

        fig_bar.update_traces(
            texttemplate='%{text:.3f}'
        ) # Esta linea permite realizar el proceso de configuración del formato del tamñano del texto que va a ser monstrado en las barras que va a visualizar el usuario por medio de la grafica

        fig_bar.update_layout(# Esta linea va a permitir realizar el proceso de configuración que se va a aplicar en el diseño de la grafica 
            template='plotly_dark', # Esta linea permite aplicar un tema oscuro a la grafica para ser entretenida para el usuario
            height=500 # Esta linea permite definir la altura donde se va a ubicar la grafico  
        )

        st.plotly_chart(
            fig_bar,
            use_container_width=True
        ) # Esta linea permite mostrar por medio de la opcion de plotly_chart poder que ese grafico de tipo histograma sea interactivo en la aplicación 

    # ==========================================
    # Simulador
    # ==========================================

    elif opcion == "Simulador": # Esta linea va a permite verificar que el usuario haya escogido la opción de "Simulador"  

        st.header("Simulador Inteligente") # Esta linea va a permitir establecer un titulo para esa sección que se va a ubicarse el usuario 

        if 'mostrar_confirmacion' not in st.session_state: # Esta linea permite verificar que la variable que representa la verificación si la variable se va a confirmar que es "mostrar_confirmacion" no exista en el estado de session_state 
            st.session_state.mostrar_confirmacion = False # Si la variable "mostrar_confirmacion" no existe, entonces la crea y permite asignar a esa variable el valor de False, esto con el objetivo de poder controlar la validación de esa variable 

        if 'simulacion_realizada' not in st.session_state: # Esta linea permite verificar que la variable que representa si la verificación si la variable de simulación realizada existe se va a confirmar que es "simulacion_realizada" no exista en el estado de session_state 
            st.session_state.simulacion_realizada = False # Si la variable "simulacion_realizada" no existe, entonces la crea y permite asignar a esa variable el valor de False, esto con el objetivo de poder controlar la validación de esa variable 

        if 'prima_simulada' not in st.session_state: # Esta linea permite realizar un proceso de verificación de que si existe la variable que va a almacenar la primera simulación que es "prima_simulada" no existe en el estado de session_state
            st.session_state.prima_simulada = None # Esta linea va a determinar esa variable con un valor de tipo None 

        if 'ultima_configuracion' not in st.session_state: # Esta linea permite realizar un proceso de verificación de que si existe la variable para guardar la última configuración que es "ultima_configuracion" no exista en el estado session_state
            st.session_state.ultima_configuracion = None # Esta linea va a determinar esa variable con un valor de tipo None 

        if 'sim_params' not in st.session_state: # Esta linea permite realizar un proceso de verificación de que si existe el diccionario de parametros de simulación que es "sim_params" no existe en el estado session_state
            st.session_state.sim_params = {} # Esta linea va a permitir inicializar una variable que se va a compartar como un diccionario vacío 

        edad = st.slider("Edad", 18, 80, 35) # Esta linea permite generar un botón deslizador, esto con el objetivo de poder seleccionar la edad que el usuario decida llevar a cabo la simulación    
        imc = st.slider("IMC", 15.0, 45.0, 30.0) # Esta linea permite generar un boton deslizador, esto con el objetivo de poder seleccionar el índice de masa corporal (IMC) que el usuario decida llevar a cabo en la simulación
        hijos = st.slider("Número de hijos", 0, 5, 0) # Esta linea permite generar un boton deslizador, esto con el objetivo de poder seleccionar la cantidad de hijos que desee el usuario para llevar a cabo en la simulación
        fumador = st.selectbox("¿Es fumador?", ['si', 'no']) # Esta linea permite generar una caja que va a permitir generar multiples opciones que en este caso es si o no es fumador la persona para que el usuario pueda llevar a cabo la simulación
        sexo = st.selectbox("Sexo", ['hombre', 'mujer']) # Esta linea permite generar una caja que va a permitir generar multiples opciones que en este caso es hombre o mujer para determinar el sexo de la persona para que el usuario pueda llevar a cabo la simulacion  
        region = st.selectbox(
            "Región",
            ['noroccidente', 'nororiente', 'suroccidente', 'suroriente']
        ) # Esta linea permite generar una caja que va a permitir generar multiples opciones que en este caso es noroccidente, nororiente, suroccidente y suoriente para determinar la región de la persona que desee el usuario para llevar a cabo la simulación 

        configuracion_actual = (
            edad,
            imc,
            hijos,
            fumador,
            sexo,
            region
        ) # Esta linea que va a permitir el proceso de guardar esa configuración actual seleccionada por el usuario para llevar a cabo la simulación

        if (
            st.session_state.ultima_configuracion is not None
            and configuracion_actual != st.session_state.ultima_configuracion
        ): # Esta linea permite llevar a cabo un proceso de verificación de que si existe una configuración previa y si no va a realizar cambios aplicados a la simulación 
            st.session_state.mostrar_confirmacion = False # Esta linea permite llevar a cabo el ocultar la confirmación anterior 
            st.session_state.simulacion_realizada = False # Esta linea permite reiniciar el estado de la simulación realizada
            st.session_state.prima_simulada = None # Esta linea permite eliminar el valor previo de la simulación que ha llevado a cabo el usuario

        st.session_state.ultima_configuracion = configuracion_actual # Esta linea permite llevar a cabo un proceso de guardado de esa configuración actual que haya realizado el usuario como la última configuración registra por el usuario para llevar a cabo la simulación

        st.markdown("---") # Esta linea permite agrergar una linea que va a dividir de manera visual la interfaz grafica que va a observar el usuario

        # ==========================================
        # Botón Principal
        # ==========================================

        if not st.session_state.mostrar_confirmacion: # Esta linea permite verificar si aún no se debe monstrar la confirmación de la simulación 

            with st.spinner("Validando filtros inteligentes..."): # Esta linea permite evidenciar de manera visual al usuario la animación de carga mientras se validan los filtros realizados por el usuario que son edad, imc, hijos, fumador, sexo y region
                time.sleep(1) # Esta linea va a permitir llevar a cabo la simulación en un rango de tiempo de 1 segundo para procesar la información que haya escogido el usuario

            st.info("Los filtros fueron configurados correctamente.") # Esta linea indica un mensaje de forma visual al usuario informadole que se ha iniciado que los filtros fueron configurados correctamente para proseguir con la simulación 

            if st.button("Realizar Simulación"): # Esta linea permite crear un botón para poder ejecutar la simulación dandole la opción al usuario de ejecutarla o no la simulación  
                st.session_state.mostrar_confirmacion = True # Esta linea permite activar de manera visual la configuración de la confirmación de la simulación para llevar a cabo la simulación
                st.rerun() # Esta linea permite llevar a cabo un proceso de recarga de la aplicación para poder actualizar la interfaz grafica que va a visualizar el usuario

        # ==========================================
        # Confirmación
        # ==========================================

        if st.session_state.mostrar_confirmacion: # Esta linea permite realizar una condicional que lo que va a realizar es verificar si se debe mostrar la confirmación de realizar la simulación que requiere el usuario realizar

            st.warning(
                "¿Está seguro de ejecutar la simulación con las opciones seleccionadas?"
            ) # Esta linea permite evidenciar de forma visual un mensaje informativo para realizar la confirmación de realizar el usuario la simulación que requiere el usuario realizar

            col1, col2 = st.columns(2) # Esta linea permite establecer la creación de las dos columnas para realizar la organización de los dos botones que va a ejecutar uno la simulación y el otro no la ejecuta la simulación que requiere el usuario
            # ==========================================
            # Botón Sí
            # ==========================================

            with col1: # Esta linea permite definir la funcionalidad del primer botón que va a permitir ejecutar la simulación que requiere el usuario

                if st.button("✅ Sí, ejecutar simulación"): # Esta linea permite realizar la acción condicional de si el usuario presiona el botón va a permitir confirmar y proceder a simular lo que requiere saber el usuario 

                    with st.spinner("Generando simulación inteligente..."): # Esta linea permite ejecutar la animación de carga durante la simulación para reproducir el proceso de simulación que requiere el usuario

                        time.sleep(1) # Esta linea permite realizar la simulación de un tiempo de procesamiento para que se pueda ejecutar la simulación

                        fumador_bool = 1 if fumador == 'si' else 0 # Esta linea permite realizar un proceso de conversión de la variable fumador a un formato binario

                        nuevo_cliente_data = { # Esta linea permite establecer la creación de un diccionario con esas nuevas variables que requiere el nuevo cliente para ser aplicados en la simulación que requiere el usuario
                            'edad': [edad], # Esta linea permite establecer la edad del cliente nuevo a aplicar a la simulación que requiere el usuario
                            'imc': [imc], # Esta linea permite establecer el índice de masa corporal (IMC) del cliente nuevo a aplicar a la simulación que requiere el usuario  
                            'hijos': [hijos], # Esta linea permite establecer el número de hijos del cliente nuevo a aplicar a la simulación que requiere el nuevo cliente para ser aplicado en la simulación que requiere el usuario
                            'imc_alto': [1 if imc >= 30 else 0], # Esta linea permite establecer una variable de tipo binaria que va a indicar si el usuario tiene obesidad o no que se establece al cliente nuevo para ser aplicado en la simulación que requiere el usuario  
                            'fumador_bool': [fumador_bool], # Esta linea permite establecer una variable de tipo binaria que indica si el usuario nuevo es fumador (1) o no es fumador (0) para ser aplicado en la simulación que requiere el usuario
                            'fumador_imc': [fumador_bool * imc], # Esta linea permite establecer una variable que va a realizar una operación matematica que va a ser la multiplicación de la variable entera que es la imc por la variable de tipo binaria que es el fumador esto establecido al cliente nuevo para ser aplicado a la simulación que requiere el usuario  
                            'edad2': [edad**2], # Esta linea permite establecer el cuadrado de la edad indicando el cuadrado de la edad del cliente nuevo a aplicar a la simulación que requiere el usuario
                            'edad_fumador': [edad * fumador_bool], # Esta linea permite establecer una variable que va a realizar una operación matematica que va a ser la multiplicación de la variable entera que es la edad por la variable de tipo binaria que es el fumador esto establecido al cliente nuevo para ser aplicado a la simulación que requiere el usuario  
                            'riesgo_metabolico': [(imc * edad) / 100], # Esta linea permite establecer una variable que va a realizar una operación matematica que va a realizar la operación de la multiplicación de las variables de tipo numéricas del icm por la edad entre 100 esto establecido al cliente nuevo para ser aplicado a la simulación que requiere el usuario
                            'sexo_mujer': [1 if sexo == 'mujer' else 0], # Esta linea permite establecer una variable de tipo binaria para indicar si es hombre (0) o mujer (1) el cliente nuevo  a aplicar a la simulación que requiere el usuario 
                            'region_nororiente': [
                                1 if region == 'nororiente' else 0
                            ], # Esta linea permite establecer una variable de tipo binaria que va a determinar si ese usuario nuevo esta ubicado (1) o no (0) en la región nororiente para ser aplicado en la simulación que requiere el usuario 
                            'region_suroccidente': [
                                1 if region == 'suroccidente' else 0
                            ], # Esta linea permite establecer una variable de tipo binaria que va a determinar si ese usuario nuevo esta ubicado (1) o no (0) en la región suroccidente para ser aplicado en la simulación que requiere el usuario 
                            'region_suroriente': [
                                1 if region == 'suroriente' else 0
                            ], # Esta linea permite establecer una variable de tipo binaria que va a determinar si ese usuario nuevo esta ubicado (1) o no (0) en la región suroriente para ser aplicado en la simulación que requiere el usuario
                            'imc_cat_Obeso (IMC>=30)': [
                                1 if imc >= 30 else 0
                            ] # Esta linea permite establcer una variable de tipo binaria que indique si el nuevo cliente tiene obecidad o no por medio de indicar si la variable numérica imc es mayor o igual a 30 si lo cumple es (1) y sino lo cumple es (0) para ser aplicado en la simulación que requiere el usuario  
                        }

                        nuevo_cliente = pd.DataFrame(
                            nuevo_cliente_data
                        ) # Esta linea establece una nueva variable que va a convertir la variable de tipo diccionario que va a conformar el nuevo DataFrame que va a ser aplicado a la simulación que requiere el usuario

                        for col in x3.columns: # Esta linea permite ejecutar un bucle que va a verificar si faltan columnas que requiere el usuario para establecer el modelo a aplicar en la simulación que requiere el usuario
                            if col not in nuevo_cliente.columns: # Esta linea permite identificar si una columna no existe, por lo tanto se crea con el valor de cero esa columna que va a aplicarse en el modelo a aplicar en la simulación que requiere el usuario 
                                nuevo_cliente[col] = 0

                        nuevo_cliente = nuevo_cliente[x3.columns] # Esta linea permite realizar el proceso de reordenamiento de las columnas según el orden esperado por el modelo que se aplica para la simulación que requiere el usuario

                        nuevo_cliente_scaled = scaler.transform(
                            nuevo_cliente
                        ) # Esta linea permite realizar un proceso de escalamiento en los datos aplicando el metodo StandardScaler que van a ser utilizados en el modelo que aplica en la simulación que requiere el usuario

                        gasto_predicho = modelo3.predict(
                            nuevo_cliente_scaled
                        )[0] # Esta linea permite realizar solamente la predicción de la variable gastos con respecto a los parametros registrados en la variable nuevo_cliente_scaled que define los parametros que se van a subministrar a la simulación del nuevo cliente

                        prima = gasto_predicho * 1.15 # Esta linea permite realizar la operación matematica de la prima aplicada en este ejercico de un 15% de forma adicional al cliente nuevo aplicado al simulador que determina el gasto del nuevo cliente

                        st.session_state.prima_simulada = prima # Esta linea permite realizar un proceso de guardado de la variable prima que se calculo con respecto al 15% en el área de session_state para agruparla con respecto a la predicho del nuevo cliente utilizando el modelo aplicado en la simluación del gasto

                        st.session_state.simulacion_realizada = True # Esta linea permite realizar la verificación de que la simulación a sido realizada con exito por lo que se pudo determinar el gasto del cliente nuevo de acuerdo a los parametros definidos

                        st.session_state.sim_params = {
                            "Edad": edad,
                            "IMC": imc,
                            "Hijos": hijos,
                            "Fuma": fumador,
                            "Sexo": sexo,
                            "Region": region
                        } # Esta linea permite guardar esos parametros que definio al nuevo cliente para ser aplicados en el modelo de simulación para determinar ese gasto

                    st.session_state.mostrar_confirmacion = False # Esta linea va a realizar la acción de ocultar la confirmación esto con base en que ya se ejecuto la simulación con total exito para determinar el gasto de ese nuevo cliente teniendo en cuenta los parametros definidos

                    st.toast(
                        "La simulación fue generada correctamente",
                        icon="📊"
                    ) # Se utiliza esta linea para poder monstrar la notificación de tipo toast que es la que se muestrta en la esquina superior del lado izquierdo de la pantalla, indicando al usuario que a terminado de generarse la simulación que determina el gasto del nuevo cliente, y se agrega un icono que representa que termino la simulación de realizarse para que pueda ser visualizado por el usuario 

                    st.rerun() # Esta linea permite establecer el retorno de la recarga de la aplicación para poder llevar a cabo la actualización de los reultados que se puedan obtener dejando las mismos parametros o cambiando esos parametros por otros

            # ==========================================
            # Botón No
            # ==========================================

            with col2: # Esta linea permite definir la funcionalidad del segundo botón que va a permitir no ejecutar la simulación que requiere el usuario para que el usuario pueda cambiar algún parametro para luego ser presionado nuevamente el botón de simular para darle despues al botón de si simular

                if st.button("❌ No, volver"): # Esta linea permite realizar la acción condicional de si el usuario presiona el botón va a no permitir confirmar y proceder a corregir los parametros que vea conveniente y despues simular por medio del botón de simulación para darle despues al botón de si simular para llevar a cabo la simulación que requiere saber el usuario

                    st.session_state.mostrar_confirmacion = False # Esta linea permite realizar el proceso de ocultar el mensaje de confirmación para poder cambiar los aparametros si requiere el usuario para proceder a simular por medio de aplicar el simulador 

                    st.rerun() # Esta linea permite establecer el retorno de la recarga de la aplicación para poder llevar a cabo la actualización de los reultados que se puedan obtener dejando las mismos parametros o cambiando esos parametros por otros 

        # ==========================================
        # Resultado
        # ==========================================

        if (
            st.session_state.simulacion_realizada
            and st.session_state.prima_simulada is not None
        ): # Esta linea permite realizar un proceso de verificación de que se haya seleccionado el botón de simulación que va a llevar a cabo la simulación del gasto del nuevo cliente, e informarle al usuario que ya a sido realizada y que existe una prima del 15% calculada que se le adiciono al gasto predicho por el simulador aplicando el modelo correspondiente  

            st.success(
                f"Prima Anual Sugerida: "
                f"${st.session_state.prima_simulada:,.2f}"
            ) # Esta linea permite visualizar al usuario un mensaje de que se a realizado con total exito la simulación del valor de la prima anual sugerida para ese nuevo cliente con base en los parametros definidos

    # ==============================================================================
    # REPORTE INTELIGENTE CON GRÁFICAS Y ANÁLISIS GENERADO POR IA Y CONTROL DE FLUJO
    # ==============================================================================

    elif opcion == "Reporte": # Verifica si el usuario ha seleccionado la opción de "Reporte" en la barra de navegación lateral

        st.header("Reporte Inteligente de Negocios y Salud") # Despliega el título principal de la sección en la interfaz de Streamlit

        st.markdown("""
        ### Análisis Ejecutivo Médico y Financiero Personalizado
        Este módulo consolida la información del Análisis Exploratorio (EDA), la evaluación de modelos de Machine Learning y su simulación para exportarlos a un documento PDF profesional.
        """) # Muestra una breve introducción en Markdown para explicarle al usuario el alcance del documento final

        # Verificar si existen los datos de simulación en el Session State de Streamlit
        simulacion_disponible = st.session_state.get('simulacion_realizada', False) # Extrae el estado booleano para validar si el usuario pasó previamente por el simulador
        
        if not simulacion_disponible: # Condicional en caso de que no se registre ninguna simulación previa en la sesión actual
            st.warning("**Nota:** No has realizado ninguna simulación en la pestaña 'Simulador'. Si generas el reporte ahora, no se incluirán tus proyecciones personalizadas.") # Lanza una advertencia visual preventiva en la interfaz

        if st.button("Generar Reporte Completo con IA y Gráficas"): # Instancia el botón de ejecución principal para iniciar la compilación del reporte interactivo
            with st.spinner("Analizando métricas, dibujando gráficos y estructurando su PDF..."): # Activa una animación de carga (spinner) mientras se procesan los datos de fondo

                # ------------------------------------------------------------------
                # 1. LLAMADA A LA API DE GROQ PARA ANÁLISIS TEXTUAL DE LA POBLACIÓN GENERAL
                # ------------------------------------------------------------------
                gasto_promedio = datos['gastos'].mean() # Calcula el promedio aritmético de la columna de costos médicos del dataset
                gasto_maximo = datos['gastos'].max() # Extrae el valor máximo histórico registrado en los gastos de salud
                edad_promedio = datos['edad'].mean() # Obtiene la media de edad de la población contenida en los datos adjuntos
                imc_promedio = datos['imc'].mean() # Obtiene el promedio del Índice de Masa Corporal de todos los registros
                porcentaje_fumadores = ((datos['fumador'] == 'si').mean()) * 100 # Determina la tasa porcentual de pacientes con hábito de tabaquismo activo
                pacientes_alto_riesgo = ((datos['imc'] >= 30) & (datos['fumador'] == 'si')).sum() # Cuantifica los casos críticos acumulados (comorbilidad de obesidad y tabaquismo)

                client = Groq(api_key=st.secrets["GROQ_API_KEY"]) # Inicializa el cliente oficial de Groq utilizando las credenciales seguras del archivo secrets de Streamlit

                prompt_sistema = (
                    "Eres un consultor experto en analítica de datos médicos y gestión de riesgos de salud. "
                    "Genera reportes ejecutivos limpios, profesionales y empáticos para el paciente."
                ) # Define el comportamiento e identidad del LLM para asegurar un tono técnico, ejecutivo y asistencial

                prompt_usuario = f"""
                Analiza los siguientes indicadores de nuestra población y genera un breve resumen ejecutivo:
                - Gasto Médico Promedio: ${gasto_promedio:,.2f}
                - Edad Promedio: {edad_promedio:.1f} años
                - Porcentaje de Fumadores: {porcentaje_fumadores:.1f}%
                - Pacientes Críticos (Obeso + Fumador): {pacientes_alto_riesgo}
                Escribe directamente 2 párrafos concisos con conclusiones de salud preventiva. Sin preámbulos.
                """ # Construye el prompt dinámico inyectando las métricas globales calculadas a partir del DataFrame de la población

                try: # Bloque de control de excepciones para mitigar caídas por problemas de red o cuotas en el servicio de la API
                    chat_completion = client.chat.completions.create( # Realiza la solicitud síncrona de generación de texto a los servidores de Groq
                        messages=[ # Estructura los roles jerárquicos requeridos por el backend de chat de Groq
                            {"role": "system", "content": prompt_sistema}, # Envía las instrucciones de contexto y rol del sistema
                            {"role": "user", "content": prompt_usuario} # Envía la solicitud y métricas específicas del usuario
                        ],
                        model="llama-3.3-70b-versatile", # Especifica el modelo Llama de alto rendimiento para el análisis interpretativo
                        temperature=0.3, # Configura una temperatura baja para forzar respuestas altamente deterministas, precisas y sin alucinaciones
                    )
                    texto_ia = chat_completion.choices[0].message.content # Almacena el resultado textual retornado por la inteligencia artificial
                except Exception as e: # Captura cualquier fallo de conexión, clave inválida o timeout de la API
                    texto_ia = "Análisis automático: Los factores de estilo de vida como el tabaquismo y el índice de masa corporal elevado representan los mayores catalizadores de costo en el fondo médico. Se sugieren programas preventivos corporativos y clínicos urgentes." # Asigna un bloque de texto predeterminado de respaldo (fallback) para que el PDF no falle en la generación

                # ------------------------------------------------------------------
                # 2. CONFIGURACIÓN E INICIALIZACIÓN DEL DOCUMENTO PDF (REPORTLAB)
                # ------------------------------------------------------------------
                buffer = BytesIO() # Crea un objeto de flujo de bytes en memoria RAM para alojar el binario del PDF sin escribir archivos locales en el servidor
                doc = SimpleDocTemplate( # Instancia la plantilla base del documento PDF definiendo dimensiones, tamaño y márgenes periféricas
                    buffer, 
                    pagesize=letter, # Configura las dimensiones del reporte bajo el estándar físico de tamaño carta
                    rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40 # Reduce los márgenes por defecto a 40 puntos para optimizar el espacio visual útil
                )
                
                styles = getSampleStyleSheet() # Recupera la hoja de estilos nativa y predeterminada proporcionada por ReportLab
                
                # Definición de Estilos Personalizados utilizando códigos Hexadecimales y espaciados controlados
                style_title = ParagraphStyle(
                    'DocTitle', parent=styles['Heading1'], fontSize=18, leading=22,
                    textColor=colors.HexColor("#0f172a"), spaceAfter=15, alignment=1
                ) # Configura el diseño estético para el título central del reporte (Color pizarra oscuro, centrado y con interlineado proporcional)
                
                style_heading = ParagraphStyle(
                    'DocHeading', parent=styles['Heading2'], fontSize=13, leading=16,
                    textColor=colors.HexColor("#1e3a8a"), spaceBefore=14, spaceAfter=6
                ) # Diseña los encabezados de las subsecciones utilizando un tono azul corporativo institucional
                
                style_body = ParagraphStyle(
                    'DocBody', parent=styles['Normal'], fontSize=9.5, leading=13.5,
                    textColor=colors.HexColor("#334155"), spaceAfter=10
                ) # Modula el cuerpo de texto general con fuentes estilizadas de alta legibilidad para el usuario final
                
                style_analysis = ParagraphStyle(
                    'DocAnalysis', parent=styles['Normal'], fontSize=9, leading=13,
                    textColor=colors.HexColor("#0f766e"), backColor=colors.HexColor("#f0fdfa"),
                    borderColor=colors.HexColor("#0d9488"), borderWidth=1, borderPadding=8, spaceAfter=12
                ) # Diseña el contenedor destacado para las cajas de análisis (Fondo verde menta suave, texto verde teal y borde sólido de 1 punto)

                story = [] # Inicializa la lista contenedora ("Story") que acumulará secuencialmente los elementos que integrarán el cuerpo del PDF

                # Construcción del bloque de encabezado institucional en la estructura del PDF
                story.append(Paragraph("<b>REPORTE CLÍNICO-ASISTENCIAL Y DE RIESGO FINANCIERO</b>", style_title)) # Agrega el título en negrita estilizado al flujo del documento
                story.append(Paragraph(f"<b>Fecha de Emisión:</b> {time.strftime('%d/%m/%Y')} | <b>Análisis:</b> Machine Learning Predictivo", style_body)) # Estampa dinámicamente la fecha actual del sistema y el enfoque metodológico
                story.append(Spacer(1, 10)) # Genera un espaciador vertical de 10 puntos para evitar superposición de elementos gráficos

                # ------------------------------------------------------------------
                # 3. GENERACIÓN E INSERCIÓN DE GRÁFICAS DE MATPLOTLIB / SEABORN
                # ------------------------------------------------------------------
                
                def matplotlib_to_image(fig): # Declara una función interna reutilizable para codificar objetos de figuras de Matplotlib en formatos legibles por ReportLab
                    img_buf = BytesIO() # Crea un buffer temporal secundario en memoria exclusivo para la imagen PNG de la gráfica
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=150) # Guarda la figura de Matplotlib en el buffer temporal, recortando bordes muertos e incrementando la densidad a 150 DPI para nitidez de impresión
                    img_buf.seek(0) # Retorna el puntero de lectura del buffer al byte inicial de la estructura binaria
                    plt.close(fig) # Libera de la memoria RAM del servidor el objeto de la figura para evitar fugas de rendimiento en ejecuciones concurrentes
                    return Image(img_buf, width=420, height=180) # Retorna un objeto Image nativo de ReportLab parametrizado con dimensiones fijas y escaladas

                # --- Gráfica 1: Análisis Exploratorio de Datos (EDA) ---
                story.append(Paragraph("1. Análisis Exploratorio de Datos (EDA) - Distribución de Costos", style_heading)) # Inserta el título de la sección 1 en el flujo del PDF
                fig, ax = plt.subplots(figsize=(6, 2.5)) # Crea la figura y los ejes con un tamaño compacto adaptado al flujo de páginas del reporte
                sns.histplot(datos['gastos'], color='#3b82f6', kde=True, ax=ax) # Dibuja el histograma de gastos utilizando Seaborn con una curva de estimación de densidad suavizada
                ax.set_title("Frecuencia de Gastos en la Población", fontsize=10, color="#1e293b") # Define el título interno de la gráfica con fuentes controladas
                ax.set_xlabel("Gastos Anuales ($)", fontsize=8) # Define la etiqueta descriptiva del eje horizontal
                ax.set_ylabel("Cantidad de Pacientes", fontsize=8) # Define la etiqueta cuantitativa del eje vertical
                story.append(matplotlib_to_image(fig)) # Llama al convertidor e inserta la imagen binaria resultante dentro de la lista contenedora del PDF

                analisis_eda = (
                    "<b>Análisis para el Paciente:</b> Esta gráfica revela cómo se distribuyen los costos médicos generales. "
                    "La mayoría de las personas se sitúan en la zona baja (izquierda), reflejando gastos médicos rutinarios. "
                    "Sin embargo, existe una 'cola larga' hacia la derecha que representa casos de contingencias severas de salud. "
                    "Para usted, esto significa que el riesgo financiero no es constante y que un imprevisto de salud sin el resguardo "
                    "de una prima correcta puede elevar exponencialmente sus gastos de bolsillo de un momento a otro."
                ) # Redacta la explicación conceptual de los comportamientos de asimetría estadística para el usuario no técnico
                story.append(Paragraph(analisis_eda, style_analysis)) # Transforma la cadena de texto en un objeto Paragraph y le aplica la caja verde contenedora

                # --- Gráfica 2: Comparación de Desempeño de Modelos ---
                story.append(Paragraph("2. Rendimiento y Comparación de Modelos Matemáticos", style_heading)) # Añade la cabecera correspondiente al bloque de evaluación de Machine Learning
                datos_resultados = pd.DataFrame([res_1, res_2, res_3]) # Estructura un sub-DataFrame unificando los diccionarios de evaluación devueltos por cada modelo entrenado
                fig, ax = plt.subplots(figsize=(6, 2.2)) # Instancia el lienzo gráfico ajustando la altura para evitar saltos de página huérfanos
                sns.barplot(data=datos_resultados, x='Modelo', y='R2', palette=['#94a3b8', '#60a5fa', '#10b981'], ax=ax) # Dibuja un gráfico de barras comparativo de las métricas R-Cuadrado asignando colores temáticos (Gris, Azul, Verde)
                ax.set_title("Métrica de Precisión R² por Modelo", fontsize=10) # Rotula el gráfico evaluativo principal
                ax.set_ylabel("Porcentaje de Precisión (R²)", fontsize=8) # Asigna nombre de métrica al eje vertical de desempeño
                
                for p in ax.patches: # Itera individualmente sobre cada una de las barras dibujadas en el eje para extraer sus propiedades geométricas
                    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height() - 0.1),
                                ha='center', va='center', color='white', fontweight='bold', fontsize=9) # Inyecta una etiqueta de texto sobre el centro de cada barra mostrando el valor numérico exacto de precisión con tres decimales
                story.append(matplotlib_to_image(fig)) # Procesa la figura y la indexa al documento PDF

                analisis_comp = (
                    "<b>Análisis para el Paciente:</b> El gráfico de barras compara la precisión de tres inteligencias algorítmicas diferentes. "
                    "El <b>Modelo 3</b> es el rotundo ganador con un coeficiente cercano al 86% ($R^2 = 0.864$). "
                    "Este indicador le da a usted la certeza y tranquilidad de que las tarifas y proyecciones de su salud no se calculan "
                    "al azar o bajo promedios generales injustos, sino mediante una fórmula matemática avanzada optimizada para comprender su caso específico."
                ) # Describe de manera simplificada e intuitiva el significado técnico del coeficiente de determinación R2
                story.append(Paragraph(analisis_comp, style_analysis)) # Inserta el bloque explicativo formateado al flujo

                # --- Gráfica 3: Calibración / Dispersión del Modelo Ganador ---
                story.append(Paragraph("3. Calibración del Modelo Ganador (Modelo 3 con Ingeniería de Variables)", style_heading)) # Inserta el título explicativo de la gráfica de correlación residual
                fig, ax = plt.subplots(figsize=(6, 2.5)) # Crea la infraestructura del lienzo gráfico
                ax.scatter(y_test_3, y_pred_3, color='#10b981', alpha=0.6, edgecolors='none', s=25) # Proyecta un gráfico de dispersión (Scatter Plot) vinculando los valores objetivos reales contra las predicciones del algoritmo con opacidad moderada
                ax.plot([y_test_3.min(), y_test_3.max()], [y_test_3.min(), y_test_3.max()], color='#ef4444', linestyle='--', linewidth=2) # Traza la diagonal de referencia de predicción perfecta (Identidad matemática de error cero)
                ax.set_title("Gastos Reales vs. Predicciones del Modelo", fontsize=10) # Configura el título interno para validación visual de varianza
                ax.set_xlabel("Gasto Real del Paciente ($)", fontsize=8) # Rotula el eje de valores empíricos observados
                ax.set_ylabel("Gasto Predicho por el Algoritmo ($)", fontsize=8) # Rotula el eje de estimaciones numéricas calculadas por la regresión
                story.append(matplotlib_to_image(fig)) # Realiza el renderizado gráfico final de Matplotlib y lo inserta al flujo del PDF

                analisis_ganador = (
                    "<b>Análisis para el Paciente:</b> En esta gráfica de dispersión, la línea roja discontinua representa una predicción perfecta. "
                    "Al observar que los puntos verdes se agrupan de forma compacta y fiel alrededor de la línea, se evidencia el nulo margen de "
                    "arbitrariedad del sistema. Al paciente le muestra que el modelo es altamente equitativo: si sus hábitos son saludables, "
                    "el sistema lo ubicará con exactitud en la parte baja de la gráfica, protegiéndolo de cobros excesivos artificiales."
                ) # Documenta la lectura visual del scatter plot traduciéndolo en métricas de equidad y precisión tarifaria para el usuario
                story.append(Paragraph(analisis_ganador, style_analysis)) # Agrega el componente de texto analítico al flujo

                # ------------------------------------------------------------------
                # 4. SECCIÓN EVALUATIVA DINÁMICA DEL SIMULADOR INDIVIDUALIZADO
                # ------------------------------------------------------------------
                story.append(Paragraph("4. Resultados de la Simulación Inteligente Individualizada", style_heading)) # Establece la cabecera del bloque personalizado

                if simulacion_disponible: # Condicional activa si existen datos reales capturados desde el simulador en la sesión del usuario
                    sim = st.session_state.get("sim_params", None) # Extrae el diccionario con los parámetros físicos configurados en los inputs de simulación
                    prima_final = st.session_state.get("prima_simulada", None) # Extrae el costo numérico de la prima estimada por el Modelo 3

                    # Construcción y estructuración de la matriz de datos biológicos para la tabla del PDF
                    tabla_datos = [
                        [Paragraph("<b>Variable Evaluada</b>", style_body), Paragraph("<b>Valor Ingresado</b>", style_body)], # Define la fila de encabezado aplicando etiquetas HTML internas
                        ["Edad del Paciente", f"{sim['Edad']} años"], # Registra de forma dinámica la edad del usuario simulado
                        ["Índice de Masa Corporal (IMC)", f"{sim['IMC']} kg/m²"], # Registra el IMC ingresado agregando las unidades métricas correspondientes
                        ["Hijos dependientes", f"{sim['Hijos']}"], # Muestra la carga familiar seleccionada
                        ["Hábito de Tabaquismo", f"{sim['Fuma'].upper()}"] # Formatea y transforma en mayúsculas sostenidas la variable binaria de tabaquismo
                    ]

                    t = Table(tabla_datos, colWidths=[200, 220]) # Instancia la tabla de ReportLab definiendo anchos estáticos por columnas para evitar desbordamientos de página
                    t.setStyle(TableStyle([ # Aplica configuraciones estructurales y estéticas de ReportLab a la matriz tabular de datos
                        ('BACKGROUND', (0,0), (1,0), colors.HexColor("#f1f5f9")), # Pinta el fondo del encabezado con un gris pizarra claro moderno
                        ('BOTTOMPADDING', (0,0), (-1,-1), 4), # Define márgenes internos inferiores reducidos para compactar la tabla
                        ('TOPPADDING', (0,0), (-1,-1), 4), # Configura márgenes internos superiores de 4 puntos
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#e2e8f0")), # Dibuja las rejillas internas divisorias con un gris sutil
                    ]))
                    story.append(t) # Añade la tabla estructurada al cuerpo del documento PDF
                    story.append(Spacer(1, 6)) # Inserta una pequeña holgura vertical previa al cálculo de precios
                    story.append(Paragraph(f"<b>PRIMA ANUAL SUGERIDA PROYECTADA: ${prima_final:,.2f}</b>", styles['Heading3'])) # Renderiza de forma destacada el valor monetario final formateado con separadores de miles y dos decimales

                    # --- INICIO DEL MOTOR DE LOGICA LOGÍSTICO/CLÍNICA DINÁMICA ---
                    if sim['Fuma'].lower() == 'si': # Condicional de riesgo para pacientes fumadores activos
                        analisis_tabaquismo = (
                            "El hábito del tabaquismo representa el multiplicador de riesgo financiero más crítico "
                            "en su evaluación, elevando de forma severa la proyección de la prima debido a la "
                            "correlación directa con patologías cardiovasculares y respiratorias de alto costo."
                        )
                    else: # Condicional de resguardo para perfiles no fumadores
                        analisis_tabaquismo = (
                            "La ausencia de hábito tabáquico actúa como un factor protector clave en su perfil, "
                            "evitando los recargos más severos del modelo predictivo y estabilizando su proyección financiera."
                        )

                    if sim['IMC'] >= 30: # Evaluación clínica orientada a rangos de Obesidad clínica según la OMS
                        analisis_imc = (
                            f"Su IMC de {sim['IMC']} kg/m² se clasifica en rango de obesidad. Al cruzarse con los "
                            "algoritmos de riesgo, este indicador incrementa la probabilidad de siniestralidad metabólica, "
                            "lo que explica el ajuste al alza en la prima sugerida para garantizar la cobertura de contingencias."
                        )
                    elif sim['IMC'] >= 25: # Evaluación clínica ajustada para perfiles con Sobrepeso moderado
                        analisis_imc = (
                            f"Su IMC de {sim['IMC']} kg/m² indica sobrepeso moderado. El sistema detecta una tendencia "
                            "de riesgo incipiente, sugiriendo un monitoreo preventivo para evitar que el indicador escale "
                            "a rangos que penalicen con mayor fuerza el costo de su cobertura."
                        )
                    else: # Diagnóstico automatizado de Normopeso (Rango saludable y óptimo)
                        analisis_imc = (
                            f"Su IMC de {sim['IMC']} kg/m² se encuentra en un rango saludable. Este equilibrio "
                            "metabólico es valorado positivamente por el modelo predictivo, manteniéndolo alejado de "
                            "los umbrales de recargo por riesgo derivado del peso."
                        )

                    info_edad = (
                        f"A los {sim['Edad']} años, el modelo integra el factor de desgaste cronológico estándar. "
                        f"La inclusión de {sim['Hijos']} hijo(s) dependiente(s) distribuye estratégicamente la carga "
                        f"del riesgo familiar dentro del cálculo actuarial de la póliza."
                    ) # Concatena las variables demográficas e individuales para justificar los coeficientes lineales de la regresión

                    if sim['Fuma'].lower() == 'si' and sim['IMC'] >= 30: # Intersección lógica de riesgo extremo por comorbilidades cruzadas
                        conclusions_preventiva = (
                            "<b>Recomendación de Alto Riesgo:</b> La combinación de tabaquismo y obesidad sitúa su perfil "
                            "en la categoría de máxima prioridad médica. Modificar cualquiera de estos dos factores "
                            "no solo transformará radicalmente su expectativa de vida, sino que reducirá drásticamente "
                            "el costo proyectado de sus primas en futuras simulaciones."
                        )
                    else: # Recomendación general de mantenimiento preventivo para perfiles estables o de bajo riesgo acumulado
                        conclusions_preventiva = (
                            "<b>Plan de Acción:</b> Se recomienda mantener o adoptar hábitos orientados a la actualización "
                            "y estabilización de sus métricas actuales. El control continuo de sus variables es la estrategia más efectiva "
                            "para mitigar la volatilidad financiera en la gestión de su salud personal."
                        )

                    # Compila todas las variables validadas en una única cadena, usando <br/><br/> para forzar el quiebre de renglón dentro de ReportLab
                    analisis_simulador = (
                        f"<b>Análisis Clínico-Financiero Personalizado:</b><br/><br/>"
                        f"{analisis_tabaquismo}<br/><br/>"
                        f"{analisis_imc}<br/><br/>"
                        f"{info_edad}<br/><br/>"
                        f"{conclusions_preventiva}"
                    )
                    # --- FIN DEL MOTOR DE LOGICA DINÁMICA ---

                    story.append(Paragraph(analisis_simulador, style_analysis)) # Inserta el bloque diagnóstico personalizado aplicando la envoltura visual destacada

                else: # Bloque de contingencia si el usuario salta directo al reporte omitiendo la simulación individual
                    story.append(Paragraph("<i>No se registran simulaciones personalizadas en esta sesión. Se utilizó un perfil de control estandarizado. Visite la pestaña 'Simulador' para parametrizar sus datos.</i>", style_body)) # Notifica formalmente la ausencia de parámetros individuales en el cuerpo del PDF

                # ------------------------------------------------------------------
                # 5. CONCLUSIONES ESTRATÉGICAS EMITIDAS POR LA IA DE GROQ
                # ------------------------------------------------------------------
                story.append(Paragraph("5. Conclusiones Clínicas y Financieras del Consultor (IA Groq)", style_heading)) # Añade la sección final de síntesis automatizada por el modelo de lenguaje

                for linea in texto_ia.split('\n'): # Fragmenta el string masivo devuelto por Groq mediante el delimitador de salto de línea estándar de Python
                    if linea.strip(): # Valida y descarta cualquier línea residual vacía generada por los saltos de párrafo redundantes del LLM
                        story.append(Paragraph(linea, style_body)) # Inserta secuencialmente cada fragmento de la conclusión de la IA como un párrafo independiente alineado al estilo base

                # ------------------------------------------------------------------
                # 6. COMPILACIÓN FISICA DEL DOCUMENTO PDF EN MEMORIA
                # ------------------------------------------------------------------
                doc.build(story) # Ejecuta el motor interno de ReportLab, procesando secuencialmente el arreglo "story", calculando saltos de página automáticos y generando la estructura binaria del PDF
                pdf_generado = buffer.getvalue() # Recupera la secuencia completa de bytes resultantes almacenados en el flujo de memoria RAM
                
                st.session_state.pdf_reporte = pdf_generado # Preserva el flujo binario del archivo PDF dentro del estado persistente de la sesión de Streamlit
                st.session_state.reporte_generado = True # Actualiza la bandera booleana para registrar que el documento ha concluido su compilación de manera óptima
                st.success("¡Reporte estratégico con gráficos analizados generado exitosamente!") # Muestra una alerta verde de éxito rotundo en la interfaz web del usuario

        # ------------------------------------------------------------------
        # 7. INTERFAZ Y RENDERIZADO DEL BOTÓN DE DESCARGA ASÍNCRONO
        # ------------------------------------------------------------------
        if st.session_state.get('reporte_generado', False): # Evalúa constantemente el estado de la sesión para comprobar si ya existe un reporte listo para su consumo
            st.download_button( # Despliega el widget oficial para transferir archivos binarios desde el servidor hacia el cliente
                label="📥 Descargar Reporte en PDF", # Define la etiqueta y el emoji descriptivo que se visualizará dentro del botón
                data=st.session_state.pdf_reporte, # Vincula el flujo binario del PDF guardado en el Session State como archivo de descarga
                file_name="Reporte_Inteligente_Gastos_Medicos.pdf", # Asigna el nombre de guardado predeterminado que tendrá el documento en el equipo local del usuario
                mime="application/pdf" # Establece el tipo MIME oficial del protocolo HTTP para forzar el reconocimiento y apertura del archivo como PDF legítimo
            )