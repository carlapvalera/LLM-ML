
import streamlit as st
from streamlit_chat import message
from streamlit.components.v1 import html
from streamlit_chat import message
from start import LLM
from dotenv import load_dotenv
import re
import os
# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener el valor de la variable de entorno
api_key = os.getenv('FIREWORKS_API')

# Crear instancia del LLM
llm = LLM(api_key=api_key)


def on_input_change():
    user_input = st.session_state.user_input
    st.session_state.past.append(user_input)
   # st.session_state.generated.append({'type':"normal","data":"The messages from Bot\nWith new line"})
    # Llamar al llm y guardarlo en los diccionarios para volver a construir la pagina
    st.session_state.generated.append({'type':"normal","data":llm.send_query(user_input)})
def on_btn_click():
    del st.session_state.past[:]
    del st.session_state.generated[:]
    
    


st.session_state.setdefault(
    'past', 
    [None]
    #['Hola en que puedo ayudarte?']
)
st.session_state.setdefault(
    'generated', [{'type': 'normal', 'data':None},]
   # [{'type': 'normal', 'data': 'Line 1 \n Line 2 \n Line 3'},]
   #  {'type': 'normal', 'data': f'<audio controls src="{audio_path}"></audio>'}, 
    # {'type': 'table', 'data': f'{table_markdown}'}]
)

# Iniciar conversacion
st.title("Chat placeholder")
message("Hola en que puedo ayudarte?")

chat_placeholder = st.empty()

with chat_placeholder.container():    
    for i in range(1,len(st.session_state['generated'])):                
        message(st.session_state['past'][i], is_user=True, key=f"{i}_user")
        message(
            st.session_state['generated'][i]['data'], 
            key=f"{i}", 
            allow_html=True,
            is_table=True if st.session_state['generated'][i]['type']=='table' else False
        )
    
    st.button("Clear message", on_click=on_btn_click)

with st.container():
    st.text_input("User Input:", on_change=on_input_change, key="user_input")