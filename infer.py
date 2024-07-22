import streamlit as st
import os; import time

st.title('Abhi Micro Med LM')
os.environ['KAGGLE_USERNAME'] = st.secrets['kaggle_username']
os.environ['KAGGLE_KEY'] = st.secrets['kaggle_key']
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'
import keras_nlp

# initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# load the model once and use it across all users and sessions
@st.cache_resource
def load_model():
    return keras_nlp.models.CausalLM.from_preset(f'kaggle://abhionic/gpt2/keras/gpt2_medqna')

template = 'Question:\n{question}\n\nAnswer:\n{answer}'
med_lm = load_model()
med_lm.preprocessor.sequence_length = 512
sampler = keras_nlp.samplers.TopKSampler(k=5, seed=2)
med_lm.compile(sampler=sampler)

# react to user input
if prompt := st.chat_input('enter health question (experimental)'):
    # add user message to chat history
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    # display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    context = template.format(question=prompt, answer='',)
    answer = med_lm.generate(context, max_length=256)
    def stream_data():
        for word in answer.split(' '):
            yield word + ' '
            time.sleep(0.02)

    # display assistant response in chat message container
    with st.chat_message('assistant'):
        response = st.write_stream(stream_data)
    # add assistant response to chat history
    st.session_state.messages.append({'role': 'assistant', 'content': response})
