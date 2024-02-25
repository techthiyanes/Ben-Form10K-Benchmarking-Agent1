# Import required modules from the langchain library and other packages
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import  RetrievalQA
# from langchain.chat_models import ChatOpenAI
import os
# from langchain.agents import initialize_agent, Tool
# from langchain.agents import AgentType
import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_core.runnables import RunnableConfig
from PIL import Image
from langchain.schema import SystemMessage
from langchain_core.prompts import PromptTemplate

# Streamlit page configuration
st.set_page_config(page_title="Ben - AR Insights", page_icon="🦜")
st.title("Ben - Annual Report Insights\n\n**powered by GPT-4 Turbo**")


# Custom CSS for Streamlit using Google Fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Epilogue:ital,wght@0,500;0,600;0,700;0,800;0,900;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
    h1, h2, h3, h4, h5, h6, [class^="st-"] {
			font-family: 'Epilogue', sans-serif !important;
			}
    </style>
    """,
    unsafe_allow_html=True,
)

# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Inter+Tight:wght@500&family=Literata:opsz,wght@7..72,500&display=swap');
#     h1, h2, h3, h4, h5, h6, [class^="st-"] {
# 			font-family: 'Exo 2', sans-serif !important;
# 			}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # Custom CSS for Streamlit using Google Fonts
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Audiowide&display=swap');
#     html, body, textarea , [class*="css"] {
# 			font-family: 'Audiowide', 'Gloria Hallelujah', 'Poppins', 'Oswald', 'DM Sans', 'Urbanist', 'Gothic A1', 'Audiowide', 'Dela Gothic One', 'Special Elite', sans-serif !important;
# 			}
#     </style>
#     """,
#     unsafe_allow_html=True,
# )


# Introductory messages and sample questions
intoductory_message='''Hello, I'm **Ben** — your go-to expert for **Annual Report Insights**!\n\nPowered by GPT-4, I can assist you with **in-depth analysis** and **benchmarking** of annual reports of some of the world's leading tech giants:\n- **Meta**\n- **Amazon**\n- **Alphabet**\n- **Apple**\n- **Microsoft**,\n\nfor the years **2022, 2021, and 2020**.\n\nWhether you are comparing financial performances, exploring trends, or seeking detailed insights across different companies and years, I transform complex data into actionable knowledge. Unlock the power of informed decision-making today with me!'''


with st.sidebar:
    st.error('**Ben - Annual Report Insights**')
    st.image(Image.open('BenLogo.png'))
    st.warning(intoductory_message)
    



# Cache the benchmarking agent for performance optimization
@st.cache_resource
def preparing_benchmarking_agent():
    # Initialize embeddings and chat models
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview", streaming=True)

    # Load local FAISS document stores for multiple companies and years
    apple_2022_docs_store = FAISS.load_local(r'data/datastores/apple_2022', embeddings)
    apple_2021_docs_store = FAISS.load_local(r'data/datastores/apple_2021', embeddings)
    apple_2020_docs_store = FAISS.load_local(r'data/datastores/apple_2020', embeddings)
   
    microsoft_2022_docs_store = FAISS.load_local(r'data/datastores/msft_2022', embeddings)
    microsoft_2021_docs_store = FAISS.load_local(r'data/datastores/msft_2021', embeddings)
    microsoft_2020_docs_store = FAISS.load_local(r'data/datastores/msft_2020', embeddings)


    amazon_2022_docs_store = FAISS.load_local(r'data/datastores/amzn_2022', embeddings)
    amazon_2021_docs_store = FAISS.load_local(r'data/datastores/amzn_2021', embeddings)
    amazon_2020_docs_store = FAISS.load_local(r'data/datastores/amzn_2020', embeddings)


    alphabet_2022_docs_store = FAISS.load_local(r'data/datastores/alphbt_2022', embeddings)
    alphabet_2021_docs_store = FAISS.load_local(r'data/datastores/alphbt_2021', embeddings)
    alphabet_2020_docs_store = FAISS.load_local(r'data/datastores/alphbt_2020', embeddings)


    meta_2022_docs_store = FAISS.load_local(r'data/datastores/meta_2022', embeddings)
    meta_2021_docs_store = FAISS.load_local(r'data/datastores/meta_2021', embeddings)
    meta_2020_docs_store = FAISS.load_local(r'data/datastores/meta_2020', embeddings)
    
    template = """Use the following pieces of context only to answer the question at the end. Your answer should be very detailed, deep, insighful, and high quality, factually correct. Use bullet points, markdown table wherever necessary.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    # Create QA retrieval chains for various companies and years
    apple_2022_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=apple_2022_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    apple_2021_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=apple_2021_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    apple_2020_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=apple_2020_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    microsoft_2022_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=microsoft_2022_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    microsoft_2021_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=microsoft_2021_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    microsoft_2020_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=microsoft_2020_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    amazon_2022_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=amazon_2022_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    amazon_2021_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=amazon_2021_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    amazon_2020_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=amazon_2020_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    meta_2022_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=meta_2022_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    meta_2021_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=meta_2021_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    meta_2020_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=meta_2020_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


    alphabet_2022_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=alphabet_2022_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    alphabet_2021_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=alphabet_2021_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    alphabet_2020_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=alphabet_2020_docs_store.as_retriever(search_kwargs={'k':10}), chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    # Define the tools (queries against different document stores)
    tools = [
        Tool(
            name="Apple Form 10K 2022",
            func=apple_2022_qa.run,
            description="useful when you need to answer from Apple 2022. Input should be a search friendly query",
        ),
        Tool(
            name="Apple Form 10K 2021",
            func=apple_2021_qa.run,
            description="useful when you need to answer from Apple 2021. Input should be a search friendly query",
        ),
        Tool(
            name="Apple Form 10K 2020",
            func=apple_2020_qa.run,
            description="useful when you need to answer from Apple 2020. Input should be a search friendly query",
        ),
        Tool(
            name="Microsoft Form 10K 2022",
            func=microsoft_2022_qa.run,
            description="useful when you need to answer from Microsoft 2022. Input should be a search friendly query",
        ),
        Tool(
            name="Microsoft Form 10K 2021",
            func=microsoft_2021_qa.run,
            description="useful when you need to answer from Microsoft 2021. Input should be a search friendly query",
        ),
        Tool(
            name="Microsoft Form 10K 2020",
            func=microsoft_2020_qa.run,
            description="useful when you need to answer from Microsoft 2020. Input should be a search friendly query",
        ),
        Tool(
            name="Meta Form 10K 2022",
            func=meta_2022_qa.run,
            description="useful when you need to answer from Meta 2022. Input should be a search friendly query",
        ),
        Tool(
            name="Meta Form 10K 2021",
            func=meta_2021_qa.run,
            description="useful when you need to answer from Meta 2021. Input should be a search friendly query",
        ),
        Tool(
            name="Meta Form 10K 2020",
            func=meta_2020_qa.run,
            description="useful when you need to answer from Meta 2020. Input should be a search friendly query",
        ),
        Tool(
            name="Alphabet Form 10K 2022",
            func=alphabet_2022_qa.run,
            description="useful when you need to answer from Alphabet or Google 2022. Input should be a search friendly query",
        ),
        Tool(
            name="Alphabet Form 10K 2021",
            func=alphabet_2021_qa.run,
            description="useful when you need to answer from Alphabet or Google 2021. Input should be a search friendly query",
        ),
        Tool(
            name="Alphabet Form 10K 2020",
            func=alphabet_2020_qa.run,
            description="useful when you need to answer from Alphabet or Google 2020. Input should be a search friendly query",
        ),
        Tool(
            name="Amazon Form 10K 2022",
            func=amazon_2022_qa.run,
            description="useful when you need to answer from Amazon 2022. Input should be a search friendly query",
        ),
        Tool(
            name="Amazon Form 10K 2021",
            func=amazon_2021_qa.run,
            description="useful when you need to answer from Amazon 2021. Input should be a search friendly query",
        ),
        Tool(
            name="Amazon Form 10K 2020",
            func=amazon_2020_qa.run,
            description="useful when you need to answer from Amazon 2020. Input should be a search friendly query",
        ),
    ]


    prompt = hub.pull("hwchase17/react")
    agent_prompt_template = '''Write a detailed, deep, insighful, high quality and factually correct answer for the question as best as you can. Always present your final answer in bullets and markdown table whenever applicable.
    You have access to the following tools:
	
	{tools}
	
	Use the following format:
	
	Question: the input question you must answer
	Thought: you should always think about what to do
	Action: the action to take, should be one of [{tool_names}]
	Action Input: the input to the action
	Observation: the result of the action
	... (this Thought/Action/Action Input/Observation can repeat N times)
	Thought: I now know the final answer
	Final Answer: the final answer to the original input question. Your final answer should be detailed, deep, insightful, high quality and should always be in bullets and markdown table whenever applicable.
	
	Begin!
	
	Question: {input}
	Thought:{agent_scratchpad}'''
    
    prompt = PromptTemplate.from_template(agent_prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor

# Initialize the agent
agent = preparing_benchmarking_agent()

# Session state to manage user messages
if "messages" not in st.session_state:
    st.session_state["messages"] = []


sample_question_1 = '''What is the net sales of Apple in 2022?'''
sample_question_2 = '''What are some of the risk factors Microsoft reported in 2021?'''
sample_question_3 = '''Give me the Apple's breakdown of net sales by products and services.'''
sample_question_4 = '''Compare the revenue of Apple, Meta, Microsoft for the year 2022. Present your response in table with columns Revenue, Apple 2022, Meta 2022, Microsoft 2022.'''
sample_question_5 = '''Compare the risk factors of Microsoft and Alphabet for 2021. Present your response in a table with columns "Risk Category", "Risk Details - Microsoft", "Risk Details - Alphabet"'''
sample_question_6 = '''Compare the covid-19 impact on Apple in 2020, 2021 and 2022? Present your response in table with columns "Impact category", "Key Details-2020", "Key Details-2021, "Key Details-2022", "Changes/Conclusion"'''

with st.chat_message("assistant"):
    st.write("Hello, I'm **Ben** — your go-to expert for **Annual Report Insights**!")
    # st.image(Image.open('BenLogo.png'))
    # st.write(intoductory_message)
    st.write(f'''Ready to dive in? Here are your starting queries:\n\n🎯 **Basic queries:**''')
    st.info(sample_question_1)
    st.info(sample_question_2)
    st.info(sample_question_3)
    st.write(f'''🎯 **Complex & Benchmarking Queries:**''')
    st.info(sample_question_5)
    st.info(sample_question_6)

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
if input_prompt := st.chat_input(placeholder="What is the net sales of Apple in 2022?"):
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    st.chat_message("user").write(input_prompt)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_callback]
        answer = agent.invoke({"input": input_prompt}, cfg)
        st.session_state.messages.append({"role": "assistant", "content": answer['output']})
        st.markdown(answer['output'])
