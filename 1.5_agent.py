import streamlit as st
import tempfile
import os

# 👇 这是最终 100% 正确导入
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.embeddings import BaichuanTextEmbeddings


st.set_page_config(page_title="Rag Agent", layout="wide")
st.title("Rag Agent")

uploaded_files = st.sidebar.file_uploader(
    label="上传txt⽂件", type=["txt"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("请先上传按TXT⽂档。")
    st.stop()

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    # ⭐ 只改这一行！去掉 dir=r"D:\\"，云环境才能跑
    temp_dir = tempfile.TemporaryDirectory()

    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())

        loader = TextLoader(temp_filepath, encoding="utf-8")
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # ⭐ 完全保留你的百川 Embedding，不动
    embeddings = BaichuanTextEmbeddings(api_key=st.secrets["BAICHUAN_API_KEY"])
    vectordb = Chroma.from_documents(splits, embeddings)

    retriever = vectordb.as_retriever()
    return retriever

retriever = configure_retriever(uploaded_files)

if "messages" not in st.session_state or st.sidebar.button("清空聊天记录"):
    st.session_state["messages"] = [{"role": "assistant", "content": "您好，我是AI智能助手，我可以查询文档"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

from langchain.tools.retriever import create_retriever_tool
tool = create_retriever_tool(
    retriever = retriever,
    name = "文档检索",
    description = "用于检索用户提出的问题，并基于检索到的文档内容进行回复.",
)
tools = [tool]

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)

# # 指令模板
# instructions = """你是一个设计用于查询文档来回答问题的代理对象。
# 你可以使用文档检索工具，并基于检索内容来回答问题
# 你可能不查询文档就知道答案，但是你仍然应该查询文档来获得答案。
# 如果你从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
# """

# 指令模板【修改1：强化格式约束+强制输出格式】
instructions = """你是一个设计用于查询文档来回答问题的代理对象。
你必须严格使用文档检索工具，基于检索到的内容回答问题，绝对不能直接回答。
回答必须严格遵循以下格式：
1. 开头说明：根据检索到的文档信息，关于「用户问题」的解答如下：
2. 分点清晰列出症状、药方等内容
3. 最后必须加上：温馨提示：以上信息仅供参考。中医诊疗强调辨证论治，个人体质与病情不同，用药前请务必咨询专业中医师。
如果你从文档中找不到任何信息用于回答问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
"""

# 基础提示模板-React提示词【修改2：强约束格式，强制模型走Action步骤】
base_prompt_template = """
{instructions}

TOOLS:
------

You have access to the following tools:

{tools}

你必须严格遵守以下格式，**绝对不能跳过任何步骤**，否则会报错：
To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: 文档检索
Action: the action to take, should be one of [{tool_names}]
Action Input: {input}
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
"""


# 创建基础提示模板
base_prompt = PromptTemplate.from_template(base_prompt_template) # from_template = 识别 {变量} partial = 提前填好一部分变量
# 创建部分填充的提示模板
prompt = base_prompt.partial(instructions=instructions)

# 创建llm
API_KEY = st.secrets["DEEPSEEK_API_KEY"]
llm = ChatOpenAI(model="deepseek-reasoner",
                   openai_api_key=API_KEY,
                   openai_api_base="https://api.deepseek.com",
                 temperature=0)

# 创建react Agent
agent = create_react_agent(llm, tools, prompt)

# 创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, memory=memory, verbose=True, handle_parsing_errors="请严格按照要求的格式输出，必须包含Thought和Action步骤",max_iterations=5)

# 创建聊天输入框
user_query = st.chat_input(placeholder="请开始提问吧!")

# 如果有用户输入的查询
if user_query:
    # 添加用户消息到session_state
    st.session_state.messages.append({"role": "user", "content": user_query})
    # 显示用户消息
    st.chat_message("user").write(user_query)
    
    #创建一个 Streamlit 的聊天消息块，用于显示助手（机器人）的回复。
    with st.chat_message("assistant"):
        # st.container(): 创建一个 Streamlit 的容器组件，用于动态更新内容。
        #StreamlitCallbackHandler 是 LangChain 的一个回调处理器，用于将模型的输出或日志信息显示在 Streamlit 界面中。通过将 st.container() 传递给 StreamlitCallbackHandler，可以将 LangChain 的输出直接渲染到 Streamlit 的容器中。
        st_cb = StreamlitCallbackHandler(st.container())
        
        # 配置 LangChain 的回调函数列表，将 StreamlitCallbackHandler 添加到回调中。通过配置回调函数，LangChain 可以在处理过程中调用StreamlitCallbackHandler，从而将输出或日志信息显示在 Streamlit 界面中。这种方式可以实现实时更新界面。
        config = {"callbacks": [st_cb]}
        
        # 执行Agent并获取响应
        response = agent_executor.invoke({"input": user_query}, config=config)
        # 添加助手消息到session_state
        st.session_state.messages.append({"role": "assistant", "content": response["output"]})
        # 显示助手响应
        st.write(response["output"])
