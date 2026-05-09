from langchain_core.prompts import ChatPromptTemplate
from pymilvus import DataType
from pymilvus import MilvusClient
## 获取到客户端对象
client = MilvusClient(
    uri="mivus_demo.db"
)
def query()->list[dict]:
    """
    根据用户问题，返回向量数据库当中所对应的相关文档信息
    :param user_query:
    :return:
    """
    res = client.query(collection_name="demo_collection",
                 filter='metadata["source"] == "/home/m1881/assets/sample.docx"', # 根据collection当中定义的其他字段信息（非向量字段）进行筛选
                 output_fields=["content"],
                 limit=1,
                 )
    return res

def query_vector(user_query):
    """
    通过语义进行向量搜索
    :param user_query:
    :return:
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    embed_model = HuggingFaceEmbeddings(model_name = "/home/m1881/models/bge-base-zh-v1.5")
    query_embedding = embed_model.embed_query(user_query)  # 查询嵌入
    context = client.search(
        collection_name="demo_collection",  # collection 名称
        data=[query_embedding],  # 搜索的向量
        anns_field="vector",  # 进行向量搜索的字段
        # 度量方式：L2 欧氏距离/IP 内积/COSINE 余弦相似度
        search_params={"metric_type": "COSINE"},
        output_fields=["content",],  # 输出字段
        limit=3,  # 搜索结果数量
    )
    real_results = context[0]
    content_list = [real_result["entity"]["content"] for real_result in real_results]
    return content_list

def get_llm_chain():
    """
    获取到一个llm执行链，这个执行链，能够基于文档内容去做应答
    :return:
    """
    from langchain_openai import ChatOpenAI
    import dotenv
    dotenv.load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini")
    chat_template = ChatPromptTemplate.from_messages(
        [
    {"role":"system","content":"请你基于以下上下文信息，回答用户相关的问题。不可以回答上下文当中不存在的信息，如果不存在，直接说不知道。上下文信息如下：\n\n {context}"},
            {"role":"user","content":"{user_question}"}
         ]
    )
    chain = chat_template | llm
    return chain


user_input = "周末北京去哪儿玩"
context_result = query_vector(user_input)

chain = get_llm_chain()
llm_result = chain.invoke({"user_question":"周末北京去哪儿玩","context":context_result})
print(llm_result)