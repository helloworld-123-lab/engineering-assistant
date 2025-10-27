import streamlit as st
import chromadb
from openai import OpenAI
import requests
import json
from typing import Dict, List, Generator
import re
import google.protobuf.message_factory as _mf
import base64
from io import BytesIO
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import langid
from langcodes import Language
import os
import urllib.request
import py7zr

# https://protobuf.dev/news/v30/#remove-deprecated
if not hasattr(_mf.MessageFactory, "GetPrototype"):
    _mf.MessageFactory.GetPrototype = staticmethod(_mf.GetMessageClass)

# 数据库下载和解压函数
def download_and_extract_database():
    db_path = "chroma_db"
    zip_url = "https://raw.githubusercontent.com/helloworld-123-lab/engineering-assistant/main/chroma_db.7z"
    
    # 如果数据库已经存在，直接返回，不显示任何消息
    if os.path.exists(db_path):
        return True
    
    # 只在需要下载时显示消息
    with st.spinner("\n正在下载数据库文件，这可能需要几分钟..."):
        try:
            # 下载文件
            urllib.request.urlretrieve(zip_url, "chroma_db.7z")
            
            # 解压文件
            with py7zr.SevenZipFile("chroma_db.7z", mode='r') as z:
                z.extractall(path=".")
            
            # 清理临时文件
            os.remove("chroma_db.7z")
            
            # 这里不显示成功消息，因为spinner会自动消失
            return True
            
        except Exception as e:
            st.error(f"数据库文件下载失败: {str(e)}")
            return False

# API密钥获取函数
def get_api_keys():
    """安全地获取API密钥"""
    # 优先从环境变量获取
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "")
    siliconflow_key = os.environ.get("SILICONFLOW_API_KEY", "")
    p_key = os.environ.get("P_API_KEY", "")
    
    return deepseek_key, siliconflow_key,p_key

# 通过硅基流动API获取嵌入向量
def get_embeddings(texts: List[str], api_key: str) -> List[List[float]]:
    """
    通过硅基流动API获取文本的嵌入向量
    """
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.siliconflow.cn/v1"
    )
    
    try:
        # 如果是单个文本，转换为列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 调用嵌入模型
        response = client.embeddings.create(
            model="Pro/BAAI/bge-m3",
            input=texts
        )
        
        # 提取嵌入向量
        embeddings = [item.embedding for item in response.data]
        return embeddings
        
    except Exception as e:
        st.error(f"获取嵌入向量失败: {str(e)}")
        # 返回随机向量作为降级方案
        return [np.random.randn(1024).tolist() for _ in texts]

# 初始化ChromaDB客户端 - 适配Render环境
@st.cache_resource
def get_chroma_collection():
    # 确保数据库文件存在
    if not download_and_extract_database():
        st.error("数据库初始化失败，请检查数据库文件")
        return None
    
    try:
        client = chromadb.PersistentClient(path="chroma_db")
        collection = client.get_collection("my_collection")
        return collection
    except Exception as e:
        st.error(f"数据库连接失败: {str(e)}")
        return None

# 获取层级选项 - 修复哈希问题
@st.cache_data
def get_level_options(_collection):
    # 获取所有文档的元数据来构建层级选项
    all_data = _collection.get(include=['metadatas'])
    metadatas = all_data['metadatas']

    course_tree = {}

    for metadata in metadatas:
        course = metadata.get('level_1')
        if not course:
            continue
        if course not in course_tree:
            course_tree[course] = {}

        node = course_tree[course]
        # 动态处理 level_2~level_5
        for level in ['level_2', 'level_3', 'level_4', 'level_5']:
            val = metadata.get(level)
            if val:
                if val not in node:
                    node[val] = {}
                node = node[val]
    return course_tree

def detect_user_language(text):
    """
    语言检测函数
    """
    try:
        # 检查是否为空或过短的文本
        if not text or len(text.strip()) < 2:
            return "中文"  # 默认语言
        
        # 使用langid进行语言检测
        lang_code, confidence = langid.classify(text)
        
        # 语言代码到中文名称的映射
        language_mapping = {
            'zh': '中文',
            'en': '英文',
            'de': '德文',
            'fr': '法文',
            'es': '西班牙文',
            'it': '意大利文',
            'ja': '日文',
            'ko': '韩文',
            'ru': '俄文',
            'pt': '葡萄牙文',
            'nl': '荷兰文',
            'pl': '波兰文',
            'sv': '瑞典文',
            'da': '丹麦文',
            'no': '挪威文',
            'fi': '芬兰文',
            'hu': '匈牙利文',
            'cs': '捷克文',
            'el': '希腊文',
            'tr': '土耳其文',
            'th': '泰文',
            'vi': '越南文',
            'id': '印尼文',
            'ms': '马来文',
            'hi': '印地文',
            'ar': '阿拉伯文',
            'he': '希伯来文',
            'fa': '波斯文'
        }
        
        # 返回对应的语言名称，如果不在映射中则返回中文
        detected_language = language_mapping.get(lang_code, '中文')
        
        # 如果置信度低于0.5，默认使用中文（特别是对于短文本）
        # if confidence < 0.5:
        #     return "中文"
        
        # 特殊处理：中文文本可能被误识别为其他语言
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text)
        
        # 如果文本中超过50%是中文字符，强制识别为中文
        if total_chars > 0 and chinese_chars / total_chars > 0.5:
            return "中文"
            
        return detected_language
        
    except Exception as e:
        print(f"语言检测失败: {e}")
        # 备用检测方法：检查是否包含中文字符
        if re.search(r'[\u4e00-\u9fff]', text):
            return "中文"
        return "中文"  # 默认使用中文

# 流式调用DeepSeek API
def call_deepseek_api_stream(question: str, context: str, api_key: str) -> Generator[str, None, None]:
    # 准确检测用户语言
    user_language = detect_user_language(question)
    
    client = OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com/v1"
    )
    
    prompt = f"""你是一位国际化的工程机械领域专家。请基于以下背景资料回答问题：
    
    背景资料：
    {context}
    
    问题：{question}
    
    请根据背景资料提供准确、专业、条理清晰的回答。如果资料中没有相关信息，请说明无法回答。
    
    重要要求：
    1. 用户使用{user_language}提问，回复的全部内容都必须使用相同的{user_language}进行回答，不要夹杂其他语言文字。
    2. 回答要专业、准确、易于理解。
    3. 如果资料不足，请明确指出。
    4. 不管检索到的资料是什么语言，都必须把资料翻译成{user_language}，必须使用{user_language}回答，不得使用其他语言。
    """

    messages=[
            {"role": "user", "content": prompt}
            ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=2000,
            stream=True
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                yield token

    except Exception as e:
        yield f"助手出错：{str(e)}"

# 非流式调用DeepSeek API - 用于生成提示词
def call_deepseek_api_non_stream(prompt: str, api_key: str) -> str:
    client = OpenAI(
        api_key=api_key, 
        base_url="https://api.deepseek.com/v1"
    )
    
    messages=[{"role": "user", "content": prompt}]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成提示词时出错：{str(e)}"

# 生成图片提示词
def generate_image_prompt(content: str, image_type: str, api_key: str) -> str:
    """生成适合图片生成的提示词"""
    prompt = f"""
    请将以下工程机械相关内容转换为适合AI图片生成的提示词，无论原文内容是什么语言，要求一定要将输出文生图提示词转换为英文：
    
    原文内容：
    {content}
    
    图片类型：{image_type}
    
    要求：
    1. 英文提示词长度控制在300字左右。
    2. 包含具体的视觉元素：颜色、构图、细节、风格。
    3. 专注于工程机械的技术特点和工作场景。
    4. 如果是示意图，要突出结构和原理。
    5. 如果是思维导图，要体现逻辑关系和层次结构。
    
    请直接输出优化后的英文图片生成提示词，不要添加其他解释。
    """
    
    return call_deepseek_api_non_stream(prompt, api_key)

# 生成Mermaid图表代码
def generate_mermaid_code(content: str, api_key: str) -> str:
    """生成Mermaid语法表示的思维导图或流程图"""
    prompt = f"""
    请将以下工程机械相关内容转换为Mermaid语法代码，如果内容是非中文的将其转换成中文：
    
    内容：
    {content}
    
    要求：
    1. 使用Mermaid的flowchart TD（自上而下流程图）或mindmap（思维导图）语法
    2. 根据内容特点选择最适合的图表类型
    3. 保持结构清晰，层次分明
    4. 使用简洁的标签描述
    5. 包含适当的样式和颜色
    6. 输出纯Mermaid代码，不要包含其他解释
    
    Mermaid代码示例：
    ```mermaid
    mindmap
      root(工程机械)
        挖掘机
          类型
            履带式
            轮式
          工作原理
            液压系统
            动力传输
    ```
    
    请根据提供的内容生成合适的Mermaid代码：
    """
    
    return call_deepseek_api_non_stream(prompt, api_key)

# 通过Kroki API生成Mermaid图像
def generate_mermaid_image(mermaid_code: str) -> Image.Image:
    """通过 Kroki API 将 Mermaid 代码直接转换为 PNG 图片"""
    try:
        clean_code = mermaid_code.strip().replace('```mermaid', '').replace('```', '').strip()
        url = "https://kroki.io/mermaid/png"  # 改为 png

        response = requests.post(url, data=clean_code.encode('utf-8'))  # 直接发送文本，不用 json
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        else:
            print(f"Kroki API 错误: {response.status_code} - {response.text}")
            return None

    except Exception as e:
        print(f"生成 Mermaid 图像时出错: {e}")
        return None

# 生成图片 - 支持多种模型
def generate_image(prompt: str, model_type: str, api_key: str = None, api_key1: str = None) -> Image.Image:
    """
    根据选择的模型生成图片
    model_type: 模型类型，包括 'kolors', 'flux', 'kontext', 'turbo', 'gptimage'
    """
    try:
        if model_type == 'kolors':
            # 使用硅基流动的Kolors模型
            client = OpenAI(
                api_key=api_key, 
                base_url="https://api.siliconflow.cn/v1"
            )
            response = client.images.generate(
                model="Kwai-Kolors/Kolors",
                prompt=prompt,
                size="1024x1024",
                n=1,
                extra_body={"step": 20}
            )
            image_url = response.data[0].url
            # 下载图片
            image_response = requests.get(image_url)
            img = Image.open(BytesIO(image_response.content))
            return img
        
        elif model_type == 'flux' or model_type == 'turbo':
            # 使用pollinations.ai的模型
            # 对prompt进行URL编码
            import urllib.parse
            encoded_prompt = urllib.parse.quote(prompt)
            
            # 构建pollinations.ai的URL
            pollinations_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model={model_type}"
            
            # 下载图片
            image_response = requests.get(pollinations_url)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                return img
            else:
                st.error(f"Pollinations API 请求失败: {image_response.status_code}")
                return None

        elif model_type == 'kontext' or model_type == 'gptimage':
            # 使用pollinations.ai的模型
            # 对prompt进行URL编码
            import urllib.parse
            encoded_prompt = urllib.parse.quote(prompt)
            # 构建pollinations.ai的URL
            url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&model={model_type}"
            headers = {"Authorization": f"Bearer {api_key1}"}
            # 下载图片
            image_response = requests.get(url, headers=headers)
            if image_response.status_code == 200:
                img = Image.open(BytesIO(image_response.content))
                st.success(f"✅ {model_type.upper()}模型图片生成成功！")
                return img
            else:
                st.error(f"❌ {model_type}模型请求失败: {image_response.status_code}")
                st.info(f"💡 提示：{model_type}模型连接不稳定，建议稍后重试或选择Kolors/Flux/Turbo模型")
                print(response.text)
                return None
                
    except Exception as e:
        st.error(f"图片生成失败: {str(e)}")
        return None

# 构建查询过滤器
def build_where_filter(selected_levels: dict) -> dict:
    conditions = []
    for level, value in selected_levels.items():
        if value and value != "全部":
            conditions.append({level: {"$eq": value}})
    
    if not conditions:
        return None
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {"$and": conditions}

# 查询向量数据库
def query_vector_db(question: str, _collection, siliconflow_api_key: str, selected_levels: Dict, n_results: int = 3):
    # 通过API获取查询文本的嵌入向量
    query_embedding = get_embeddings(question, siliconflow_api_key)[0]
    
    where_filter = build_where_filter(selected_levels)
    
    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_filter,
        include=['documents', 'metadatas', 'distances']
    )
    
    return results

# 清理文本显示
def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff，。！？；：""''（）【】]', '', text)
    return text.strip()

# 主应用
def main():
    st.set_page_config(
        page_title="工程机械知识助手",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 工程机械知识助手")
    st.markdown("基于向量数据库的智能问答系统")
    
    # 初始化session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'is_generating' not in st.session_state:
        st.session_state.is_generating = False
    if 'current_response' not in st.session_state:
        st.session_state.current_response = ""
    if 'streaming_placeholder' not in st.session_state:
        st.session_state.streaming_placeholder = None
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = {}
    if 'image_history' not in st.session_state:
        st.session_state.image_history = []
    if 'generating_image_type' not in st.session_state:
        st.session_state.generating_image_type = None
    if 'latest_illustration' not in st.session_state:
        st.session_state.latest_illustration = None
    if 'latest_mindmap' not in st.session_state:
        st.session_state.latest_mindmap = None
    if 'target_assistant_idx' not in st.session_state:    
        st.session_state.target_assistant_idx = None
        
    
    # 默认密钥
    DEFAULT_DEEPSEEK_API_KEY, DEFAULT_SILICONFLOW_API_KEY, P_API_KEY= get_api_keys()
    
    # 初始化资源
    try:
        collection = get_chroma_collection()
        if collection is None:
            st.error("数据库初始化失败，请检查数据库文件配置")
            return
        level_options = get_level_options(collection)
    except Exception as e:
        st.error(f"初始化失败: {e}")
        return
    
    # 侧边栏 - 筛选条件
    with st.sidebar:
        st.header("📁 文档筛选")
        level_names = {
            'level_1': '一级分类',
            'level_2': '二级分类', 
            'level_3': '三级分类',
            'level_4': '四级分类',
            'level_5': '五级分类'
        }
        course_tree = level_options
    
        # 选择课程
        courses = ["全部"] + list(course_tree.keys())
        selected_course = st.selectbox("选择课程", courses, key="level_1")
    
        selected_levels = {"level_1": selected_course}
    
        current_tree = course_tree.get(selected_course, {}) if selected_course != "全部" else {}
        for i, level in enumerate(['level_2','level_3','level_4','level_5']):
            if current_tree:
                options = ["全部"] + list(current_tree.keys())
                selected = st.selectbox(f"选择{level}", options, key=level)
                selected_levels[level] = selected
                if selected != "全部":
                    current_tree = current_tree.get(selected, {})
                else:
                    current_tree = {}
            else:
                selected_levels[level] = "全部"
        
        st.markdown("---")
        st.header("🔧 设置")
        
        deepseek_api_key = st.text_input("DeepSeek API Key（留空使用默认值）", type="password")
        if not deepseek_api_key:
            deepseek_api_key = DEFAULT_DEEPSEEK_API_KEY
            
        siliconflow_api_key = st.text_input("SiliconFlow API Key（留空使用默认值）", type="password")
        if not siliconflow_api_key:
            siliconflow_api_key = DEFAULT_SILICONFLOW_API_KEY
        
        n_results = st.slider("检索结果数量", 1, 10, 3)
        
        # 图片生成模型选择
        st.markdown("---")
        st.header("🎨 图片生成设置")
        
        image_model = st.selectbox(
            "选择图片生成模型",
            ["Kolors (硅基流动)", "Flux (Pollinations)", "Kontext (Pollinations)", "Turbo (Pollinations)", "GPTImage (Pollinations)"],
            index=0,
            help="选择用于生成图片的AI模型\n\n注意：Kontext和GPTImage模型生成时间较长且连接不稳定"
        )

        # 如果选择了Kontext或GPTImage，显示警告
        if image_model in ["Kontext (Pollinations)", "GPTImage (Pollinations)"]:
            st.warning("⚠️ 注意：Kontext和GPTImage模型生成图片时间较长，连接不稳定，如果请求不成功可稍后重试或选择其他模型")
        
        # 将显示名称映射到模型标识符
        model_mapping = {
            "Kolors (硅基流动)": "kolors",
            "Flux (Pollinations)": "flux",
            "Kontext (Pollinations)": "kontext", 
            "Turbo (Pollinations)": "turbo",
            "GPTImage (Pollinations)": "gptimage"
        }
        
        selected_image_model = model_mapping[image_model]
        
        st.markdown("---")
        st.header("🗂️ 对话管理")
        if st.button("🗑️ 清空对话", use_container_width=True, disabled=st.session_state.is_generating):
            st.session_state.messages = []
            st.session_state.current_response = ""
            st.session_state.is_generating = False
            st.session_state.streaming_placeholder = None
            st.session_state.generated_images = {}
            st.session_state.image_history = []
            st.session_state.generating_image_type = None
            st.session_state.latest_illustration = None
            st.session_state.latest_mindmap = None
            st.session_state.target_assistant_idx = None
            st.rerun()
        
        # 显示数据库信息
        st.markdown("---")
        st.header("📊 数据库信息")
        total_docs = collection.count()
        st.write(f"总文档数: {total_docs}")
        
        # 显示当前筛选条件
        st.subheader("🔍 当前筛选")
        active_filters = {name: selected_levels[key] for key, name in level_names.items() 
                         if selected_levels[key] and selected_levels[key] != "全部"}
        
        if active_filters:
            for name, value in active_filters.items():
                st.write(f"**{name}:** {value}")
        else:
            st.write("全部文档")
    
    # 主区域 - 聊天界面设计
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 对话")
        
        # 显示聊天历史
        for idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # 显示生成的图片（如果有）
                if message["role"] == "assistant" and str(idx) in st.session_state.generated_images:
                    images_dict = st.session_state.generated_images[str(idx)]
                    # 显示示意图
                    if "illustration" in images_dict:
                        image_data, prompt, model_info = images_dict["illustration"]
                        st.image(image_data, caption=f"生成的示意图 - {model_info}", use_container_width=True)
                    # 显示思维导图
                    if "mindmap" in images_dict:
                        image_data, prompt = images_dict["mindmap"]
                        st.image(image_data, caption="生成的思维导图/流程图", use_container_width=True)
                
                # 显示相关文档（如果有）
                if message["role"] == "assistant" and "documents" in message and message["documents"]:
                    with st.expander(f"📄 查看相关文档 ({len(message['documents'])} 条)", expanded=False):
                        for i, doc in enumerate(message["documents"], 1):
                            data = doc['data']
                            st.markdown(f"**文档 {i}** (相似度: {1-doc.get('distance', 0):.3f})")
                            st.markdown(f"**内容**: {clean_text(data.get('document', '未知'))}")
                            if doc.get('metadata'):
                                metadata = doc['metadata']
                                for key, value in metadata.items():
                                    if value and value != "全部":
                                        st.markdown(f"**{key}**: {value}")
                            st.markdown("---")
        
            # 如果当前消息是最后一条用户消息且正在生成，则在它下面显示占位符
            if idx == len(st.session_state.messages) - 1 and st.session_state.is_generating:
                if st.session_state.streaming_placeholder is None:
                    st.session_state.streaming_placeholder = st.empty()
                st.session_state.streaming_placeholder.markdown(st.session_state.current_response + "▌")
    
    with col2:
        st.subheader("📊 检索信息")
        
        # 显示最近一次检索的信息
        if st.session_state.messages and not st.session_state.is_generating:
            last_message = st.session_state.messages[-1]
            if last_message["role"] == "assistant" and "documents" in last_message:
                documents = last_message["documents"]
                
                st.metric("检索文档数", len(documents))
                
                # 显示相似度分布
                if documents:
                    st.subheader("📈 相似度分析")
                    similarities = [1 - doc.get('distance', 0) for doc in documents]
                    avg_similarity = sum(similarities) / len(similarities)
                    st.metric("平均相似度", f"{avg_similarity:.3f}")
                    
                    # 显示每个文档的相似度
                    for i, similarity in enumerate(similarities):
                        # 确保相似度在 0.0 到 1.0 之间
                        if similarity < 0.0:
                            progress_value = 0.0
                        elif similarity > 1.0:
                            progress_value = 1.0
                        else:
                            progress_value = similarity
                        
                        st.progress(progress_value, text=f"文档 {i+1}: {similarity:.3f}")

        # 生成状态指示器
        if st.session_state.is_generating:
            st.info("🔄 正在生成回答，请稍候...")
        
        # 在右侧底部添加图片展示区域
        st.markdown("---")
        st.subheader("🎨 生成的图片")
        
        # 显示图片生成状态
        if st.session_state.generating_image_type:
            image_type_name = "示意图" if st.session_state.generating_image_type == "illustration" else "思维导图/流程图"
            st.info(f"🎨 正在生成{image_type_name}，请稍候...")
        
        # 显示最新图片
        col_illus, col_mind = st.columns(2)
        
        with col_illus:
            st.subheader("最新示意图")
            if st.session_state.latest_illustration:
                image_data, prompt, model_info = st.session_state.latest_illustration
                st.image(image_data, caption=f"模型: {model_info}", use_container_width=True)
            else:
                st.info("暂无示意图")
        
        with col_mind:
            st.subheader("最新思维导图")
            if st.session_state.latest_mindmap:
                image_data, prompt = st.session_state.latest_mindmap
                st.image(image_data, use_container_width=True)
            else:
                st.info("暂无思维导图")
        
        # 显示历史图片
        if st.session_state.image_history:
            st.markdown("---")
            st.subheader("📚 历史图片")
            
            # 按类型分组显示历史图片
            history_illustrations = [item for item in st.session_state.image_history if item[2] == "illustration"]
            history_mindmaps = [item for item in st.session_state.image_history if item[2] == "mindmap"]
            
            if history_illustrations:
                st.markdown("#### 历史示意图")
                cols = st.columns(min(3, len(history_illustrations)))
                for idx, (img_data, prompt, img_type, model_info) in enumerate(history_illustrations[-3:]):  # 只显示最近3张
                    with cols[idx]:
                        thumbnail = img_data.copy()
                        thumbnail.thumbnail((100, 100))
                        st.image(thumbnail, caption=f"模型: {model_info}", use_container_width=True)
            
            if history_mindmaps:
                st.markdown("#### 历史思维导图")
                cols = st.columns(min(3, len(history_mindmaps)))
                for idx, (img_data, prompt, img_type) in enumerate(history_mindmaps[-3:]):  # 只显示最近3张
                    with cols[idx]:
                        thumbnail = img_data.copy()
                        thumbnail.thumbnail((100, 100))
                        st.image(thumbnail, use_container_width=True)
    
    # 输入框和图片生成按钮
    st.markdown("---")
    
    input_container = st.container()
    with input_container:
        col_input1, col_input2, col_input3 = st.columns([4, 1, 1])
        
        with col_input1:
            # 聊天输入框
            prompt = st.chat_input("请输入您的问题...")
            
            if prompt:
                question = prompt
                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.is_generating = True
                st.session_state.current_response = ""
                st.session_state.streaming_placeholder = None
                st.rerun()
        
        with col_input2:
            # 生成示意图按钮
            illustration_disabled = (
                st.session_state.is_generating or 
                not st.session_state.messages or 
                st.session_state.generating_image_type is not None
            )
            
            if st.button("🎨 生成示意图", disabled=not st.session_state.messages, use_container_width=True):
                assistant_indices = [i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"]
                if assistant_indices:
                    st.session_state.generating_image_type = "illustration"
                    st.session_state.target_assistant_idx = assistant_indices[-1]
                    st.rerun()
        
        with col_input3:
            # 生成思维导图按钮
            mindmap_disabled = (
                st.session_state.is_generating or 
                not st.session_state.messages or 
                st.session_state.generating_image_type is not None
            )
            
            if st.button("📊 生成思维导图", disabled=not st.session_state.messages, use_container_width=True):
                assistant_indices = [i for i, m in enumerate(st.session_state.messages) if m["role"] == "assistant"]
                if assistant_indices:
                    st.session_state.generating_image_type = "mindmap"
                    st.session_state.target_assistant_idx = assistant_indices[-1]
                    st.rerun()

    # 处理图片生成
    if st.session_state.generating_image_type and st.session_state.target_assistant_idx is not None:
        idx = st.session_state.target_assistant_idx
        response_text = st.session_state.messages[idx]["content"]
                
        if idx is not None:
            with st.spinner("正在生成图片，请稍候..."):
                try:
                    if st.session_state.generating_image_type == "illustration":
                        # 生成示意图
                        image_prompt = generate_image_prompt(response_text, "简洁明了的技术示意图", deepseek_api_key)
                        if not image_prompt.startswith("生成提示词时出错"):
                            # 根据选择的模型生成图片
                            image_data = generate_image(image_prompt, selected_image_model, siliconflow_api_key, P_API_KEY)
                            if image_data:
                                # 保存到当前消息
                                st.session_state.generated_images.setdefault(str(idx), {})
                                # 保存图片数据和模型信息
                                model_display_name = {
                                    "kolors": "Kolors (硅基流动)",
                                    "flux": "Flux (Pollinations)",
                                    "kontext": "Kontext (Pollinations)",
                                    "turbo": "Turbo (Pollinations)", 
                                    "gptimage": "GPTImage (Pollinations)"
                                }
                                model_info = model_display_name.get(selected_image_model, selected_image_model)
                                
                                st.session_state.generated_images[str(idx)]["illustration"] = (image_data, image_prompt, model_info)
                                
                                # 更新最新示意图
                                st.session_state.latest_illustration = (image_data, image_prompt, model_info)
                                
                                # 添加到历史记录
                                st.session_state.image_history.append((image_data, image_prompt, "illustration", model_info))
                                st.success(f"示意图生成成功！使用模型: {model_info}")
                                
                    elif st.session_state.generating_image_type == "mindmap":
                        # 生成思维导图
                        mermaid_code = generate_mermaid_code(response_text, deepseek_api_key)
                        if not mermaid_code.startswith("生成提示词时出错"):
                            image_data = generate_mermaid_image(mermaid_code)
                            if image_data:
                                st.session_state.generated_images.setdefault(str(idx), {})
                                st.session_state.generated_images[str(idx)]["mindmap"] = (image_data, mermaid_code)
                                # 更新最新思维导图
                                st.session_state.latest_mindmap = (image_data, mermaid_code)
                                    
                                # 添加到历史记录
                                st.session_state.image_history.append((image_data, mermaid_code, "mindmap"))
                                st.success("思维导图生成成功！")
                            else:
                                st.error("思维导图生成失败，请检查Mermaid代码或重试")
                        else:
                            st.error(f"生成思维导图代码失败: {mermaid_code}")
                
                except Exception as e:
                    st.error(f"图片生成失败: {str(e)}")
                
                finally:
                    st.session_state.generating_image_type = None
                    st.rerun()

    # 处理消息生成
    if st.session_state.is_generating and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        user_question = st.session_state.messages[-1]["content"]
        
        try:
            # 查询向量数据库
            results = query_vector_db(user_question, collection, siliconflow_api_key, selected_levels, n_results)
            
            # 处理检索结果
            documents_data = []
            if results['documents'] and results['documents'][0]:
                context = "\n\n".join(results['documents'][0])
                sources = results['documents'][0]
                distances = results['distances'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                
                # 构建文档数据
                for i, (source, distance) in enumerate(zip(sources, distances)):
                    doc_data = {
                        'data': {'document': source},
                        'distance': distance,
                        'metadata': metadatas[i] if i < len(metadatas) else {}
                    }
                    documents_data.append(doc_data)
            else:
                context = "未找到相关文档。"
                documents_data = []
            
            # 流式输出
            full_response = ""
            
            for chunk in call_deepseek_api_stream(user_question, context, deepseek_api_key):
                full_response += chunk
                st.session_state.current_response = full_response
                if st.session_state.streaming_placeholder is not None:
                    st.session_state.streaming_placeholder.markdown(full_response + "▌")
            
            if st.session_state.streaming_placeholder is not None:
                st.session_state.streaming_placeholder.markdown(full_response)
            
            # 将助手消息添加到聊天历史
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "documents": documents_data
            })
            
        except Exception as e:
            error_response = f"处理问题时出错: {str(e)}"
            st.session_state.messages.append({
                "role": "assistant", 
                "content": error_response,
                "documents": []
            })
        
        finally:
            st.session_state.is_generating = False
            st.session_state.current_response = ""
            st.session_state.streaming_placeholder = None
            st.rerun()

if __name__ == "__main__":
    main()
