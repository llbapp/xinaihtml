from openai import OpenAI
from src.data_processing import Data_process
from src.config.config import (
    select_num
)
from src.util.llm import load_llm, load_async_llm
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='app.log')

url = "https://u429062-8653-35f1c9ab.cqa1.seetacloud.com:8443/v1"

import time
from datetime import datetime
import asyncio

class Chatchain:
    def __init__(self):
        self.chat_histories = {}  # 使用uuid区分用户编号，构建不同的上下文队列
        self.last_activity = {}  # 记录每个会话的最后活动时间
        # 获取一个logger，使用类名作为logger的名字
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_chat_history(self, session_id):
        if session_id not in self.chat_histories:
            self.chat_histories[session_id] = []
        return self.chat_histories[session_id]

    def create_chat_session(self, message, content, session_id):
        openai, model = load_llm()
        max_tokens = 100
        user_history = self.get_chat_history(session_id)

        # 系统消息，只在第一次对话时添加
        if len(user_history) == 0:
            system_input = {
                "role": "system",
                "content": f"你是一个拥有丰富心理学知识的温柔邻家温柔女大学生心爱，我有一些心理问题，\
                请你用专业的知识和温柔、可爱、俏皮的口吻帮我解决，\
                回复中可以穿插一些可爱的Emoji表情符号或者文本符号。\n根据下面检索回来的信息，回答问题。{content}"
            }
            user_history.append(system_input)

        # 用户输入
        user_input = {
            "role": "user",
            "content": message
        }
        user_history.append(user_input)

        # 调用 OpenAI API 获取回复
        response = openai.chat.completions.create(
            model=model,
            messages=user_history,
            max_tokens=max_tokens
        )

        # 助手回复
        reply = response.choices[0].message.content
        assistant_input = {
            "role": "assistant",
            "content": reply
        }
        user_history.append(assistant_input)

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

        return reply

    async def create_stream_chat_session(self, message, content, session_id):
        client, model = await load_async_llm()
        max_tokens = 100
        user_history = self.get_chat_history(session_id)

        # 系统消息，只在第一次对话时添加
        if len(user_history) == 0:
            system_input = {
                "role": "system",
                "content": f"你是一个拥有丰富心理学知识的温柔邻家温柔女大学生心爱，我有一些心理问题，\
                请你用专业的知识和温柔、可爱、俏皮的口吻帮我解决，\
                回复中可以穿插一些可爱的Emoji表情符号或者文本符号。\n根据下面检索回来的信息，回答问题。{content}"
            }
            user_history.append(system_input)

        user_input = {
            "role": "user",
            "content": message
        }
        user_history.append(user_input)

        # 使用异步方式调用 OpenAI API
        stream = await client.chat.completions.create(
            model=model,
            messages=user_history,
            max_tokens=max_tokens,
            stream=True
        )

        full_response = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                yield content

        # 更新对话历史
        user_history.append({"role": "assistant", "content": full_response})

        # 更新最后活动时间
        self.last_activity[session_id] = time.time()

    def clean_inactive_sessions(self, inactive_threshold_minutes=30):
        """
        删除超过指定时间未活动的会话历史
        :param inactive_threshold_minutes: 不活跃阈值（分钟）
        """
        current_time = time.time()
        inactive_threshold = current_time - (inactive_threshold_minutes * 60)

        inactive_sessions = [
            session_id for session_id, last_active in self.last_activity.items()
            if last_active < inactive_threshold
        ]

        for session_id in inactive_sessions:
            del self.chat_histories[session_id]
            del self.last_activity[session_id]

        self.logger.info(f"Cleaned {len(inactive_sessions)} inactive sessions.")

    def get_active_sessions_count(self):
        """
        获取当前活跃会话数量
        """
        return len(self.chat_histories)

    def print_session_stats(self):
        """
        打印会话统计信息
        """
        total_sessions = len(self.chat_histories)
        current_time = datetime.now()

        self.logger.info(f"Total active sessions: {total_sessions}")
        for session_id, last_active in self.last_activity.items():
            last_active_time = datetime.fromtimestamp(last_active)
            inactive_duration = current_time - last_active_time
            # 获取会话历史
            session_history = self.chat_histories.get(session_id, [])
            
            # 格式化会话历史
            formatted_history = []
            for message in session_history:
                if isinstance(message, dict):
                    role = message.get('role', 'unknown')
                    content = message.get('content', '')
                    formatted_history.append(f"{role}: {content[:50]}...")  # 只显示前50个字符
                else:
                    formatted_history.append(str(message))
            
            history_summary = ' | '.join(formatted_history)
            
            self.logger.info(f"Session {session_id}: Last active {inactive_duration} ago")
            self.logger.info(f"Session Context: {history_summary}")
            self.logger.info("-" * 50)  # 分隔线

async def stats_context(chatchain, interval_minutes=1):
    while True:
        await asyncio.sleep(interval_minutes * 60)
        chatchain.print_session_stats()

async def periodic_cleanup(chatchain, interval_minutes=5):
    while True:
        await asyncio.sleep(interval_minutes * 60)
        chatchain.clean_inactive_sessions()
        chatchain.print_session_stats()

if __name__ == "__main__":
    chain = Chatchain()
    dp = Data_process()
    vector_db = dp.load_vector_db()
    
    while True:
        query = input("请输入：（输入exit退出对话）")
        if query == "exit":
            break

        docs, retriever = dp.retrieve(query, vector_db, k=10)
        content = dp.return_answer(query, docs, select_num)
        response = chain.create_chat_session(query, content)
        print(f'心爱：{response}')
