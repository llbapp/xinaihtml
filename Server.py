# server.py
import uuid
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, WebSocket, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd

from src.pipeline import Chatchain, periodic_cleanup, stats_context
from src.data_processing_for_server import data_processor
from src.config.config import select_num

class Query(BaseModel):
    text: str

class BatchQuery(BaseModel):
    queries: List[str]

class AsyncMultiprocessChatbot:
    def __init__(self):
        self.chain = Chatchain()
        self.dp = data_processor  # 使用预初始化的实例

    async def process_query(self, query, session_id):
        docs = await asyncio.to_thread(self.dp.retrieve, query, k=10)
        content = await asyncio.to_thread(self.dp.return_answer, query, docs, select_num)
        response = await asyncio.to_thread(self.chain.create_chat_session, query, content, session_id)
        return query, response

    async def stream_query(self, query, session_id):
        docs = await asyncio.to_thread(self.dp.retrieve, query, k=10)
        content = await asyncio.to_thread(self.dp.return_answer, query, docs, select_num)
        async for response in self.chain.create_stream_chat_session(query, content, session_id):
            yield response

chatbot = AsyncMultiprocessChatbot()

@asynccontextmanager
async def lifespan(app: FastAPI):
    #每30分钟清理不活跃的session，防止内存爆炸
    cleanup_task = asyncio.create_task(periodic_cleanup(chatbot.chain))
    #每分钟运行一次对session的统计
    static_task = asyncio.create_task(stats_context(chatbot.chain))
    yield
    # 关闭时运行的代码
    cleanup_task.cancel()
    static_task.cancel()
    try:
        await cleanup_task
        await static_task
    except asyncio.CancelledError:
        pass

app = FastAPI(lifespan=lifespan)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中，应该指定允许的源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 会话ID处理
async def get_or_create_session_id(x_session_id: str = Header(None)):
    if not x_session_id:
        x_session_id = str(uuid.uuid4())
    return x_session_id

@app.post("/chat")
async def chat(query: Query, session_id: str = Depends(get_or_create_session_id)):
    query, response = await chatbot.process_query(query.text, session_id)
    return {"query": query, "response": response}

@app.post("/batch_chat")
async def batch_chat(batch_query: BatchQuery, background_tasks: BackgroundTasks, session_id: str = Depends(get_or_create_session_id)):
    tasks = [chatbot.process_query(query, session_id) for query in batch_query.queries]
    results = await asyncio.gather(*tasks)
    background_tasks.add_task(save_to_excel, results)
    return [{"query": query, "response": response} for query, response in results]

@app.websocket("/stream_chat")
async def stream_chat(websocket: WebSocket):
    await websocket.accept()
    session_id = await get_or_create_session_id(websocket.headers.get("x-session-id"))
    while True:
        try:
            data = await websocket.receive_text()
            query = data.strip()
            async for response in chatbot.stream_query(query, session_id):
                await websocket.send_text(response)
        except Exception as e:
            print(f"WebSocket error: {e}")
            break

def save_to_excel(qa_list):
    df = pd.DataFrame(qa_list, columns=['用户提问', 'AI回答'])
    df.to_excel('对话记录.xlsx', index=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)