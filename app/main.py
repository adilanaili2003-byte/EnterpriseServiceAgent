from fastapi import FastAPI

from app.api.routes.chat import router as chat_router


app = FastAPI(
    title="Enterprise Service Agent",
    description="企业级智能售后服务系统",
    version="1.0.0"
)


# 注册聊天路由
app.include_router(chat_router)


@app.get("/")
async def root():
    return {
        "message": "Enterprise Service Agent is running"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }