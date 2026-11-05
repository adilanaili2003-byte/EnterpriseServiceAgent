from fastapi import FastAPI

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Enterprise Service Agent",
    description="基于 LangGraph 的企业级智能售后服务系统",
    version="0.1.0"
)


@app.get("/")
async def root():
    """项目首页"""
    return {
        "message": "Enterprise Service Agent is running!",
        "version": "0.1.0"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "service": "Enterprise Service Agent"
    }