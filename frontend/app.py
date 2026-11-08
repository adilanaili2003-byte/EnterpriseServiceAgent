import os
from uuid import uuid4

import httpx
import streamlit as st


# ==================== 页面配置 ====================

st.set_page_config(
    page_title="Enterprise Service Agent",
    page_icon="🤖",
    layout="wide",
)

API_BASE_URL = os.getenv(
    "BACKEND_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")


# ==================== 会话状态初始化 ====================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==================== 后端接口 ====================

@st.cache_data(ttl=15)
def check_backend() -> bool:
    """检查 FastAPI 服务是否正常。"""
    try:
        response = httpx.get(
            f"{API_BASE_URL}/health",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json().get("status") == "healthy"
    except (httpx.HTTPError, ValueError):
        return False


def send_message(message: str) -> str:
    """向 FastAPI 发送消息并获取 Agent 回复。"""
    response = httpx.post(
        f"{API_BASE_URL}/api/v1/chat",
        json={
            "message": message,
            "session_id": st.session_state.session_id,
        },
        timeout=120.0,
    )

    response.raise_for_status()

    return response.json()["reply"]

def hide_policy_reference(reply: str) -> str:
    """隐藏回答末尾的政策文件引用。"""
    lines = reply.splitlines()

    while lines and lines[-1].strip().startswith("参考政策："):
        lines.pop()

    return "\n".join(lines).rstrip()


# ==================== 侧边栏 ====================

with st.sidebar:
    st.title("🤖 Enterprise Agent")
    st.caption("企业智能客服工作台")

    st.divider()

    if st.button("＋ 新建会话", use_container_width=True):
        st.session_state.session_id = str(uuid4())
        st.session_state.messages = []
        st.rerun()

    st.subheader("当前会话")

    st.code(st.session_state.session_id)

    st.divider()

    st.subheader("服务状态")

    if check_backend():
        st.success("后端服务运行正常")
    else:
        st.error("后端服务暂不可用")

    st.caption("Powered by LangGraph & DeepSeek")


# ==================== 主页面 ====================

st.title("企业智能客服 Agent")

st.markdown(
    "你好！我是企业智能客服助手，可以帮助你查询订单、"
    "了解售后政策以及解答相关问题。"
)

st.divider()


# ==================== 历史消息展示 ====================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]

        if message["role"] == "assistant":
            content = hide_policy_reference(content)

        st.markdown(content)


# ==================== 消息输入与处理 ====================

user_input = st.chat_input("请输入你的问题，例如：我的订单可以申请退款吗？")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("智能客服 正在思考，请稍候..."):
            try:
                reply = send_message(user_input)
                display_reply = hide_policy_reference(reply)

                st.markdown(display_reply)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": display_reply,
                })

            except httpx.HTTPStatusError as exc:
                detail = "后端请求失败"

                try:
                    raw_detail = exc.response.json().get("detail")
                except ValueError:
                    raw_detail = None

                if isinstance(raw_detail, list) and raw_detail:
                    first = raw_detail[0]
                    detail = (
                        first.get("msg", detail)
                        if isinstance(first, dict)
                        else str(first)
                    )
                elif isinstance(raw_detail, str) and raw_detail:
                    detail = raw_detail

                st.error(f"请求失败：{detail}")

            except httpx.TimeoutException:
                st.error("请求超时，后端响应时间过长，请稍后重试。")

            except httpx.HTTPError:
                st.error(
                    "无法连接后端服务，请检查 FastAPI 是否已经启动。"
                )

            except (KeyError, ValueError):
                st.error("后端返回的数据格式不正确。")