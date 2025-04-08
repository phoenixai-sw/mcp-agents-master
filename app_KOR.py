# 파일명: app_KOR.py

import streamlit as st
import asyncio
import nest_asyncio
import json

nest_asyncio.apply()

if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_teddynote.messages import astream_graph, random_uuid
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# 환경 변수 로드 (.env 파일에서 API 키 등 로드)
load_dotenv(override=True)

# 페이지 설정: 제목, 아이콘, 레이아웃 구성
st.set_page_config(page_title="MCP 도구 연동 에이전트", page_icon="🧠", layout="wide")

# 사이드바 상단에 제작자 정보 (한글)
st.sidebar.markdown("### ✍️ 테디노트 제작")
st.sidebar.divider()

# 메인 페이지 타이틀 및 설명
st.title("🤖 MCP 도구 연동 에이전트")
st.markdown("✨ MCP 도구를 활용하여 ReAct 에이전트에 질문해보세요.")

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.mcp_client = None
    st.session_state.timeout_seconds = 120

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

# ======================================================================
# 수정: 고정 MCP 서버 접속 정보 - 고정 IP: 3.35.28.26, 포트: 8005, SSE 엔드포인트 "/sse" 적용
FIXED_PUBLIC_IP = "3.35.28.26"
MCP_PORT = "8005"
SSE_PATH = "/sse"
# ======================================================================

async def cleanup_mcp_client():
    """
    기존 MCP 클라이언트를 안전하게 종료합니다.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback
            # 필요 시 오류 로그를 출력할 수 있음
            # st.warning(f"MCP 클라이언트 종료 중 오류: {str(e)}")
            # st.warning(traceback.format_exc())

def print_message():
    """
    대화 기록을 화면에 출력합니다.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                if (i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"):
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2
                else:
                    i += 1
        else:
            i += 1

def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    스트리밍 콜백 함수를 생성합니다.
    """
    accumulated_text = []
    accumulated_tool = []
    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        message_content = message.get("content", None)
        if isinstance(message_content, AIMessageChunk):
            content = message_content.content
            if isinstance(content, list) and len(content) > 0:
                message_chunk = content[0]
                if message_chunk["type"] == "text":
                    accumulated_text.append(message_chunk["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                elif message_chunk["type"] == "tool_use":
                    if "partial_json" in message_chunk:
                        accumulated_tool.append(message_chunk["partial_json"])
                    else:
                        tool_call_chunks = message_content.tool_call_chunks
                        tool_call_chunk = tool_call_chunks[0]
                        accumulated_tool.append("\n```json\n" + str(tool_call_chunk) + "\n```\n")
                    with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                        st.markdown("".join(accumulated_tool))
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append("\n```json\n" + str(message_content.content) + "\n```\n")
            with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None
    return callback_func, accumulated_text, accumulated_tool

async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자의 질문을 처리하여 응답을 생성합니다.
    """
    try:
        if st.session_state.agent:
            streaming_callback, accumulated_text_obj, accumulated_tool_obj = get_streaming_callback(text_placeholder, tool_placeholder)
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(recursion_limit=100, thread_id=st.session_state.thread_id)
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                return {"error": error_msg}, error_msg, ""
            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return ({"error": "🚫 에이전트가 초기화되지 않았습니다."}, "🚫 에이전트가 초기화되지 않았습니다.", "")
    except Exception as e:
        import traceback
        error_msg = f"❌ 질문 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.
    """
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            await cleanup_mcp_client()
            if mcp_config is None:
                # 고정 IP와 "/sse" 엔드포인트를 사용하여 MCP 설정 구성
                mcp_config = {
                    "weather": {
                        "url": f"http://{FIXED_PUBLIC_IP}:{MCP_PORT}{SSE_PATH}",
                        "transport": "sse"
                    }
                }
            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            tools = client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            model = ChatAnthropic(
                model="claude-3-7-sonnet-latest",
                temperature=0.1,
                max_tokens=20000
            )
            agent = create_react_agent(
                model,
                tools,
                checkpointer=MemorySaver(),
                prompt="도구를 사용하여 질문에 답변하세요. 한국어로 답변해 주세요.",
            )
            st.session_state.agent = agent
            st.session_state.session_initialized = True
            return True
    except Exception as e:
        st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# --- 사이드바 UI: MCP 도구 추가 인터페이스 (모든 문구 한글로 변환) ---
with st.sidebar.expander("MCP 도구 추가", expanded=False):
    # 기본 설정: 고정 IP와 "/sse" 엔드포인트 적용
    default_config = """{
  "weather": {
    "url": "http://3.35.28.26:8005/sse",
    "transport": "sse"
  }
}"""
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = json.loads(
                st.session_state.get("mcp_config_text", default_config)
            )
        except Exception as e:
            st.error(f"초기 pending config 설정 실패: {e}")

    st.subheader("개별 도구 추가")
    st.markdown(
        """
하나의 도구를 **JSON 형식**으로 입력하세요:
        
```json
{
  "도구이름": {
    "command": "실행명령어",
    "args": ["인자1", "인자2", ...],
    "transport": "stdio"
  }
}
