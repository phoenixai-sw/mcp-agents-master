# 파일명: app_KOR.py

import streamlit as st
import asyncio
import nest_asyncio
import json
import requests  # HTTP 요청을 위한 모듈

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

# 환경 변수 로드 (.env 파일 등, API 키 포함)
load_dotenv(override=True)

# 페이지 설정
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")
st.sidebar.markdown("### ✍️ Made by [테디노트](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()
st.title("🤖 Agent with MCP Tools")
st.markdown("✨ MCP 도구를 활용한 ReAct 에이전트에게 질문해보세요.")

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.mcp_client = None
    st.session_state.timeout_seconds = 120

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()


### 1. 인스턴스 메타데이터 API를 통한 공인 IP 조회 함수

def get_public_ip():
    """
    EC2 인스턴스 메타데이터 API를 통해 동적으로 공인 IP를 조회합니다.
    반환값: 공인 IP 문자열 (조회 실패 시 None)
    """
    try:
        metadata_url = "http://169.254.169.254/latest/meta-data/public-ipv4"
        response = requests.get(metadata_url, timeout=1)
        if response.status_code == 200:
            return response.text.strip()
        else:
            print(f"메타데이터 응답 코드: {response.status_code}")
    except Exception as e:
        print(f"공인 IP 조회 예외 발생: {e}")
    return None


### 2. MCP 초기화 함수 수정 (동적 IP 적용)

async def initialize_session(mcp_config=None):
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            await cleanup_mcp_client()
            if mcp_config is None:
                # 동적 공인 IP 조회
                public_ip = get_public_ip()
                if public_ip is None:
                    raise Exception("EC2 인스턴스의 공인 IP를 조회할 수 없습니다.")
                # 포트 8005는 고정, 공인 IP는 동적으로 가져옴
                mcp_config = {
                    "weather": {
                        "url": f"http://{public_ip}:8005",
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
                prompt="Use your tools to answer the question. Answer in Korean.",
            )
            st.session_state.agent = agent
            st.session_state.session_initialized = True
            return True
    except Exception as e:
        st.error(f"❌ 초기화 중 오류 발생: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


### 기타 기존 함수들은 그대로 유지
async def cleanup_mcp_client():
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback

def print_message():
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                if (i+1 < len(st.session_state.history)) and st.session_state.history[i+1]["role"] == "assistant_tool":
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.markdown(st.session_state.history[i+1]["content"])
                    i += 2
                else:
                    i += 1
        else:
            i += 1

def get_streaming_callback(text_placeholder, tool_placeholder):
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
        error_msg = f"❌ 쿼리 처리 중 오류 발생: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

# --- 사이드바 UI 등 나머지 코드는 기존 구조를 유지합니다 ---

with st.sidebar.expander("MCP 도구 추가", expanded=False):
    default_config = """{
  "weather": {
    "url": "http://43.200.183.196:8005",
    "transport": "sse"
  }
}"""
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = json.loads(st.session_state.get("mcp_config_text", default_config))
        except Exception as e:
            st.error(f"초기 pending config 설정 실패: {e}")
    st.subheader("개별 도구 추가")
    st.markdown(
        """
        **하나의 도구**를 JSON 형식으로 입력하세요:
        
        ```json
        {
          "도구이름": {
            "command": "실행 명령어",
            "args": ["인자1", "인자2", ...],
            "transport": "stdio"
          }
        }
        ```    
        ⚠️ **중요**: JSON은 반드시 중괄호({})로 감싸야 합니다.
        """
    )
    new_tool_json = st.text_area("도구 JSON", json.dumps({
        "github": {
            "command": "npx",
            "args": ["-y", "@smithery/cli@latest", "run", "@smithery-ai/github", "--config", '{"githubPersonalAccessToken":"your_token_here"}'],
            "transport": "stdio"
        }
    }, indent=2, ensure_ascii=False), height=250)
    if st.button("도구 추가", type="primary", key="add_tool_button", use_container_width=True):
        try:
            if not new_tool_json.strip().startswith("{") or not new_tool_json.strip().endswith("}"):
                st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                st.markdown('올바른 형식: `{ "도구이름": { ... } }`')
            else:
                parsed_tool = json.loads(new_tool_json)
                if "mcpServers" in parsed_tool:
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' 형식이 감지되었습니다. 자동으로 변환합니다.")
                if len(parsed_tool) == 0:
                    st.error("최소 하나 이상의 도구를 입력해주세요.")
                else:
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        if "url" in tool_config:
                            tool_config["transport"] = "sse"
                            st.info(f"'{tool_name}' 도구에 URL이 감지되어 transport를 'sse'로 설정했습니다.")
                        elif "transport" not in tool_config:
                            tool_config["transport"] = "stdio"
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(f"'{tool_name}' 도구 설정에는 'command' 또는 'url' 필드가 필요합니다.")
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(f"'{tool_name}' 도구 설정에는 'args' 필드가 필요합니다.")
                        elif "command" in tool_config and not isinstance(tool_config["args"], list):
                            st.error(f"'{tool_name}' 도구의 'args' 필드는 반드시 배열([]) 형식이어야 합니다.")
                        else:
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(f"{success_tools[0]} 도구가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(f"총 {len(success_tools)}개 도구({tool_names})가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")
        except Exception as e:
            st.error(f"오류 발생: {e}")
    st.divider()
    st.subheader("현재 도구 설정 (읽기 전용)")
    st.code(json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False))

with st.sidebar.expander("등록된 도구 목록", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception as e:
        st.error("유효한 MCP 도구 설정이 아닙니다.")
    else:
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("삭제", key=f"delete_{tool_name}"):
                del st.session_state.pending_mcp_config[tool_name]
                st.success(f"{tool_name} 도구가 삭제되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")

with st.sidebar:
    if st.button("도구설정 적용하기", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 변경사항을 적용하고 있습니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)
            st.session_state.mcp_config_text = json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
            progress_bar.progress(100)
            if success:
                st.success("✅ 새로운 MCP 도구 설정이 적용되었습니다.")
            else:
                st.error("❌ 새로운 MCP 도구 설정 적용에 실패하였습니다.")
        st.rerun()

if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버와 에이전트를 초기화합니다. 잠시만 기다려주세요...")
    success = st.session_state.event_loop.run_until_complete(initialize_session())
    if success:
        st.success(f"✅ 초기화 완료! {st.session_state.tool_count}개의 도구가 로드되었습니다.")
    else:
        st.error("❌ 초기화에 실패했습니다. 페이지를 새로고침해 주세요.")

print_message()

user_query = st.chat_input("💬 질문을 입력하세요")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user").markdown(user_query)
        with st.chat_message("assistant"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = st.session_state.event_loop.run_until_complete(
                process_query(user_query, text_placeholder, tool_placeholder, st.session_state.timeout_seconds)
            )
        if "error" in resp:
            st.error(resp["error"])
        else:
            st.session_state.history.append({"role": "user", "content": user_query})
            st.session_state.history.append({"role": "assistant", "content": final_text})
            if final_tool.strip():
                st.session_state.history.append({"role": "assistant_tool", "content": final_tool})
            st.rerun()
    else:
        st.warning("⏳ 시스템이 아직 초기화 중입니다. 잠시 후 다시 시도해주세요.")

with st.sidebar:
    st.subheader("🔧 시스템 정보")
    st.write(f"🛠️ MCP 도구 수: {st.session_state.get('tool_count', '초기화 중...')}")
    st.write("🧠 모델: Claude 3.7 Sonnet")
    st.subheader("⏱️ 타임아웃 설정")
    st.session_state.timeout_seconds = st.slider("응답 생성 제한 시간(초)", min_value=60, max_value=300, value=st.session_state.timeout_seconds, step=10, help="에이전트가 응답 생성하는 최대 시간을 설정합니다. 복잡한 작업은 더 긴 시간이 필요할 수 있습니다.")
    st.divider()
    if st.button("🔄 대화 초기화", use_container_width=True, type="primary"):
        st.session_state.thread_id = random_uuid()
        st.session_state.history = []
        st.success("✅ 대화가 초기화되었습니다.")
        st.rerun()
