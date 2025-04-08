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

# 환경 변수 로드 (.env 파일에서 API 키 등의 설정을 가져옴)
load_dotenv(override=True)

# 페이지 설정: 제목, 아이콘, 레이아웃 구성
st.set_page_config(page_title="Agent with MCP Tools", page_icon="🧠", layout="wide")

# 사이드바 상단에 저자 정보 추가
st.sidebar.markdown("### ✍️ Made by [TeddyNote](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()

# 메인 페이지 타이틀 및 설명
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

# ======================================================================
# 고정 MCP 서버 접속 정보 (고정 IP: 3.35.28.26, 포트: 8005)
FIXED_PUBLIC_IP = "3.35.28.26"
MCP_PORT = "8005"
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
            # 오류 로그 출력을 원하면 여기에 추가

def print_message():
    """
    채팅 기록을 화면에 출력합니다.
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
                    with st.expander("🔧 Tool Call Information", expanded=False):
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
                    with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                        st.markdown("".join(accumulated_tool))
        elif isinstance(message_content, ToolMessage):
            accumulated_tool.append("\n```json\n" + str(message_content.content) + "\n```\n")
            with tool_placeholder.expander("🔧 Tool Call Information", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool

async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """
    사용자 질문을 처리하고 응답을 생성합니다.
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
                error_msg = f"⏱️ Request time exceeded {timeout_seconds} seconds. Please try again later."
                return {"error": error_msg}, error_msg, ""
            final_text = "".join(accumulated_text_obj)
            final_tool = "".join(accumulated_tool_obj)
            return response, final_text, final_tool
        else:
            return ({"error": "🚫 Agent has not been initialized."}, "🚫 Agent has not been initialized.", "")
    except Exception as e:
        import traceback
        error_msg = f"❌ Error during query processing: {str(e)}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.
    """
    try:
        with st.spinner("🔄 Connecting to MCP server..."):
            # 기존 MCP 클라이언트 정리
            await cleanup_mcp_client()
            if mcp_config is None:
                # 고정 IP를 사용하여 MCP 설정 구성
                mcp_config = {
                    "weather": {
                        "url": f"http://{FIXED_PUBLIC_IP}:{MCP_PORT}",
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
        st.error(f"❌ Error during initialization: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# --- Sidebar UI: MCP Tool Addition Interface ---
with st.sidebar.expander("Add MCP Tools", expanded=False):
    default_config = """{
  "weather": {
    "url": "http://3.35.28.196:8005",
    "transport": "sse"
  }
}"""
    # 고정 IP 설정은 위가 기본값; 필요 시, 세션에 저장된 값 사용
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = json.loads(
                st.session_state.get("mcp_config_text", default_config)
            )
        except Exception as e:
            st.error(f"Failed to set initial pending config: {e}")

    st.subheader("Add Individual Tool")
    st.markdown(
        """
    Enter **one tool** in JSON format:
    
    ```json
    {
      "tool_name": {
        "command": "execution_command",
        "args": ["arg1", "arg2", ...],
        "transport": "stdio"
      }
    }
    ```    
    ⚠️ **Important**: JSON must be enclosed in curly braces.
    """
    )

    example_json = {
        "github": {
            "command": "npx",
            "args": [
                "-y",
                "@smithery/cli@latest",
                "run",
                "@smithery-ai/github",
                "--config",
                '{"githubPersonalAccessToken":"your_token_here"}',
            ],
            "transport": "stdio",
        }
    }

    default_text = json.dumps(example_json, indent=2, ensure_ascii=False)

    new_tool_json = st.text_area(
        "Tool JSON",
        default_text,
        height=250,
    )

    if st.button("Add Tool", type="primary", key="add_tool_button", use_container_width=True):
        try:
            if not new_tool_json.strip().startswith("{") or not new_tool_json.strip().endswith("}"):
                st.error("JSON must start and end with curly braces ({}).")
                st.markdown('Correct format: `{ "tool_name": { ... } }`')
            else:
                parsed_tool = json.loads(new_tool_json)
                if "mcpServers" in parsed_tool:
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' format detected. Converting automatically.")
                if len(parsed_tool) == 0:
                    st.error("Please enter at least one tool.")
                else:
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        if "url" in tool_config:
                            tool_config["transport"] = "sse"
                            st.info(f"URL detected in '{tool_name}' tool, setting transport to 'sse'.")
                        elif "transport" not in tool_config:
                            tool_config["transport"] = "stdio"
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(f"'{tool_name}' tool configuration requires either 'command' or 'url' field.")
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(f"'{tool_name}' tool configuration requires 'args' field.")
                        elif "command" in tool_config and not isinstance(tool_config["args"], list):
                            st.error(f"'args' field in '{tool_name}' tool must be an array ([]) format.")
                        else:
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(f"{success_tools[0]} tool has been added. Press 'Apply' button to apply changes.")
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(f"Total {len(success_tools)} tools ({tool_names}) have been added. Press 'Apply' to apply changes.")
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {e}")
            st.markdown(
                """
            **How to fix**:
            1. Check the JSON format.
            2. All keys and string values must be wrapped in double quotes.
            3. Escape double quotes within strings.
            """
            )
        except Exception as e:
            st.error(f"Error occurred: {e}")

    st.divider()
    st.subheader("Current Tool Settings (Read-only)")
    st.code(json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False))

with st.sidebar.expander("Registered Tools List", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception as e:
        st.error("Not a valid MCP tool configuration.")
    else:
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("Delete", key=f"delete_{tool_name}"):
                del st.session_state.pending_mcp_config[tool_name]
                st.success(f"{tool_name} tool has been deleted. Press 'Apply' to apply changes.")

with st.sidebar:
    if st.button("Apply Tool Settings", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 Applying changes. Please wait...")
            progress_bar = st.progress(0)
            st.session_state.mcp_config_text = json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
            progress_bar.progress(100)
            if success:
                st.success("✅ New MCP tool settings have been applied.")
            else:
                st.error("❌ Failed to apply new MCP tool settings.")
        st.rerun()

if not st.session_state.session_initialized:
    st.info("🔄 Initializing MCP server and agent. Please wait...")
    success = st.session_state.event_loop.run_until_complete(initialize_session())
    if success:
        st.success(f"✅ Initialization complete! {st.session_state.tool_count} tools loaded.")
    else:
        st.error("❌ Initialization failed. Please refresh the page.")

print_message()

user_query = st.chat_input("💬 Enter your question")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user").markdown(user_query)
        with st.chat_message("assistant"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            resp, final_text, final_tool = st.session_state.event_loop.run_until_complete(
                process_query(
                    user_query,
                    text_placeholder,
                    tool_placeholder,
                    st.session_state.timeout_seconds,
                )
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
        st.warning("⏳ System is still initializing. Please try again in a moment.")

with st.sidebar:
    st.subheader("🔧 System Information")
    st.write(f"🛠️ MCP Tool Count: {st.session_state.get('tool_count', 'Initializing...')}")
    st.write("🧠 Model: Claude 3.7 Sonnet")

    st.subheader("⏱️ Timeout Settings")
    st.session_state.timeout_seconds = st.slider(
        "Response generation time limit (seconds)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="Set the maximum time for the agent to generate a response. Complex tasks may require more time.",
    )

    st.divider()

    if st.button("🔄 Reset Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = random_uuid()
        st.session_state.history = []
        st.success("✅ Conversation has been reset.")
        st.rerun()
