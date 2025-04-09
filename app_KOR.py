import streamlit as st
import asyncio
import nest_asyncio
import json
import os

# nest_asyncio 적용: 이미 실행 중인 이벤트 루프 내에서 중첩 호출 허용
nest_asyncio.apply()

# 전역 이벤트 루프 생성 및 재사용 (한번 생성한 후 계속 사용)
if "event_loop" not in st.session_state:
    loop = asyncio.new_event_loop()
    st.session_state.event_loop = loop
    asyncio.set_event_loop(loop)

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
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

# 사이드바 최상단에 저자 정보 추가
st.sidebar.markdown("### ✍️ Made by [테디노트](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()  # 구분선 추가

# --- 사이드바: LLM 모델 선택 ---
model_options = ["claude 3.7sonnet", "gpt-4o", "gemini-2.0", "Grok3", "Llamma3.3", "phoenix-1.0"]
selected_model = st.sidebar.selectbox("LLM 모델 선택", model_options, index=0, key="model_choice")
# 지원되지 않는 모델에 대한 경고 메시지 표시
if selected_model not in ["claude 3.7sonnet", "gpt-4o"]:
    st.sidebar.warning("⚠️ 선택한 모델은 현재 지원되지 않습니다. Claude 3.7sonnet 또는 GPT-4o만 사용 가능합니다.")
# 모델 선택 변경 시 세션 재초기화
if "current_model" not in st.session_state:
    st.session_state.current_model = selected_model
elif st.session_state.current_model != selected_model:
    st.session_state.current_model = selected_model
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.mcp_client = None

# 기존 페이지 제목 및 설명
st.title("🤖 Agent with MCP Tools")
st.markdown("✨ MCP 도구를 활용한 ReAct 에이전트에게 질문해보세요.")

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False
    st.session_state.agent = None
    st.session_state.history = []
    st.session_state.mcp_client = None
    st.session_state.timeout_seconds = 120  # 응답 생성 제한 시간(초)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

# --- 함수 정의 부분 ---

async def cleanup_mcp_client():
    """기존 MCP 클라이언트를 안전하게 종료합니다."""
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception:
            # MCP 클라이언트 종료 중 발생한 오류는 무시 (필요 시 로그 처리 가능)
            pass

def print_message():
    """현재까지의 대화 기록을 화면에 출력합니다."""
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # Assistant 메시지 출력 컨테이너 생성
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                # 바로 다음 메시지가 도구 호출 정보인지 확인
                if i + 1 < len(st.session_state.history) and st.session_state.history[i + 1]["role"] == "assistant_tool":
                    # 도구 호출 정보는 expander로 같은 컨테이너 내에 표시
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # 메시지 2개(user 질문+tool 정보)를 한 번에 처리
                else:
                    i += 1
        else:
            # "assistant_tool" 역할의 메시지는 위에서 함께 처리하므로 건너뜀
            i += 1

def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    스트리밍 응답 생성을 처리하는 콜백 함수를 생성합니다.
    - text_placeholder: 실시간 생성되는 텍스트 응답 표시용
    - tool_placeholder: 실시간 생성되는 도구 호출 정보 표시용
    """
    accumulated_text = []
    accumulated_tool = []

    def callback_func(message: dict):
        nonlocal accumulated_text, accumulated_tool
        content = message.get("content", None)
        if isinstance(content, AIMessageChunk):
            chunk = content.content
            if isinstance(chunk, list) and len(chunk) > 0:
                part = chunk[0]
                if part["type"] == "text":
                    # 텍스트 응답 조각
                    accumulated_text.append(part["text"])
                    text_placeholder.markdown("".join(accumulated_text))
                elif part["type"] == "tool_use":
                    # 도구 호출 조각
                    if "partial_json" in part:
                        accumulated_tool.append(part["partial_json"])
                    else:
                        tool_call_chunk = content.tool_call_chunks[0]
                        accumulated_tool.append("\n```json\n" + str(tool_call_chunk) + "\n```\n")
                    with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                        st.markdown("".join(accumulated_tool))
        elif isinstance(content, ToolMessage):
            # 최종 도구 호출 결과 메시지
            accumulated_tool.append("\n```json\n" + str(content.content) + "\n```\n")
            with tool_placeholder.expander("🔧 도구 호출 정보", expanded=True):
                st.markdown("".join(accumulated_tool))
        return None

    return callback_func, accumulated_text, accumulated_tool

async def process_query(query, text_placeholder, tool_placeholder, timeout_seconds=60):
    """사용자 질문을 받아 에이전트의 응답을 생성하고 반환합니다."""
    try:
        if st.session_state.agent:
            # 스트리밍 콜백 및 누적 버퍼 초기화
            streaming_callback, accumulated_text, accumulated_tool = get_streaming_callback(text_placeholder, tool_placeholder)
            try:
                response = await asyncio.wait_for(
                    astream_graph(
                        st.session_state.agent,
                        {"messages": [HumanMessage(content=query)]},
                        callback=streaming_callback,
                        config=RunnableConfig(recursion_limit=100, thread_id=st.session_state.thread_id),
                    ),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
                error_msg = f"⏱️ 요청 시간이 {timeout_seconds}초를 초과했습니다. 나중에 다시 시도해 주세요."
                return {"error": error_msg}, error_msg, ""
            # 최종 응답 취합
            final_text = "".join(accumulated_text)
            final_tool = "".join(accumulated_tool)
            return response, final_text, final_tool
        else:
            return ({"error": "🚫 에이전트가 초기화되지 않았습니다."}, "🚫 에이전트가 초기화되지 않았습니다.", "")
    except Exception as e:
        import traceback
        error_msg = f"❌ 쿼리 처리 중 오류 발생: {e}\n{traceback.format_exc()}"
        return {"error": error_msg}, error_msg, ""

async def initialize_session(mcp_config=None):
    """
    MCP 클라이언트와 LangChain 에이전트를 초기화합니다.
    - mcp_config: MCP 도구 설정 (JSON dict). None이면 기본 원격 설정 사용.
    반환: 초기화 성공(bool)
    """
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            # 기존 MCP 클라이언트 종료 처리
            await cleanup_mcp_client()
            if mcp_config is None:
                # 기본 설정: 원격 MCP 서버(SSE) Weather 툴
                mcp_config = {
                    "weather": {
                        "url": "http://3.35.28.26:8005/sse",
                        "transport": "sse",
                    }
                }
            # MCP 멀티서버 클라이언트 생성 및 접속
            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            tools = client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            # LLM 모델 선택에 따라 적절한 LangChain Chat 객체 생성
            model_choice = st.session_state.get("model_choice", "claude 3.7sonnet")
            if model_choice == "claude 3.7sonnet":
                model = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.1, max_tokens=20000)
            elif model_choice == "gpt-4o":
                model = ChatOpenAI(model="gpt-4o", temperature=0.1, max_tokens=20000)
            else:
                # 지원되지 않는 모델인 경우 초기화 실패 처리
                st.error("🚫 선택한 LLM 모델은 지원되지 않습니다. 다른 모델을 선택해주세요.")
                return False

            # ReAct 에이전트 생성 (LangGraph)
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
        st.error(f"❌ 초기화 중 오류 발생: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False

# --- 사이드바: MCP 도구 추가 ---
with st.sidebar.expander("도구 추가", expanded=False):
    default_config = """{
  "weather": {
    "url": "http://3.35.28.26:8005/sse",
    "transport": "sse"
  }
}"""
    # session_state에 pending 설정이 없으면 기본값 사용
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
    ⚠️ **중요**: JSON을 반드시 중괄호(`{}`)로 감싸야 합니다.
    """
    )

    # 예시 JSON (GitHub 툴 예제)
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
    new_tool_json = st.text_area("도구 JSON", default_text, height=250)

    # "도구 추가" 버튼 처리
    if st.button("도구 추가", type="primary", key="add_tool_button", use_container_width=True):
        try:
            # JSON 문자열 기본 검증
            if not new_tool_json.strip().startswith("{") or not new_tool_json.strip().endswith("}"):
                st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                st.markdown('올바른 형식: `{ "도구이름": { ... } }`')
            else:
                parsed_tool = json.loads(new_tool_json)
                # 혹시 최상위에 "mcpServers" 키가 있으면 내부 내용으로 대체
                if "mcpServers" in parsed_tool:
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' 형식이 감지되어 내부 도구 설정으로 변환했습니다.")
                if len(parsed_tool) == 0:
                    st.error("최少 하나 이상의 도구를 입력해주세요.")
                else:
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        # URL 존재 시 transport를 "sse"로 강제
                        if "url" in tool_config:
                            tool_config["transport"] = "sse"
                            st.info(f"'{tool_name}' 도구에 URL이 있어 transport를 'sse'로 설정했습니다.")
                        elif "transport" not in tool_config:
                            tool_config["transport"] = "stdio"  # transport 누락 시 기본 "stdio"
                        # 필수 필드 검증
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(f"'{tool_name}' 설정에 'command' 또는 'url' 필드가 필요합니다.")
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(f"'{tool_name}' 설정에 'args' 필드가 필요합니다.")
                        elif "command" in tool_config and not isinstance(tool_config["args"], list):
                            st.error(f"'{tool_name}'의 'args'는 리스트 형식이어야 합니다.")
                        else:
                            # pending 설정에 도구 추가
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)
                    # 추가 성공 알림
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(f"{success_tools[0]} 도구가 추가되었습니다. 적용하려면 '도구 설정 적용' 버튼을 눌러주세요.")
                        else:
                            tool_list = ", ".join(success_tools)
                            st.success(f"총 {len(success_tools)}개 도구({tool_list})가 추가되었습니다. 적용하려면 '도구 설정 적용' 버튼을 눌러주세요.")
        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 에러: {e}")
            st.markdown(
                """
            **수정 방법**:  
            1. JSON 형식이 올바른지 확인하세요.  
            2. 모든 키와 문자열 값을 큰따옴표(")로 감싸야 합니다.  
            3. 문자열 내에 큰따옴표를 포함해야 한다면 `\"`처럼 이스케이프 처리를 해야 합니다.
            """
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")

    st.divider()
    st.subheader("현재 도구 설정 (읽기 전용)")
    st.code(json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False))

# --- 등록된 도구 목록 및 삭제 ---
with st.sidebar.expander("등록된 도구 목록", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception:
        st.error("유효한 MCP 도구 설정이 아닙니다.")
    else:
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("삭제", key=f"delete_{tool_name}"):
                del st.session_state.pending_mcp_config[tool_name]
                st.success(f"{tool_name} 도구가 삭제되었습니다. 적용하려면 '도구 설정 적용' 버튼을 눌러주세요.")

with st.sidebar:
    # "도구 설정 적용" 버튼: pending 설정을 실제 반영하여 재초기화
    if st.button("도구 설정 적용", key="apply_button", type="primary", use_container_width=True):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 변경사항을 적용하고 있습니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)
            # 새로운 설정을 세션 상태에 저장
            st.session_state.mcp_config_text = json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
            # 세션 재초기화 플래그 설정
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            # 에이전트 재생성
            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
            progress_bar.progress(100)
            if success:
                st.success("✅ 새로운 MCP 도구 설정이 적용되었습니다.")
            else:
                st.error("❌ 새로운 MCP 도구 설정 적용에 실패하였습니다.")
        st.rerun()  # 페이지 재실행

# --- 초기 세션 자동 초기화 ---
if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버와 에이전트를 초기화합니다. 잠시만 기다려주세요...")
    success = st.session_state.event_loop.run_until_complete(initialize_session())
    if success:
        st.success(f"✅ 초기화 완료! {st.session_state.tool_count}개의 도구가 로드되었습니다.")
    else:
        # 모델 미지원으로 실패한 경우와 기타 오류 구분
        if st.session_state.get("model_choice") not in ["claude 3.7sonnet", "gpt-4o"]:
            st.error("❌ 선택한 LLM 모델은 현재 지원될 수 없습니다. 지원되는 모델로 변경해주세요.")
        else:
            st.error("❌ 초기화에 실패했습니다. 페이지를 새로고침해 주세요.")

# --- 대화 기록 출력 ---
print_message()

# --- 사용자 입력 처리 ---
user_query = st.chat_input("💬 질문을 입력하세요")
if user_query:
    if st.session_state.session_initialized:
        st.chat_message("user").markdown(user_query)
        with st.chat_message("assistant"):
            tool_placeholder = st.empty()
            text_placeholder = st.empty()
            # 비동기 에이전트 응답 생성
            resp, final_text, final_tool = st.session_state.event_loop.run_until_complete(
                process_query(user_query, text_placeholder, tool_placeholder, st.session_state.timeout_seconds)
            )
            # 에러 발생 시 에러 메시지 표시
            if isinstance(resp, dict) and resp.get("error"):
                st.markdown(final_text)
            else:
                # 최종 응답 출력
                st.markdown(final_text)
    else:
        st.error("🚫 에이전트가 초기화되지 않았습니다. 좌측에서 설정을 확인하고 다시 시도하세요.")
