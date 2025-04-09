알겠습니다. MCP 도구들이 외부에서 JSON 형식으로 제공되며 이를 내부 시스템에 업로드하여 사용할 수 있도록 설계된 구조를 반영해, Streamlit 인터페이스와 에이전트가 해당 도구들을 정확히 인식하고 연동할 수 있도록 전체 코드를 수정하겠습니다.

또한, 사용자 질문이 '당신은 무엇을 할 수 있어?'일 경우 현재 MCP 도구의 로딩 여부에 따라 적절한 응답이 나가도록 로직도 포함하겠습니다.

코드가 준비되는 대로 전체 수정본을 제공드리겠습니다.

```python
import streamlit as st
import asyncio
import nest_asyncio
import json

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

# 사이드바 최상단에 저자 정보 추가 (다른 사이드바 요소보다 먼저 배치)
st.sidebar.markdown("### ✍️ Made by [테디노트](https://youtube.com/c/teddynote) 🚀")
st.sidebar.divider()  # 구분선 추가

# 기존 페이지 타이틀 및 설명
st.title("🤖 Agent with MCP Tools")
st.markdown("✨ MCP 도구를 활용한 ReAct 에이전트에게 질문해보세요.")

# 세션 상태 초기화
if "session_initialized" not in st.session_state:
    st.session_state.session_initialized = False  # 세션 초기화 상태 플래그
    st.session_state.agent = None  # ReAct 에이전트 객체 저장 공간
    st.session_state.history = []  # 대화 기록 저장 리스트
    st.session_state.mcp_client = None  # MCP 클라이언트 객체 저장 공간
    st.session_state.timeout_seconds = 120  # 응답 생성 제한 시간(초), 기본값 120초

if "thread_id" not in st.session_state:
    st.session_state.thread_id = random_uuid()

# --- 함수 정의 부분 ---

async def cleanup_mcp_client():
    """
    기존 MCP 클라이언트를 안전하게 종료합니다.

    기존 클라이언트가 있는 경우 정상적으로 리소스를 해제합니다.
    """
    if "mcp_client" in st.session_state and st.session_state.mcp_client is not None:
        try:
            await st.session_state.mcp_client.__aexit__(None, None, None)
            st.session_state.mcp_client = None
        except Exception as e:
            import traceback
            # MCP 클라이언트 종료 중 발생한 오류를 무시하거나 로깅만 수행
            # st.warning(f"MCP 클라이언트 종료 중 오류: {str(e)}")
            # st.warning(traceback.format_exc())

def print_message():
    """
    채팅 기록을 화면에 출력합니다.

    사용자와 어시스턴트의 메시지를 구분하여 화면에 표시하고,
    도구 호출 정보는 어시스턴트 메시지 컨테이너 내에 표시합니다.
    """
    i = 0
    while i < len(st.session_state.history):
        message = st.session_state.history[i]
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
            i += 1
        elif message["role"] == "assistant":
            # 어시스턴트 메시지 컨테이너 생성
            with st.chat_message("assistant"):
                # 어시스턴트 메시지 내용 표시
                st.markdown(message["content"])
                # 다음 메시지가 도구 호출 정보인지 확인
                if (
                    i + 1 < len(st.session_state.history)
                    and st.session_state.history[i + 1]["role"] == "assistant_tool"
                ):
                    # 도구 호출 정보를 동일한 컨테이너 내에 expander로 표시
                    with st.expander("🔧 도구 호출 정보", expanded=False):
                        st.markdown(st.session_state.history[i + 1]["content"])
                    i += 2  # 두 메시지를 함께 처리했으므로 2 증가
                else:
                    i += 1  # 일반 메시지만 처리했으므로 1 증가
        else:
            # assistant_tool 메시지는 위에서 처리되므로 건너뜀
            i += 1

def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    스트리밍 콜백 함수를 생성합니다.

    매개변수:
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트

    반환값:
        callback_func: 스트리밍 콜백 함수
        accumulated_text: 누적된 텍스트 응답을 저장하는 리스트
        accumulated_tool: 누적된 도구 호출 정보를 저장하는 리스트
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
    사용자 질문을 처리하고 응답을 생성합니다.

    매개변수:
        query: 사용자가 입력한 질문 텍스트
        text_placeholder: 텍스트 응답을 표시할 Streamlit 컴포넌트
        tool_placeholder: 도구 호출 정보를 표시할 Streamlit 컴포넌트
        timeout_seconds: 응답 생성 제한 시간(초)
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
                        config=RunnableConfig(recursion_limit=100, thread_id=st.session_state.thread_id),
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
            return (
                {"error": "🚫 에이전트가 초기화되지 않았습니다."},
                "🚫 에이전트가 초기화되지 않았습니다.",
                "",
            )
    except Exception as e:
        import traceback
        err_str = str(e)
        lower_err = err_str.lower()
        if ("api key" in lower_err or "401" in lower_err or "unauthorized" in lower_err or "forbidden" in lower_err):
            # API 키 관련 오류인 경우 사용자에게 친절한 메시지 출력
            if "gpt" in st.session_state.get("selected_model", "").lower() or "openai" in st.session_state.get("selected_model", "").lower():
                user_error_msg = "❌ OpenAI API 키 오류: 제공된 OpenAI API 키가 유효하지 않거나 권한이 없습니다. 환경 변수 설정을 확인해주세요."
            elif "claude" in st.session_state.get("selected_model", "").lower() or "anthropic" in st.session_state.get("selected_model", "").lower():
                user_error_msg = "❌ Anthropic API 키 오류: 제공된 Anthropic API 키가 유효하지 않거나 권한이 없습니다. 환경 변수 설정을 확인해주세요."
            else:
                user_error_msg = "❌ API 키 인증 오류: 제공된 API 키가 유효하지 않거나 권한이 없습니다. 환경 변수 설정을 확인해주세요."
            return {"error": user_error_msg}, user_error_msg, ""
        else:
            error_msg = f"❌ 쿼리 처리 중 오류 발생: {err_str}\n{traceback.format_exc()}"
            return {"error": error_msg}, error_msg, ""

async def initialize_session(mcp_config=None):
    """
    MCP 세션과 에이전트를 초기화합니다.

    매개변수:
        mcp_config: MCP 도구 설정 정보(JSON). None인 경우 기본 설정 사용

    반환값:
        bool: 초기화 성공 여부
    """
    try:
        with st.spinner("🔄 MCP 서버에 연결 중..."):
            # 먼저 기존 클라이언트를 안전하게 정리
            await cleanup_mcp_client()

            if mcp_config is None:
                # 기본 설정 사용 (예: weather 도구)
                mcp_config = {
                    "weather": {
                        "command": "python",
                        "args": ["./mcp_server_local.py"],
                        "transport": "stdio",
                    },
                }
            client = MultiServerMCPClient(mcp_config)
            await client.__aenter__()
            tools = client.get_tools()
            st.session_state.tool_count = len(tools)
            st.session_state.mcp_client = client

            # 사용자가 선택한 LLM 모델에 따라 모델 객체 생성
            model_name = st.session_state.get("selected_model", "Claude 3.7 Sonnet")
            if model_name is None:
                model_name = "Claude 3.7 Sonnet"
            model_obj = None
            mn_lower = model_name.lower()
            if "claude" in mn_lower:
                model_obj = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.1, max_tokens=20000)
            elif "gpt-4" in mn_lower or "openai" in mn_lower:
                model_obj = ChatOpenAI(model="gpt-4", temperature=0.1, max_tokens=8000)
            elif any(x in mn_lower for x in ["gemini", "grok", "llama", "phoenix"]):
                # 아직 지원되지 않는 모델이 선택된 경우
                raise Exception("선택한 LLM 모델은 현재 지원되지 않습니다.")
            else:
                # 알 수 없는 값인 경우 기본값 사용
                model_obj = ChatAnthropic(model="claude-3-7-sonnet-latest", temperature=0.1, max_tokens=20000)

            # ReAct 에이전트 생성
            agent = create_react_agent(
                model_obj,
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

# --- 사이드바 UI: MCP 도구 추가 ---
with st.sidebar.expander("MCP 도구 추가", expanded=False):
    default_config = """{
  "weather": {
    "command": "python",
    "args": ["./mcp_server_local.py"],
    "transport": "stdio"
  }
}"""
    # pending config가 없으면 기존 mcp_config_text 기반으로 생성
    if "pending_mcp_config" not in st.session_state:
        try:
            st.session_state.pending_mcp_config = json.loads(
                st.session_state.get("mcp_config_text", default_config)
            )
        except Exception as e:
            st.error(f"초기 pending config 설정 실패: {e}")

    # 개별 도구 추가를 위한 UI
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

    # 보다 명확한 예시 제공
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
        "도구 JSON",
        default_text,
        height=250,
    )

    # 추가하기 버튼
    if st.button(
        "도구 추가",
        type="primary",
        key="add_tool_button",
        use_container_width=True,
    ):
        try:
            # 입력값 검증
            if not new_tool_json.strip().startswith("{") or not new_tool_json.strip().endswith("}"):
                st.error("JSON은 중괄호({})로 시작하고 끝나야 합니다.")
                st.markdown('올바른 형식: `{ "도구이름": { ... } }`')
            else:
                # JSON 파싱
                parsed_tool = json.loads(new_tool_json)

                # mcpServers 형식인지 확인하고 처리
                if "mcpServers" in parsed_tool:
                    # mcpServers 안의 내용을 최상위로 이동
                    parsed_tool = parsed_tool["mcpServers"]
                    st.info("'mcpServers' 형식이 감지되었습니다. 자동으로 변환합니다.")

                # 입력된 도구 수 확인
                if len(parsed_tool) == 0:
                    st.error("최소 하나 이상의 도구를 입력해주세요.")
                else:
                    # 모든 도구에 대해 처리
                    success_tools = []
                    for tool_name, tool_config in parsed_tool.items():
                        # URL 필드 확인 및 transport 설정
                        if "url" in tool_config:
                            # URL이 있는 경우 transport를 "sse"로 설정
                            tool_config["transport"] = "sse"
                            st.info(f"'{tool_name}' 도구에 URL이 감지되어 transport를 'sse'로 설정했습니다.")
                        elif "transport" not in tool_config:
                            # URL이 없고 transport도 없는 경우 기본값 "stdio" 설정
                            tool_config["transport"] = "stdio"

                        # 필수 필드 확인
                        if "command" not in tool_config and "url" not in tool_config:
                            st.error(f"'{tool_name}' 도구 설정에는 'command' 또는 'url' 필드가 필요합니다.")
                        elif "command" in tool_config and "args" not in tool_config:
                            st.error(f"'{tool_name}' 도구 설정에는 'args' 필드가 필요합니다.")
                        elif "command" in tool_config and not isinstance(tool_config["args"], list):
                            st.error(f"'{tool_name}' 도구의 'args' 필드는 반드시 배열([]) 형식이어야 합니다.")
                        else:
                            # pending_mcp_config에 도구 추가
                            st.session_state.pending_mcp_config[tool_name] = tool_config
                            success_tools.append(tool_name)

                    # 성공 메시지
                    if success_tools:
                        if len(success_tools) == 1:
                            st.success(f"{success_tools[0]} 도구가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")
                        else:
                            tool_names = ", ".join(success_tools)
                            st.success(f"총 {len(success_tools)}개 도구({tool_names})가 추가되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")
        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 에러: {e}")
            st.markdown(
                """
            **수정 방법**:
            1. JSON 형식이 올바른지 확인하세요.
            2. 모든 키는 큰따옴표(")로 감싸야 합니다.
            3. 문자열 값도 큰따옴표(")로 감싸야 합니다.
            4. 문자열 내에서 큰따옴표를 사용할 경우 이스케이프(\")해야 합니다.
            """
            )
        except Exception as e:
            st.error(f"오류 발생: {e}")

    # 구분선 추가
    st.divider()

    # 현재 설정된 도구 설정 표시 (읽기 전용)
    st.subheader("현재 도구 설정 (읽기 전용)")
    st.code(json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False))

# --- 등록된 도구 목록 표시 및 삭제 ---
with st.sidebar.expander("등록된 도구 목록", expanded=True):
    try:
        pending_config = st.session_state.pending_mcp_config
    except Exception as e:
        st.error("유효한 MCP 도구 설정이 아닙니다.")
    else:
        # pending config의 키(도구 이름) 목록을 표시
        for tool_name in list(pending_config.keys()):
            col1, col2 = st.columns([8, 2])
            col1.markdown(f"- **{tool_name}**")
            if col2.button("삭제", key=f"delete_{tool_name}"):
                # pending config에서 해당 도구 삭제 (즉시 적용되지는 않음)
                del st.session_state.pending_mcp_config[tool_name]
                st.success(f"{tool_name} 도구가 삭제되었습니다. 적용하려면 '적용하기' 버튼을 눌러주세요.")

with st.sidebar:
    # 적용하기 버튼: pending config를 실제 설정에 반영하고 세션 재초기화
    if st.button(
        "도구설정 적용하기",
        key="apply_button",
        type="primary",
        use_container_width=True,
    ):
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 변경사항을 적용하고 있습니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)
            # 설정 저장
            st.session_state.mcp_config_text = json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
            # 세션 초기화 준비
            st.session_state.session_initialized = False
            st.session_state.agent = None
            progress_bar.progress(30)
            # 초기화 실행
            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
            progress_bar.progress(100)
            if success:
                st.success("✅ 새로운 MCP 도구 설정이 적용되었습니다.")
            else:
                st.error("❌ 새로운 MCP 도구 설정 적용에 실패하였습니다.")
        # 페이지 새로고침
        st.rerun()

# --- 기본 세션 초기화 (초기화되지 않은 경우) ---
if not st.session_state.session_initialized:
    st.info("🔄 MCP 서버와 에이전트를 초기화합니다. 잠시만 기다려주세요...")
    success = st.session_state.event_loop.run_until_complete(initialize_session())
    if success:
        st.success(f"✅ 초기화 완료! {st.session_state.tool_count}개의 도구가 로드되었습니다.")
    else:
        st.error("❌ 초기화에 실패했습니다. 페이지를 새로고침해 주세요.")

# --- 대화 기록 출력 ---
print_message()

# --- 사용자 입력 및 처리 ---
user_query = st.chat_input("💬 질문을 입력하세요")
if user_query:
    query_text = user_query.strip()
    # 특정 질문 ("당신은 무엇을 할 수 있어?")에 대한 처리
    capability_queries = [
        "당신은 무엇을 할 수 있어?",
        "당신은 무엇을 할 수 있어요?",
        "당신은 무엇을 할 수 있나요?",
        "너는 무엇을 할 수 있어?",
        "너는 무엇을 할 수 있어요?",
        "너는 무엇을 할 수 있니?",
    ]
    if st.session_state.session_initialized and query_text in capability_queries:
        # 현재 사용 가능한 MCP 도구 정보에 기반하여 답변 생성
        if st.session_state.get("tool_count", 0) < 1:
            answer_msg = "현재 장착된 도구가 없습니다."
        else:
            # 도구 이름 목록 가져오기
            tool_names = []
            try:
                if st.session_state.mcp_client:
                    tool_list = st.session_state.mcp_client.get_tools()
                    tool_names = [getattr(t, "name", str(t)) for t in tool_list]
            except Exception:
                tool_names = list(st.session_state.pending_mcp_config.keys())
            if not tool_names:
                answer_msg = "현재 장착된 도구가 없습니다."
            elif len(tool_names) == 1:
                t = tool_names[0]
                # 도구 이름에 따른 간단한 기능 설명 추론
                desc = ""
                tl = t.lower()
                if "weather" in tl or "날씨" in tl:
                    desc = "날씨 정보를 조회"
                elif "github" in tl or "git" in tl:
                    desc = "GitHub 데이터를 검색"
                elif "wiki" in tl:
                    desc = "위키백과 정보를 검색"
                elif "wolfram" in tl or "계산" in tl:
                    desc = "복잡한 계산을 수행"
                elif "google" in tl or "검색" in tl:
                    desc = "웹 검색을 수행"
                elif "news" in tl or "뉴스" in tl:
                    desc = "뉴스를 검색"
                elif "image" in tl or "이미지" in tl or "vision" in tl:
                    desc = "이미지를 분석"
                elif "pdf" in tl:
                    desc = "PDF 문서를 분석"
                elif "translate" in tl or "번역" in tl:
                    desc = "텍스트를 번역"
                else:
                    desc = "해당 도구의 기능을 활용한 작업"
                answer_msg = f"저는 현재 '{t}' 도구를 사용할 수 있으며, 이를 통해 {desc}할 수 있습니다."
            else:
                # 둘 이상의 도구가 장착된 경우
                # 첫 두 개의 도구를 예로 들어 설명
                t1 = tool_names[0]
                t2 = tool_names[1]
                # 기능 설명 추론 함수
                def infer_desc(tool_name: str) -> str:
                    name = tool_name.lower()
                    if "weather" in name or "날씨" in name:
                        return "날씨 정보를 조회"
                    if "github" in name or "git" in name:
                        return "GitHub 데이터를 검색"
                    if "wiki" in name:
                        return "위키백과 정보를 검색"
                    if "wolfram" in name or "계산" in name:
                        return "복잡한 계산을 수행"
                    if "google" in name or "검색" in name:
                        return "웹 검색을 수행"
                    if "news" in name or "뉴스" in name:
                        return "뉴스를 검색"
                    if "image" in name or "이미지" in name or "vision" in name:
                        return "이미지를 분석"
                    if "pdf" in name:
                        return "PDF 문서를 분석"
                    if "translate" in name or "번역" in name:
                        return "텍스트를 번역"
                    return "다양한 작업을 수행"
                desc1 = infer_desc(t1)
                desc2 = infer_desc(t2)
                tool_list_str = ", ".join([f"'{tn}'" for tn in tool_names])
                answer_msg = f"저는 현재 {tool_list_str} 등의 도구를 사용할 수 있으며, 이를 통해 다양한 작업을 수행할 수 있습니다. 예를 들어, '{t1}' 도구로 {desc1}할 수 있고, '{t2}' 도구로 {desc2}할 수 있습니다."
                if len(tool_names) > 2:
                    answer_msg += " 이 외에도 다양한 작업을 수행할 수 있습니다."
        # 답변을 채팅 창에 표시 및 대화 기록에 추가
        st.chat_message("user").markdown(query_text)
        st.chat_message("assistant").markdown(answer_msg)
        st.session_state.history.append({"role": "user", "content": query_text})
        st.session_state.history.append({"role": "assistant", "content": answer_msg})
        st.rerun()
    elif st.session_state.session_initialized:
        # 일반적인 질문 처리
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
        st.warning("⏳ 시스템이 아직 초기화 중입니다. 잠시 후 다시 시도해주세요.")

# --- 사이드바: 시스템 정보 표시 및 모델 선택 ---
with st.sidebar:
    st.subheader("🔧 시스템 정보")
    st.write(f"🛠️ MCP 도구 수: {st.session_state.get('tool_count', '초기화 중...')}")
    # LLM 모델 선택 드롭다운
    model_options = ["Claude 3.7 Sonnet", "GPT-4 (OpenAI)", "Gemini 2.0", "Grok 3", "Llama 3.3", "Phoenix 1.0"]
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_options[0]
    selected_model = st.selectbox("🧠 LLM 모델 선택", model_options, index=model_options.index(st.session_state.selected_model))
    if selected_model != st.session_state.selected_model:
        # 모델 변경 시 세션 재초기화
        st.session_state.selected_model = selected_model
        st.session_state.session_initialized = False
        st.session_state.agent = None
        apply_status = st.empty()
        with apply_status.container():
            st.warning("🔄 선택한 LLM 모델로 에이전트를 재시작합니다. 잠시만 기다려주세요...")
            progress_bar = st.progress(0)
            # 현재 MCP 도구 설정 저장
            st.session_state.mcp_config_text = json.dumps(st.session_state.pending_mcp_config, indent=2, ensure_ascii=False)
            progress_bar.progress(30)
            success = st.session_state.event_loop.run_until_complete(initialize_session(st.session_state.pending_mcp_config))
            progress_bar.progress(100)
            if success:
                st.success(f"✅ 새로운 LLM 모델 '{selected_model}'로 에이전트가 초기화되었습니다.")
            else:
                st.error("❌ LLM 모델 변경 적용에 실패했습니다.")
        st.rerun()
    # 타임아웃 설정 슬라이더
    st.subheader("⏱️ 타임아웃 설정")
    st.session_state.timeout_seconds = st.slider(
        "응답 생성 제한 시간(초)",
        min_value=60,
        max_value=300,
        value=st.session_state.timeout_seconds,
        step=10,
        help="에이전트가 응답을 생성하는 최대 시간을 설정합니다. 복잡한 작업은 더 긴 시간이 필요할 수 있습니다.",
    )
    # 구분선 추가
    st.divider()
    # 사이드바 최하단에 대화 초기화 버튼 추가
    if st.button("🔄 대화 초기화", use_container_width=True, type="primary"):
        # thread_id 초기화
        st.session_state.thread_id = random_uuid()
        # 대화 히스토리 초기화
        st.session_state.history = []
        # 알림 메시지
        st.success("✅ 대화가 초기화되었습니다.")
        # 페이지 새로고침
        st.rerun()
```