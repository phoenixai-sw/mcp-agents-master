from mcp.server.fastmcp import FastMCP

# MCP 서버 초기화: AWS EC2의 public IP에서 모든 IP의 접속 허용, 포트 8005 사용
mcp = FastMCP(
    "Weather",  # MCP 서버 이름
    instructions="You are a weather assistant that can answer questions about the weather in a given location.",
    host="0.0.0.0",  # 모든 IP로부터의 연결 허용
    port=8005,       # 포트 번호 8005 사용
)

@mcp.tool()
async def get_weather(location: str) -> str:
    """
    지정된 위치의 날씨 정보를 반환합니다.
    실제 날씨 API 호출 대신 고정된 응답을 반환합니다.
    """
    return f"It's always Sunny in {location}"

if __name__ == "__main__":
    print("mcp remote server is running...")
    mcp.run(transport="sse")
