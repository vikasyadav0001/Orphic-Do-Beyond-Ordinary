from protocols.mcp.remote_mcp_client_config import get_mcp_tools
from dotenv import load_dotenv
import asyncio
from utils.logger import get_logger


logger = get_logger(__name__)
logger.info("Connecting to MCP servers...")
async def get_tools():
    try:
        tools = await get_mcp_tools() 
        print(f"✅ name: {len(tools)} tools loaded")
        for t in tools:
            print(f"   - {t.name}")
        logger.info("Connected to MCP servers...")
    except Exception as e:
            import traceback
            traceback.print_exc()
            if hasattr(e, 'exceptions'):
                for idx, sub_e in enumerate(e.exceptions):
                    print(f"🔴 Sub-error {idx+1}: {type(sub_e).__name__}: {sub_e}")
            else:
                print(f"🔴 Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(get_tools())