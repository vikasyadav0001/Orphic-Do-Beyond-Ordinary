import asyncio
import sys
from orchestrators.graph import stream_response
from memory.memory_extractor import graph as extractor_graph
from utils.logger import get_logger

logger = get_logger(__name__)

async def main():
    thread_id = "test-thread-3"
    user_id = "default_user"

    print("=" * 50)
    print("  Orphic Agent - Interactive Mode")
    print("  Type 'exit' to quit")
    print("=" * 50)

    while True:
        try:
            user_input = input("\nYOU: ").strip()

            if not user_input:
                continue

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            print("\nAI: ", end="", flush=True)

            #stream the response
            full_response = ""
            try:
                async for token in stream_response(user_input, thread_id):
                    print(token, end="", flush=True)
                    full_response += token
                print()
            except Exception as e:
                logger.error(f"Error during streaming: {e}")
                print(f"\n[Error: {e}]")
                continue

            # Extract memories
            try:
                conversation = f"User: {user_input}\nAssistant: {full_response}"
                await extractor_graph.ainvoke(
                    {"messages": [{"role": "user", "content": conversation}]},
                    config={"configurable": {"thread_id": thread_id, "user_id": user_id}}
                )
            except Exception as e:
                logger.error(f"Memory extraction failed: {e}")
                # Don't crash - memory extraction is secondary

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nEOF received. Exiting.")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}")
            print(f"\n[Unexpected error: {e}]")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
