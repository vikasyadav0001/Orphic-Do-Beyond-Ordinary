"""
clear_checkpoints.py
====================
Deletes stale LangGraph checkpoints from NeonDB.

LangGraph stores checkpoint data across THREE tables:
  - checkpoints        → main checkpoint chain
  - checkpoint_blobs   → serialised message blobs
  - checkpoint_writes  → pending tool-call writes (the main culprit for the 400 error)

You MUST delete from all three to fully clear a thread.

Usage:
  # Delete specific thread IDs:
  python scripts/clear_checkpoints.py test-thread-1 test-thread-3

  # Delete ALL threads (nuclear option — wipes entire checkpoint history):
  python scripts/clear_checkpoints.py --all
"""

import asyncio
import sys
import os
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

load_dotenv()

DB_URI = os.getenv("DATABASE_URL")
if not DB_URI:
    print("❌ DATABASE_URL not found in .env — aborting.")
    sys.exit(1)


async def clear_threads(thread_ids: list[str] | None):
    """
    Deletes checkpoint data for the given thread_ids.
    If thread_ids is None, deletes ALL threads.
    """
    pool = AsyncConnectionPool(
        conninfo=DB_URI,
        min_size=1,
        max_size=2,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await pool.open()

    async with pool.connection() as conn:
        if thread_ids is None:
            # ── Nuclear: wipe everything ──────────────────────────────────
            print("⚠️  Deleting ALL checkpoints from all threads...")
            await conn.execute("DELETE FROM checkpoint_writes;")
            await conn.execute("DELETE FROM checkpoint_blobs;")
            await conn.execute("DELETE FROM checkpoints;")
            print("✅ All checkpoints cleared.")
        else:
            # ── Targeted: only the specified thread IDs ───────────────────
            for tid in thread_ids:
                print(f"🧹 Clearing thread: '{tid}'")
                await conn.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s;", (tid,)
                )
                await conn.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s;", (tid,)
                )
                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s;", (tid,)
                )
                print(f"   ✅ Done — '{tid}' is now clean.")

    await pool.close()
    print("\n🎉 Checkpoint cleanup complete. Restart your agent to begin fresh.")


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python scripts/clear_checkpoints.py test-thread-1 test-thread-3")
        print("  python scripts/clear_checkpoints.py --all")
        sys.exit(0)

    if "--all" in args:
        confirm = input("⚠️  This will delete ALL conversation history. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)
        thread_ids_to_clear = None
    else:
        thread_ids_to_clear = args

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(clear_threads(thread_ids_to_clear))
