# cognition/reasoning/
# Reasoning strategies for individual agents.
# Responsibilities:
#   - chain_of_thought.py  → Step-by-step reasoning before answering (CoT prompting)
#   - react.py             → Reasoning + Acting loop (Thought → Action → Observation)
#   - reflection.py        → Agent critiques its own output and self-corrects
#   - tree_of_thought.py   → Explores multiple reasoning branches before deciding
# Import these strategies into any agent that needs structured thinking.
