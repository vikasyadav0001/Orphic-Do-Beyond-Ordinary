"""
1. Treating the Upload as a "System Message Event"
When a file is uploaded, you emit a System Notification Message into the chat thread's history.

Instead of writing custom state-syncing code, you write a notification message directly into the database or checkpointer messages list. For example:

SystemMessage(content="[Event: File Uploaded] Name: backend.pdf | Path: /home/dell/Desktop/Orphic-Do-Beyond-Ordinary/document_parser/backend.pdf")

Why this works:
No State Syncing: The message is saved directly in the thread's message history by LangGraph's checkpointer.
Immediate Agent Awareness: When the main agent runs on any subsequent turn, it reads the conversation history, sees the SystemMessage declaring that backend.pdf is available at that specific file path, and knows it has access to it.
Precise Tool Calling: If the user asks, "What is in backend.pdf?", the agent reads the system message, extracts the path (/home/dell/.../backend.pdf), and passes it directly to your search_uploaded_documents tool.

2. Wiring the API / Controller Layer (The Event Coordinator)

The API endpoint acts as the coordinator between the Document Pipeline Graph and the Main Agent Graph. Here is how it handles incoming requests:

Case A: The User uploads a file WITHOUT a query
Run Ingestion: The API runs the Document Ingest Graph:
It parses the document and indexes it.
It generates the opening_offer.
Commit Event: The API appends the SystemMessage notifying the agent about the new file to the thread history.
Respond: The API returns the opening_offer to the user and finishes.

Case B: The User uploads a file WITH a query (e.g. "What are the highlights of backend.pdf?")
Run Ingestion: The API runs the Document Ingest Graph to index the file.
Commit Event: The API appends the SystemMessage notifying the agent about the file to the thread history.
User Message: The API appends the user's query (HumanMessage) to the thread history.
Hand-off to Agent: The API invokes the Main Agent's astream_events stream.
Agent Search: The Main Agent starts. It reads the chat history, sees the file upload notification, executes the search tool using the specified file path, and streams the answer back to the user.
"""