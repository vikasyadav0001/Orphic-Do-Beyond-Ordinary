from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from memory.graph_checkpointer import pool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()


#initialise the Chatbot system State
class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage],add_messages] # using a reducer fuctoin in the state to keep the messages in the list is we do not use it the system will keep replacing the messages in the state

#iniitialize the llm model
llm = ChatOpenAI(model='gpt-5.2', streaming=True)

#make the graph node currently there is only one node
@traceable(name="chat_node")
async def chat_node(state: ChatState):

    messages = state['messages']

    response = await llm.ainvoke(messages)  # use async invoke so it doesn't block the event loop

    return {'messages': [response]}  # return the message response in the list because the message state is a list


#initialze the graph with the chatbot statte
graph = StateGraph(ChatState)

#add the one and only node
graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)


checkpointer = AsyncPostgresSaver(pool)
bot = graph.compile(checkpointer=checkpointer)

@traceable(name="stream_response")
async def stream_response(user_message: str, thread_id: str):
    """
    Async generator that yields individual tokens from LangGraph streaming
    """

    config = {'configurable' : {"thread_id" : thread_id}}

    async for event in bot.astream_events(
        {"messages" : [HumanMessage(content=user_message)]},
        config=config,
        version='v2'
    ):
        if event['event'] == 'on_chat_model_stream':
            chunk = event['data'].get('chunk')
            if chunk and hasattr(chunk, 'content') and chunk.content:
                yield chunk.content