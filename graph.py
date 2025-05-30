import os
import re
from dotenv import load_dotenv, find_dotenv

from langchain_ollama import OllamaEmbeddings
from langchain_ollama.chat_models import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, trim_messages
from langchain_core.documents import Document

from langgraph.store.memory import InMemoryStore
from langgraph.store.memory import Item
from langgraph.store.base import BaseStore
from langgraph.graph.message import add_messages
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver 

from typing import Sequence, Literal, List
from typing_extensions import Annotated, TypedDict

import json
from json import JSONDecodeError
from pydantic import BaseModel, Field, ValidationError

from uuid import uuid4

# Load configurations from .env file
load_dotenv()

OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL')

user_id = "default"
namespace = (user_id,) # like tags on data

# Initialize embedding model
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model="nomic-embed-text",
)

# Initialize vector store with embeddings
vector_store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 768,
    }
)

# Initialize LLM for chatbot
llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model="llama3.2:1b-instruct-fp16",
    temperature=0.8,
)

# Initialize LLM for routing and memory operations
llm_json_mode = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    # model="deepseek-r1:1.5b-qwen-distill-fp16",
    model="llama3.2:1b-instruct-fp16",
    temperature=0,
    format="json",
)

# Define agent state
class ChatState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    last_msg: HumanMessage
    memories: List[Item]

class ChatStateIO(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

chatbot_sys_prompt = """
You are a helpful assistant. 
You have persistent memory, the relevant memories to the user query are listed below.

Relevant Memories:
<memories>
{memories}
</memories>
"""
chatbot_prompt = ChatPromptTemplate(
    [
        SystemMessagePromptTemplate.from_template(chatbot_sys_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define trimmer
trimmer = trim_messages(
    max_tokens=8192,
    strategy="last",
    token_counter=llm,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

async def chatbot(state: ChatState, store: BaseStore):
    trimmed_messages = trimmer.invoke(state["messages"])
    fmt_memories = ""
    
    # Save last message
    last_msg = state["messages"][-1]

    # Use most similar memories for context
    memories = await store.asearch(
        namespace,
        # query=str([m.content for m in state['messages'][-3:]]),
        query=str(last_msg.content),
        limit=10,
    )

    fmt_memories = "\n".join(f"[{mem.key}]: {mem.value} (similarity: {mem.score})\tLast updated: {mem.updated_at}" for mem in memories)

    print(fmt_memories)

    prompt = chatbot_prompt.invoke(
        {
            "messages": trimmed_messages,
            "memories": fmt_memories
        }
    )
    response = await llm.ainvoke(prompt)
    
    return {"messages": [response], "last_msg": last_msg, "memories": memories}


route_message_sys_prompt = """
You are an intelligent assistant tasked with determining whether a user's message contains information that should be stored as a memory for future reference in a Retrieval-Augmented Generation (RAG) system or discarded and provide a rationale of why. Follow these steps:

1. **Relevance Check**: Does the message include specific, factual, or contextual information that could be useful for answering related queries in the future? Avoid storing generic conversational phrases or greetings (e.g., "Hello," "How are you?"). Examples include:
   - Personal preferences (e.g., "I like Italian food.")
   - Personal details (e.g., "I have a brother named Jerry.")
   - Specific facts or events (e.g., "I visited Paris last summer.")
   - Unique identifiers or preferences (e.g., "My favorite color is blue.")
2. **Uniqueness Check**: Is this information new and not already stored in the existing memories below? Avoid duplicating existing data.
3. **Utility Check**: Could this information improve the accuracy or personalization of future responses? Consider whether it provides actionable insights or context.

**Existing memories**
<memories>
{memories}
</memories>

### Output Format
Return JSON object in the format {{"route": "", "rationale": ""}}.
- Set "route" to "store" if the message meets all criteria and should be added to memory.
- Otherwise, set "route" to "discard" if the message does not meet the criteria or is irrelevant.
"""

route_message_prompt = ChatPromptTemplate.from_messages(
    [
        route_message_sys_prompt, 
        MessagesPlaceholder(variable_name="message")
    ]
)

class MessageRouterOutput(BaseModel):
    route: Literal['store', 'discard']

async def route_message(state: ChatState, store: BaseStore):
    fmt_memories = "\n".join(f"[key: {mem.key}]: {mem.value} (similarity: {mem.score})\tLast updated: {mem.updated_at}" for mem in state['memories'])
    prompt = route_message_prompt.invoke(
        {
            'memories': fmt_memories, 
            'message': [state['last_msg']]
        }
    )
    response = await llm_json_mode.ainvoke(prompt)
    try:
        content = re.sub(r"(?s)<think>.*?</think>", "", response.content)
        json_data = json.loads(content)
        MessageRouterOutput(**json_data)
        if json_data['route'] == 'store':
            wasMemUpdated = []
            for mem in state['memories']:
                if mem.score > 0.6:
                    memory_content = mem.value['content']
                    result = await check_memory(memory_content, state['last_msg'].content)
                    result = True if result == "yes" else False
                    if result:
                        await update_memory(state, store, mem.key, memory_content)
                    wasMemUpdated.append(result)
            if not any(wasMemUpdated):
                return 'insert'
        return END
    except (JSONDecodeError, ValidationError):
        return END
    return END

check_memory_sys_prompt = """
Determine if the following user message and the vector database record below contain conflicting information and provide a rationale of why. Conflicting information arises when the two statements cannot logically coexist or describe incompatible facts, events, or relationships.

Follow the instructions, read carefully:
1. Check for Contradictions: 
    - Analyze whether the two statements assert mutually exclusive facts. For example:
        - Statement 1: "I have a brother named Jerry."
        - Statement 2: "My brother Jerry changed his name to Jerrio."
        - These statements conflict because they describe incompatible states of Jerry's name.
2. Evaluate Coreference:
    - Determine if the statements refer to the same subject or event. If they do not share a common subject or context, they are unlikely to conflict. For example:
        - Statement 1: "I like apples."
        - Statement 2: "I have a brother."
        - These statements do not conflict because they address unrelated topics.
3. Identify Semantic Contradictions:
    - Look for linguistic inconsistencies (e.g., opposing meanings or incompatible descriptions). For example:
        - Statement 1: "The sky is blue."
        - Statement 2: "The sky is green."
        - These statements conflict because the sky cannot simultaneously be blue and green.
4. Consider Negation and Antonymy:
    - Check for negation or opposing terms that might indicate a contradiction. For example:
        - Statement 1: "The bombers entered the embassy."
        - Statement 2: "The bombers had not managed to enter the embassy."
        - These statements conflict due to negation.
5. Examine Structural Features:
    - Compare subjects and objects in each statement to detect mismatches in relationships or roles. For example:
        - Statement 1: "Jacques Santer succeeded Jacques Delors as president of the European Commission."
        - Statement 2: "Delors succeeded Santer in the presidency of the European Commission."
        - These statements conflict due to reversed subject-object relationships.
6. Numeric and Temporal Mismatches:
    - Check for inconsistencies in numbers, dates, or times. For example:
        - Statement 1: "More than 2,000 people died in the Johnstown Flood."
        - Statement 2: "100 people died in a ferry sinking."
        - These do not conflict because they describe different events.

**Vector Database Record**:
{record}

### Output Format
Return JSON object in the format {{"conflict": "", "rationale": ""}}.
- Set "conflict" to "yes" if the human message contains conflicting information.
- Otherwise, set "conflict" to "no" as the output.
"""

check_memory_prompt = ChatPromptTemplate.from_messages(
    [
        check_memory_sys_prompt, 
        MessagesPlaceholder(variable_name="message")
    ]
)

class CheckMemory(BaseModel):
    conflict: Literal["yes", "no"]
    rationale: str

async def check_memory(existing_mem: str, new_mem: str):
    prompt = check_memory_prompt.invoke(
        {
            'record': existing_mem,
            'message': [HumanMessage(content=new_mem)]
        }
    )
    response = await llm_json_mode.ainvoke(prompt)
    try:
        content = re.sub(r"(?s)<think>.*?</think>", "", response.content)
        content = str(content).lower()
        json_data = json.loads(content)
        CheckMemory(**json_data)
        return json_data['conflict']
    except (JSONDecodeError, ValidationError):
        return False
    return False

update_memory_sys_prompt = """
Act as a natural language memory editing specialist. Update the existing record using the new user message while preserving natural language phrasing using these strict rules:

**Existing record**:
{record}

1. **Update Logic**:
   - Merge new information into existing phrasing
   - Identify overlapping elements: entities (names/dates), actions, purposes
   - Preserve original phrasing where possible
2. **Output**: Return JSON object in the format {{"content": ""}}
"""

update_memory_prompt = ChatPromptTemplate.from_messages(
    [
        update_memory_sys_prompt, 
        MessagesPlaceholder(variable_name="message")
    ]
)

class Memory(BaseModel):
    content: str = Field(max_length=200)

async def update_memory(state: ChatState, store: BaseStore, memory_id: str, memory_content: str):
    prompt = update_memory_prompt.invoke(
        {
            'record': memory_content,
            'message': [state['last_msg']]
        }
    )
    response = await llm_json_mode.ainvoke(prompt)
    try:
        content = re.sub(r"(?s)<think>.*?</think>", "", response.content)
        json_data = json.loads(content)
        Memory(**json_data)
        memory_content = json_data['content']
    except (JSONDecodeError, ValidationError):
        return

    mem_id = uuid4()
    await store.aput(
        namespace,
        key=memory_id,
        value={'content': memory_content},
    )
    return

insert_memory_sys_prompt = """
Act as an information distillation expert. Convert this user message into a compact text entry using these rules:

1. **Key Elements**:
   - Identify: Subject, Action, Context, Purpose
   - Preserve: Numbers/dates, relationships, intent
   - Remove: Greetings, filler words, duplicates
2. **Format Requirements**:
   - Single "content" field in JSON
   - Use natural language phrasing
   - Maximum 200 character
3. **Output**: Return JSON object in the format {{"content": ""}}
"""

insert_memory_prompt = ChatPromptTemplate.from_messages(
    [
        insert_memory_sys_prompt, 
        MessagesPlaceholder(variable_name="message")
    ]
)

async def insert_memory(state: ChatState, store: BaseStore):
    # Create memory
    prompt = insert_memory_prompt.invoke(
        {
            'message': [state['last_msg']]
        }
    )
    response = await llm_json_mode.ainvoke(prompt)
    try:
        content = re.sub(r"(?s)<think>.*?</think>", "", response.content)
        json_data = json.loads(content)
        Memory(**json_data)
        memory_content = json_data['content']
    except (JSONDecodeError, ValidationError):
        memory_content = state['last_msg'].content

    mem_id = uuid4()
    await store.aput(
        namespace,
        key=str(mem_id),
        value={'content': memory_content},
    )
    return {}

# Define a new graph
builder = StateGraph(state_schema=ChatState, input=ChatStateIO, output=ChatStateIO)

# Define the nodes and edges in the graph
builder.add_node("chatbot", chatbot)
builder.add_node("insert", insert_memory)

builder.add_edge(START, "chatbot")
builder.add_conditional_edges("chatbot", route_message, ["insert", END])

# Add memory
memory = InMemorySaver()
graph = builder.compile(store=vector_store, checkpointer=memory)