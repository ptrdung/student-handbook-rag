from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# System prompt defining the AI's role and behavior
SYSTEM_PROMPT = """You are an AI virtual assistant specializing in supporting students based on the Student Handbook.
Your task is to answer student questions accurately, concisely, and helpfully based on the provided context.

Required rules:
1. ALWAYS use information from the context to answer whenever possible.
2. If the context is empty or does not contain information to answer the question, you MUST begin your answer with: "I'm sorry, I couldn't find this information in the Student Handbook." Only then are you allowed to offer advice or information based on your general knowledge, but you must clearly note that it is external reference information.
3. Answers must be formatted using Markdown for easy readability (bold, italics, lists, and tables are supported if needed).
4. Communication language: Vietnamese, friendly and professional.
"""

# Prompt for the human message which includes context and the query
HUMAN_PROMPT = """Student Handbook context:
{context_str}

Student question: {query}
"""

def get_generation_prompt() -> ChatPromptTemplate:
    """
    Returns the ChatPromptTemplate for generation including chat history.
    """
    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
    ])
