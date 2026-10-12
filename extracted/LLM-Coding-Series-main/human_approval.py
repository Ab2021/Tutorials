from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool

load_dotenv()

# --------------------------------
# LLM
# --------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# --------------------------------
# Tool
# --------------------------------

@tool
def send_email(message: str) -> str:
    """Send an email message."""

    # Demo only — no real email API is called
    return f"Email sent successfully: {message}"


# --------------------------------
# Agent creates proposed action
# --------------------------------

def create_action(user_query):

    prompt = f"""
You are an AI agent.

The user has asked:

{user_query}

Decide what action should be performed.

If an email should be sent, return only the email message that should be sent.

Do not actually execute the action.
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------
# Human approval
# --------------------------------

def get_human_approval(action):

    print("\n========== PROPOSED ACTION ==========")
    print(action)

    approval = input(
        "\nDo you approve this action? (yes/no): "
    )

    return approval.lower() == "yes"


# --------------------------------
# Execute action
# --------------------------------

def execute_action(action):

    result = send_email.invoke({
        "message": action
    })

    return result


# --------------------------------
# Final response
# --------------------------------

def create_final_response(user_query, result):

    prompt = f"""
The user asked:

{user_query}

The action result is:

{result}

Give the user a short and clear final response.
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------
# Main
# --------------------------------

user_query = input("\nAsk the agent: ")


# Step 1: Agent creates an action
action = create_action(user_query)


# Step 2: Human reviews action
approved = get_human_approval(action)


# Step 3: Execute only if approved
if approved:

    result = execute_action(action)

    print("\nTool Result:")
    print(result)

    final_response = create_final_response(
        user_query,
        result
    )

    print("\n========== FINAL ANSWER ==========")
    print(final_response)

else:

    print("\nAction rejected by human.")
    print("No tool was executed.")