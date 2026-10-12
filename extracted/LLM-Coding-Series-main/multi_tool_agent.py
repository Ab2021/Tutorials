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
# Tools
# --------------------------------

@tool
def get_population(country: str) -> str:
    """Get population of a country."""
    
    population = {
        "india": 1420000000,
        "usa": 340000000,
        "uk": 68000000
    }

    return str(
        population.get(
            country.lower(),
            "Population not available"
        )
    )


@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    
    try:
        return str(eval(expression))
    except Exception:
        return "Invalid expression"


tools = {
    "get_population": get_population,
    "calculator": calculator
}


# --------------------------------
# Planner
# --------------------------------

def create_plan(user_query):

    prompt = f"""
You are a planning agent.

Break the user's request into executable steps.

Available tools:

1. get_population(country)
2. calculator(expression)

User request:
{user_query}

Return the plan in this format:

STEP 1:
TOOL: tool_name
INPUT: tool_input

STEP 2:
TOOL: tool_name
INPUT: tool_input
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------
# Executor
# --------------------------------

def execute_plan(plan):

    results = []

    steps = plan.split("STEP ")[1:]

    for step in steps:

        print("\nExecuting:")
        print(step)

        lines = step.strip().split("\n")

        tool_name = None
        tool_input = None

        for line in lines:

            if line.startswith("TOOL:"):
                tool_name = line.replace(
                    "TOOL:", ""
                ).strip()

            elif line.startswith("INPUT:"):
                tool_input = line.replace(
                    "INPUT:", ""
                ).strip()

        if tool_name in tools:

            if tool_name == "get_population":

                result = tools[tool_name].invoke({
                    "country": tool_input
                })

            elif tool_name == "calculator":

                result = tools[tool_name].invoke({
                    "expression": tool_input
                })

            print("Tool Result:", result)

            results.append(result)

    return results


# --------------------------------
# Final Answer
# --------------------------------

def create_final_answer(user_query, results):

    prompt = f"""
Answer the user's question using the tool results.

User question:
{user_query}

Tool results:
{results}

Give a clear final answer.
"""

    response = llm.invoke(prompt)

    return response.content


# --------------------------------
# Main
# --------------------------------

user_query = input("\nAsk a question: ")


# 1. Planner creates the plan
plan = create_plan(user_query)

print("\n========== PLAN ==========")
print(plan)


# 2. Executor executes the plan
results = execute_plan(plan)


# 3. LLM creates final answer
final_answer = create_final_answer(
    user_query,
    results
)

print("\n========== FINAL ANSWER ==========")
print(final_answer)