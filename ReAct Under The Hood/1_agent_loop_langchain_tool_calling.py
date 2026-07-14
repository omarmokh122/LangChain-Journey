from dotenv import load_dotenv  # Load key=value pairs from a .env file into process environment variables

load_dotenv()  # Actually read .env so API keys / config are available to the process

from langchain.chat_models import init_chat_model  # Helper that builds a chat model from a provider:model string
from langchain_core.tools import tool  # Decorator that turns a Python function into a LangChain tool the LLM can call
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage  # Message types for the chat history
from langsmith import traceable  # Decorator that traces this function in LangSmith for observability

MAX_ITERATIONS = 10  # Safety cap: stop the agent loop after this many LLM→tool rounds
MODEL = "qwen3:1.7b"  # Local Ollama model name to use for the chat LLM

# Tools - LangChain Tools Decorator
# Each @tool function becomes something the model can request by name + JSON args

@tool  # Register get_product_price as a LangChain tool (name/schema come from the function + docstring)
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catalog"""  # Docstring is shown to the model as the tool description
    print(f"  >> Executing get_product_price(product='{product_name}')")  # Debug log when the tool runs
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}  # Fake in-memory product catalog
    return prices.get(product_name, 0)  # Return price if known, else 0

@tool  # Register apply_discount as a second tool the model can call
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold"""  # Tells the model which tiers are valid and what the tool returns
    print(f"  >> Executing apply_discount(price='{price}', discount_tier='{discount_tier}')")  # Debug log of the call args
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}  # Map tier name → percent off
    discount = discount_percentages.get(discount_tier, 0)  # Look up tier; unknown tiers get 0% off
    return round(price * (1 - discount / 100), 2)  # Compute discounted price, rounded to 2 decimals


# --- Agent's Loop ---
@traceable(name="LangChain Agent Loop")  # Wrap the agent in a LangSmith span named "LangChain Agent Loop"
def run_agent(question: str):
    tools = [get_product_price, apply_discount]  # List of tools we expose to the model
    tools_dict = {t.name: t for t in tools}  # Map tool name → tool object so we can look up by name at runtime

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)  # Create the chat model via Ollama; temp=0 for more deterministic replies
    llm_with_tools = llm.bind_tools(tools)  # Attach tool schemas so the model can emit structured tool_calls

    print(f"Question: {question}")  # Show the user question at the start of a run
    print("=" * 60)  # Visual separator in the console

    messages = [  # Conversation history we keep growing each iteration
        SystemMessage(  # System prompt defining role + hard rules for tool use
            content=(
                "You are a helpful shopping assistant. "  # Role: shopping helper
                "you have access to a product catalog tool. "  # Mentions get_product_price
                " and a discount tool. \n\n"  # Mentions apply_discount
                "STRICT RULES - you must follow these exactl:\n"  # Begin mandatory tool-usage rules
                "1. NEVER guess or assume any product price. "  # Force real lookup before answering about prices
                "you mUST call get_product_price first to get the real price.\n"  # Step 1: price tool first
                "2. Only call apply_discount AFTER you have received "  # Step 2: discount only after a real price
                "a price from get_product_price. Pass the exact price "  # Do not invent a price for discounting
                "returned by get_product_price - do NOT pass a made-up number.\n"
            )
        ),
        HumanMessage(content=question),  # First user turn: the shopping question to solve
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):  # ReAct-style loop: think → (optionally) act → observe → repeat
        ai_message = llm_with_tools.invoke(messages)  # Ask the LLM; may return text and/or tool_calls
        tool_calls = ai_message.tool_calls  # List of tools the model wants to run this turn (may be empty)

        # If no tool calls, this is the final answer
        if not tool_calls:  # Model chose to answer in natural language instead of calling a tool
            print(f"\nFinal Answer: {ai_message.content}")  # Print the text answer
            return ai_message.content  # Exit successfully with the final response

        # Process only first tool call - force one tool per iteration
        tool_call = tool_calls[0]  # Ignore any extra tool_calls; only run the first one
        tool_name = tool_call.get("name")  # Which tool the model selected (e.g. "get_product_price")
        tool_args = tool_call.get("args", {})  # JSON args the model provided for that tool
        tool_call_id = tool_call.get("id")  # ID required so ToolMessage can be paired with this call

        print(f"[Tools Selected] {tool_name} with args: {tool_args}")  # Log the chosen tool + args
        tool_to_use = tools_dict.get(tool_name)  # Resolve the Python tool object by name
        if tool_to_use is None:  # Model asked for a tool we did not register
            raise ValueError(f"Tool '{tool_name}' not found")  # Fail fast rather than silently continuing

        observation = tool_to_use.invoke(tool_args)  # Actually run the tool and get the return value
        print(f"[Tool Result] {observation}")  # Log the observation for debugging

        messages.append(ai_message)  # Keep the AI's tool-call message in history so the next turn stays coherent
        messages.append(  # Feed the tool result back as a ToolMessage linked by tool_call_id
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )  # Next loop iteration will see both the call and the result

    print("Error: Max Iterations reached without a final answer")  # Loop exhausted without a text-only reply
    return None  # Signal failure: no final answer within the iteration budget


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")