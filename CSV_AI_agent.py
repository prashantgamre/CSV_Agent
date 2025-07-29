from langchain.schema import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

inventory = pd.read_csv("inventory_data.csv")
# suppliers = pd.read_csv("Suppliers.csv")
# products = pd.read_csv("Products.csv")
# sales = pd.read_csv("Sales.csv")

load_dotenv()
st.Secrets("ANTHROPIC_API_KEY")
model = ChatAnthropic(api_key=st.secrets("ANTHROPIC_API_KEY"),model="claude-3-5-sonnet-20241022", temperature=0.5)

    

# Initialize the agent with error handling
agent = create_pandas_dataframe_agent(
    model, 
    df=[inventory], 
    verbose=True, 
    allow_dangerous_code=True,
    handle_parsing_errors=True,  # Add error handling
    max_iterations=10,  # Prevent infinite loops
    return_intermediate_steps=True  # For better debugging
)

# res = agent.invoke("total solf count of Product-001?")


CSV_PROMPT_PREFIX = """
First set the pandas display options to show all rows and columns,
get the column names, then answer the question.

Product.csv has : product details 
Inventory.csv has : inventory details such as how many product has in currently in stock
Sales.csv has : sales details such as how many product has been sold and at which date and time
Suppliers.csv has : supplier details

"""


CSV_PROMPT_SUFFIX = """
You are a helpful assistant that can answer questions about a CSV file.
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result, reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE**
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n"
In the explanation, mention the column names that you used to get
to the final answer.
"""
# Question = "which product has the highest sales?"

# res = agent.invoke(CSV_PROMPT_PREFIX + Question + CSV_PROMPT_SUFFIX)

st.title("CSV AI Agent")

st.write("### Dataset preview")

# Initialize session state for the button if it doesn't exist
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False

def set_button_clicked():
    st.session_state.button_clicked = True

st.write("### Question")
question = st.text_input("Question:", "which product has the highest sales?")

# Button with a unique key
if st.button("Ask", key="ask_button", on_click=set_button_clicked) or st.session_state.button_clicked:
    if question:  # Only process if there's a question
        with st.spinner('Analyzing...'):
            res = agent.invoke(CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX)
            st.write("### Answer")
            st.markdown(res["output"])
            st.session_state.button_clicked = False  # Reset the button state
    else:
        st.warning("Please enter a question.")

