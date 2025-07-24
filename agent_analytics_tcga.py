#pip install langchain_experimental, tabulate

from langchain_core.prompts.chat import ChatPromptTemplate
#from langchain_ollama.llms import OllamaLLM #Don't use this
from langchain_ollama import ChatOllama
from langchain_experimental.tools import PythonAstREPLTool
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser

import pandas as pd

llm = ChatOllama(model="llama3.2") #model must support tool calling. deepseek-r1 doesn't work

df = pd.read_csv("C:\\Users\\user\\Documents\\tcga_metadata.csv")

# Tool 

tool = PythonAstREPLTool(locals={"df": df})
llm_with_tools = llm.bind_tools([tool], tool_choice=tool.name)

tool_response = llm_with_tools.invoke(
    "I have a dataframe 'df' and want to know the correlation between the 'age_at_diagnosis' and 'days_to_death' columns"
)

parser = JsonOutputKeyToolsParser(key_name=tool.name, first_tool_only=True)

system = f"""You have access to a pandas dataframe `df`. \
Given a user question, write the Python code to answer it. \
Return ONLY the valid Python code and nothing else. \
Don't assume you have access to any libraries other than built-in Python ones and pandas."""

prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{question}")])

chain = prompt | llm_with_tools | parser | tool
chain_response = chain.invoke({"question": "What's the correlation between 'age_at_diagnosis' and 'days_to_death'"})

chain_response

# Prebuilt agent
# Work perfectly with GPT-4o but llama3.2 returns only meaningless response: 'output': '{"name": "corr", "parameters": {"x": "df[\'age_at_diagnosis\']", "y": "df[\'days_to_death\']"}}'

from langchain_experimental.agents import create_pandas_dataframe_agent

agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    agent_type="tool-calling",
    verbose=True,
    allow_dangerous_code=True
)

agent_response = agent_executor.invoke(
    {
        "input": "What's the correlation between 'age_at_diagnosis' and 'days_to_death' columns in 'df'?"
    }
)

agent_response
