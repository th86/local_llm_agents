# local_llm_agents

Set up locally hosted LLM agent stack. No subscription and no data leak.

tabular data analytics agent example `agent_analytics_tcga.py`

## Windows

Install python from App Store

Install [Ollama](https://ollama.com/) and pull models in CMD

`ollama pull gemma3:4b` or `deepseek-r1`

```
pip install ollama
pip install -qU langchain_ollama
```

Should use `uv` to manage the environment

## Python

### Minimal example
From [Ollama Documents](https://github.com/ollama/ollama-python)
```
from ollama import chat
resp = chat(model='deepseek-r1', messages=[  {'role': 'user','content': 'Why is the sky blue? keep the answer short'},])`
```

### LangChain example
From [LangChain Documents](https://python.langchain.com/docs/integrations/llms/ollama/)
```
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

template = """Question: {question}
Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="deepseek-r1")

chain = prompt | model
chain.invoke({"question": "What is LangChain?"})
```

### LangChain vision example
From [LangChain Documents](https://python.langchain.com/docs/integrations/llms/ollama/)
require `pip install pillow`
```
import base64
from io import BytesIO
from PIL import Image

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


file_path = "C:\\Users\\user\\Documents\\Smiley_Face.jpg"
pil_image = Image.open(file_path)
image_b64 = convert_to_base64(pil_image)

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma3:4b")
llm_with_image_context = llm.bind(images=[image_b64])
llm_with_image_context.invoke("What is in the picture?")
```
Note: Ollama CLI use

```
C:\Users\ouyangt>ollama run gemma3:4b
>>> what is in the picture C:\Users\user\Documents\Smiley_Face.jpg
```

## TODO

* [Browser Use local setup](https://docs.browser-use.com/development/local-setup)

* [FastMCP](https://gofastmcp.com/)

* [LangChain Document Loader PDF](https://python.langchain.com/docs/how_to/document_loader_pdf/), [Word](https://python.langchain.com/docs/integrations/document_loaders/microsoft_word/), [Web](https://python.langchain.com/docs/how_to/document_loader_web/)

* [Pandas/Tabular data](https://python.langchain.com/api_reference/experimental/agents/langchain_experimental.agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent.html) or [here](https://python.langchain.com/docs/how_to/sql_csv/)

* [Local SQL database](https://danielroelfs.com/posts/querying-databases-using-langchain-and-ollama/)

* [RAG](https://www.elastic.co/search-labs/blog/local-rag-agent-elasticsearch-langgraph-llama3) 


