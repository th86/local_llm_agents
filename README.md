# local_llm_agents

Set up locally hosted LLM agent stack. No subscription and no data leak.

- Tabular data analytics agent example `agent_analytics_tcga.py`
- RAG example `rag.py`, based on [RAG1](https://www.elastic.co/search-labs/blog/local-rag-agent-elasticsearch-langgraph-llama3) , [RAG2](https://www.singlestore.com/blog/build-a-local-ai-agent-python-ollama-langchain-singlestore/) , [RAG3](https://dev.to/mohsin_rashid_13537f11a91/rag-with-ollama-1049)



## Windows

Install python from App Store

Install [Ollama](https://ollama.com/) and pull models in CMD

`ollama pull gemma3:4b` (multimodal) or `deepseek-r1` (reasoning) or `llama3.2` (tool)

```
pip install ollama
pip install -qU langchain_ollama
```

Should use [miniforge](https://github.com/conda-forge/miniforge) to manage the environment

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

## Biomni on locally hosted LLM
[Biomni](https://github.com/snap-stanford/Biomni) is a general purpose biomedical agent.

Install the python package `pip install biomni` and the dependencies (langraph, openai, aws, google gen ai). This may take more than half of an hour.

```
import sys
sys.path.append("../")
from biomni.agent import A1

agent = A1(path='./', llm='llama3.2:latest')
agent.go("Predict ADMET properties for this compound: C1=CC=C(C=C1)C=O")
```
llama3.2:latest works, deepseek does not return useful results.

## TODO

* [Browser Use local setup](https://docs.browser-use.com/development/local-setup)

* [FastMCP](https://gofastmcp.com/)

* [LangChain Document Loader PDF](https://python.langchain.com/docs/how_to/document_loader_pdf/), [Word](https://python.langchain.com/docs/integrations/document_loaders/microsoft_word/), [Web](https://python.langchain.com/docs/how_to/document_loader_web/)

* [Pandas/Tabular data](https://python.langchain.com/api_reference/experimental/agents/langchain_experimental.agents.agent_toolkits.pandas.base.create_pandas_dataframe_agent.html) or [here](https://python.langchain.com/docs/how_to/sql_csv/)

* [Local SQL database](https://danielroelfs.com/posts/querying-databases-using-langchain-and-ollama/)

* [qwen3-coder](https://ollama.com/library/qwen3-coder) coding agent
