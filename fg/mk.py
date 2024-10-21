from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from typing import Union, List
import os
import re
from dotenv import load_dotenv
from crawl4ai import WebCrawler
from openai import OpenAI
import tempfile
import uvicorn


# Load environment variables
load_dotenv("C:\SAARAGH\.env")

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Define URL request model
class URLRequest(BaseModel):
    urls: Union[List[str], str]

    @field_validator('urls')
    def ensure_list(cls, v):
        if isinstance(v, str):
            return [v]
        return v

# Define question request model
class QuestionRequest(BaseModel):
    question: str

# Helper functions
def list_to_string(my_list, separator=', '):
    return separator.join(map(str, my_list))

# Scrape content from URLs and save as Markdown
def scrape_urls(urls):
    print("URLS", urls)
    crawler = WebCrawler()
    crawler.warmup()
    patterns = [
        r"##  Subscribe to our emails.*?\* Opens in a new window\.",
        r"Skip to content.*?View cart Check out  Continue shopping"
    ]
    scraped_contents = []
    for url in urls:
        try:
            result = crawler.run(url)
            content = result.markdown
            for pattern in patterns:
                content = re.sub(pattern, "", content, flags=re.DOTALL)
            scraped_contents.append(content)
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=f"Failed to scrape {url}: {str(e)}")
    
    # Combine all scraped contents into a single Markdown file
    combined_content = "\n\n---\n\n".join(scraped_contents)
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as temp_file:
        temp_file.write(combined_content)
        temp_file_path = temp_file.name
    
    return temp_file_path

# Create OpenAI Assistant and Vector Store
def create_assistant_and_vectorstore(file_path):
    description = """
    You are the question answer bot who answers the user
    question with precise response by using only the vectorstore's content
    """
    instructions = """
    Answer only from the stored content, in case if you couldn't
    find the answer say that you couldn't found the answer
    """

    assistant = client.beta.assistants.create(
        name="Question-Answer Assistant",
        description=description,
        instructions=instructions,
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
    )

    vector_store = client.beta.vector_stores.create(name="Scraped Content")

    with open(file_path, "rb") as file_stream:
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=[file_stream]
        )

    print(file_batch.status)
    print(file_batch.file_counts)

    assistant = client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    return assistant, vector_store

# Execute RAG pipeline using OpenAI
def execute_rag_pipeline(question: str, assistant, vector_store):
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": question,
            }
        ]
    )

    print(thread.tool_resources.file_search)

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id, assistant_id=assistant.id
    )

    messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))
    message_content = messages[0].content[0].text
    annotations = message_content.annotations
    citations = []

    for index, annotation in enumerate(annotations):
        message_content.value = message_content.value.replace(annotation.text, f"[{index}]")
        if file_citation := getattr(annotation, "file_citation", None):
            cited_file = client.files.retrieve(file_citation.file_id)
            citations.append(f"[{index}] {cited_file.filename}")

    answer = message_content.value + "\n\n" + "\n".join(citations)
    return answer

# FastAPI routes
@app.post("/scrape_and_upsert")
def scrape_and_upsert(request: URLRequest):
    try:
        urls = request.urls
        md_file_path = scrape_urls(urls)
        assistant, vector_store = create_assistant_and_vectorstore(md_file_path)
        return {
            "message": f"Successfully scraped content from {len(urls)} URLs and created assistant and vector store.",
            "assistant_id": assistant.id,
            "vector_store_id": vector_store.id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask_question")
def ask_question(request: QuestionRequest, assistant_id: str, vector_store_id: str):
    try:
        assistant = client.beta.assistants.retrieve(assistant_id)
        vector_store = client.beta.vector_stores.retrieve(vector_store_id)
        answer = execute_rag_pipeline(request.question, assistant, vector_store)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
