import chainlit as cl
import os
import weaviate
import weaviate.classes as wvc
from literalai import LiteralClient
from openai import AsyncOpenAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

client = LiteralClient(api_key=os.getenv("LITERAL_API_KEY"))
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

cl.instrument_openai()

def create_weaviate_index():
    # initialize Weaviate
    weaviate_client = weaviate.connect_to_embedded(
        headers={
            "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")  # Replace with your API key
        },
    )

    # Index documents to Weaviate
    movie_collection = weaviate_client.collections.get("Movies")
    # movie_collection = weaviate_client.collections.delete("Movies")

    if not weaviate_client.collections.exists("Movies"):

        weaviate_client.collections.create(
            "Movies",
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai()
        )

        df = pd.read_csv('data/movies.CSV').sample(n=10)

        movie_collection = weaviate_client.collections.get("Movies")

        with movie_collection.batch.dynamic() as batch:
            for index, data_row in df.iterrows():
                properties = {
                    "title": data_row["title"],
                    "content": data_row["overview"]
                }
                batch.add_object(
                    properties=properties,
                )

    return movie_collection

weaviate_index = create_weaviate_index()


# Embed and retrieve context (semantic search in Weaviate)
@cl.step(name="Retrieve", type="retrieval")
async def retrieve(query):
    if weaviate_index == None:
        raise Exception("Weaviate index not initialized")
    response = weaviate_index.query.near_text(query="query", limit=5, return_metadata=wvc.query.MetadataQuery(distance=True))

    return response


# Generate answer with retrieved context
@cl.step(name="Generate", type="llm")
async def generate(
    prompt,
    chat_model="gpt-4-turbo-preview",
):
    messages = cl.user_session.get("messages", [])
    messages.append(prompt)
    settings = {"temperature": 0, "stream": True, "model": chat_model}
    stream = await openai_client.chat.completions.create(messages=messages, **settings)
    message = cl.message.Message(content="")
    await message.send()

    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await message.stream_token(token)

    await message.update()
    messages.append({"role": "assistant", "content": message.content})
    cl.user_session.set("messages", messages)
    return message.content

        
# Call context retriever and answer generator
@cl.step(name="Query", type="run")
async def run(query):
    stored_contexts = await retrieve(query)
    contexts = []
    prompt = await client.api.get_prompt(name="RAG prompt")

    if not prompt:
        raise Exception("Prompt not found")
    for object in stored_contexts.objects:
        title = object.properties["title"]
        summary = object.properties["content"]
        context = f"Movie title: {title}. Movie summary: {summary}"
        contexts.append(context)

    completion = await generate(prompt.format({"context": contexts, "question": query})[-1])

    return completion

# Find and set prompt on chat start
@cl.on_chat_start
async def on_chat_start():
    prompt = await client.api.get_prompt(name="RAG prompt")

    if not prompt:
        raise Exception("Prompt not found")
    cl.user_session.set(
        "messages",
        [prompt.format()[0]],
    )

@cl.on_message
async def main(message: cl.Message):
    await run(message.content)