from backend.generation.chain import RAGChain
import asyncio

async def test():
    chain = RAGChain()
    print("Sending query...")
    stream, sources = await chain.answer_stream_async("Explain the A* search algorithm")
    print("Got stream! Sources:")
    print(sources)
    print("Reading tokens...")
    async for t in stream:
        print(t, end="", flush=True)
    print("\nDone!")

asyncio.run(test())
