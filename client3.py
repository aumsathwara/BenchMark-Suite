import os
import asyncio
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)

class SimpleChatClient:
    """A plain Gemini-based chat client without function-calling capabilities."""
    def __init__(self):
        self.client = client

    async def process_query(self, query: str) -> dict:
        """
        Sends a user query to Gemini and returns both the text response and usage metadata.
        """
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[query],
            # No tools defined
        )
        # Extract usage metadata for token counts
        um = response.usage_metadata
        prompt_tokens = um.prompt_token_count
        resp_tokens   = um.candidates_token_count

        # Get the text of the first candidate
        text = response.candidates[0].content.parts[0].text

        return {
            "text": text,
            "usage_metadata": {
                "prompt_token_count": prompt_tokens,
                "response_token_count": resp_tokens
            }
        }

    async def chat_loop(self):
        """Runs an interactive REPL, printing responses and token usage."""
        print("\nSimple Chat Client Started! (No Tools)")
        print("Type your queries or 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ('quit', 'exit'):
                break
            result = await self.process_query(query)
            text = result['text']
            um   = result['usage_metadata']
            total = um['prompt_token_count'] + um['response_token_count']

            print(f"\n{text}")
            print(f"\n[Tokens used: {total} (prompt {um['prompt_token_count']}, response {um['response_token_count']})]")

async def async_main():
    client = SimpleChatClient()
    await client.chat_loop()

if __name__ == '__main__':
    asyncio.run(async_main())
