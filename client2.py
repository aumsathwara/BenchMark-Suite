import os
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
import asyncio

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Base directory allowed for file ops
ALLOWED_BASE = Path.cwd()

def normalize_path(file_path: str) -> Path:
    expanded = Path(os.path.expanduser(file_path))
    resolved = expanded.resolve(strict=False)
    if not resolved.is_relative_to(ALLOWED_BASE):
        raise PermissionError(f"Access denied: {resolved}")
    return resolved

async def read_file(file_path: str) -> str:
    path = normalize_path(file_path)
    data = path.read_bytes()
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return base64.b64encode(data).decode('ascii')

async def write_file(file_path: str, content: str) -> str:
    path = normalize_path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding='utf-8') if path.exists() else ''
    path.write_text(existing + content, encoding='utf-8')
    return "Write successful"

async def search_file(extension: str, search_dir: str = '.') -> list[str]:
    dirp = normalize_path(search_dir)
    return [str(p) for p in dirp.rglob(f"*{extension}")]

# Function schemas for Gemini (no unsupported fields)
func_schemas = [
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "write_file",
        "description": "Append content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "content": {"type": "string", "description": "Content to append"}
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "search_file",
        "description": "Search for files by extension",
        "parameters": {
            "type": "object",
            "properties": {
                "extension": {"type": "string", "description": "File extension"},
                "search_dir": {"type": "string", "description": "Directory to search"}
            },
            "required": ["extension"]
        }
    }
]
# Wrap function schemas in a Tool object
tool_def = types.Tool(function_declarations=func_schemas)

class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    async def process_query(self, query: str) -> dict:
        """
        Sends the query to Gemini, optionally calls local functions,
        and returns a dict with 'text' and 'usage_metadata'.
        """
        # Initial call
        response = self.client.models.generate_content(
            model="gemini-1.5-flash",
            contents=[query],
            config=types.GenerateContentConfig(
                tools=[tool_def]
            )
        )
        # Pull usage_metadata from the response
        um = response.usage_metadata  # UsageMetadata holds token counts :contentReference[oaicite:0]{index=0}
        prompt_tokens = um.prompt_token_count
        resp_tokens   = um.candidates_token_count

        msg = response.candidates[0].content.parts[0]

        # If the model wants to call a function
        if hasattr(msg, 'function_call') and msg.function_call:
            fname = msg.function_call.name
            raw_args = msg.function_call.args
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            print(f"\n[Calling function {fname} with args {args}]")
            if fname == "read_file":
                result = await read_file(**args)
            elif fname == "write_file":
                result = await write_file(**args)
            elif fname == "search_file":
                result = await search_file(**args)
            else:
                result = f"Unknown function: {fname}"
            print(f"\nResults: {result}")

            # Feed result back for natural-language response
            follow = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[json.dumps({"name": fname, "result": result})],
                config=types.GenerateContentConfig(
                    tools=[tool_def]
                )
            )
            # Add follow-up usage
            fum = follow.usage_metadata
            prompt_tokens += fum.prompt_token_count
            resp_tokens   += fum.candidates_token_count

            final_text = follow.candidates[0].content.parts[0].text
        else:
            final_text = msg.text

        return {
            "text": final_text,
            "usage_metadata": {
                "prompt_token_count": prompt_tokens,
                "response_token_count": resp_tokens
            }
        }

    async def chat_loop(self):
        print("\nGemini Client Started! (Function Calling)")
        print("Type your queries or 'quit' to exit.")
        while True:
            query = input("\nQuery: ").strip()
            if query.lower() in ('quit', 'exit'):
                break
            try:
                result = await self.process_query(query)
                text = result["text"]
                um   = result["usage_metadata"]
                total = um["prompt_token_count"] + um["response_token_count"]
                print(f"\n{text}")
                print(f"\n[Tokens used: {total} "
                      f"(prompt {um['prompt_token_count']}, "
                      f"response {um['response_token_count']})]")
            except Exception as e:
                print(f"Error: {e}")

    async def cleanup(self):
        pass

async def async_main():
    client = GeminiClient()
    try:
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(async_main())
