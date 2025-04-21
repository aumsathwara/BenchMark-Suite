from mcp.server.fastmcp import FastMCP
from pathlib import Path
import os
import base64
from typing import List

# Initialize the MCP server with the name "FileSystem"
mcp = FastMCP(name="FileSystem")

# Security configuration
ALLOWED_DIRS = [Path.cwd()]

def normalize_path(file_path: str) -> Path:
    """Normalize and resolve path with security checks."""
    expanded = Path(os.path.expanduser(file_path))
    resolved = expanded.resolve(strict=False)  # Resolve symlinks

    # Validate path components recursively
    for i in range(1, len(resolved.parts) + 1):
        partial = Path(*resolved.parts[:i])
        if partial.is_symlink():
            real_partial = partial.resolve(strict=True)
            if not is_path_allowed(real_partial):
                raise PermissionError(f"Symlink target restricted: {real_partial}")

    return resolved

def is_path_allowed(resolved_path: Path) -> bool:
    """Check if path is within allowed directories."""
    return any(resolved_path.is_relative_to(allowed_dir) for allowed_dir in ALLOWED_DIRS)

def validate_file_operation(path: Path) -> None:
    """Validate path for file operations."""
    parent = path.parent.resolve()
    if not parent.exists():
        raise FileNotFoundError(f"Parent directory does not exist: {parent}")
    if not is_path_allowed(parent):
        raise PermissionError(f"Parent directory not allowed: {parent}")

# Tool to read a file
@mcp.tool(name="read_file", description="Reads the content of a file")
async def read_file(file_path: str) -> list:
    """Read file content with binary support."""
    try:
        normalized = normalize_path(file_path)
        if not is_path_allowed(normalized):
            raise PermissionError("Access denied")

        with open(normalized, "rb") as f:
            content = f.read()

            # Attempt UTF-8 decoding for text files
            try:
                text_content = content.decode('utf-8')
                return [{"type": "text", "text": text_content}]
            except UnicodeDecodeError:
                # Fallback to base64 for binary files
                return [{"type": "text", "text": base64.b64encode(content).decode('ascii')}]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {str(e)}"}]

# Tool to write content to a file
@mcp.tool(name="write_file", description="Writes content to a file")
async def write_file(file_path: str, content: str) -> list:
    """Write content to a file with comprehensive validation."""
    try:
        normalized = normalize_path(file_path)
        validate_file_operation(normalized)

        if not is_path_allowed(normalized.parent):
            raise PermissionError("Parent directory not whitelisted")

        with open(normalized, "a") as f:
            f.write(content)
            return [{"type": "text", "text": "Updated file content successfully"}]
    except Exception as e:
        return [{"type": "text", "text": f"Error: {str(e)}"}]

# Tool to search for files by extension
@mcp.tool(name="search_file", description="Search for files with a specific extension")
async def search_file(file_extension: str, search_dir: str = str(Path.cwd())) -> list:
    """Search for files with a specific extension within allowed directories."""
    try:
        resolved_search_dir = normalize_path(search_dir).resolve(strict=True)

        if not is_path_allowed(resolved_search_dir):
            return [{"type": "text", "text": "Error: Search directory not in allowed paths"}]

        results = [
            str(resolved)
            for resolved in resolved_search_dir.rglob(f"*{file_extension}")
            if is_path_allowed(resolved)
        ]

        if not results:
            return [{"type": "text", "text": f"No files found with {file_extension}"}]

        return [{"type": "text", "text": "\n".join(results)}]

    except Exception as e:
        return [{"type": "text", "text": f"Error: {str(e)}"}]

def main():
    mcp.run(transport='stdio')
    
    
if __name__ == "__main__":
    main()