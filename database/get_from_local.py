import os

def retrieve_chunks(relative_paths, data_folder="data/chunks"):
    formatted_output = []

    for rel_path in relative_paths:
        # Construct the full path
        full_path = os.path.join(data_folder, rel_path.replace("\\", "/"))

        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as file:
                content = file.read()
                formatted_output.append(f"**{full_path}**:\n\"{content}\"\n")
        else:
            formatted_output.append(f"**{full_path}**:\n(File not found)\n")

    return "\n".join(formatted_output)


import os


def retrieve_docs(file_paths):
    docs = {}
    seen = set()  # Track processed (folder, subfolder) pairs

    for fp in file_paths:
        # Normalize path separators and trim any whitespace
        normalized_fp = fp.replace("\\", "/").strip()
        parts = normalized_fp.split("/")

        # Determine folder and subfolder based on the file path structure.
        if parts[0].lower() == "data" and parts[1].lower() == "chunks":
            # Expecting structure: data/chunks/<folder>/<subfolder>/chunk_xxx.txt
            if len(parts) < 4:
                continue
            folder = parts[2]
            subfolder = parts[3]
        else:
            # Assume structure: <folder>/<subfolder>/chunk_xxx.txt
            if len(parts) < 2:
                continue
            folder = parts[0]
            subfolder = parts[1]

        # Use a tuple (in lowercase) to avoid duplicates due to case differences
        key = (folder.lower(), subfolder.lower())
        if key in seen:
            continue
        seen.add(key)

        # Build normalized markdown file path: data/markdown_files/<folder>/<subfolder>.md
        md_path = os.path.normpath(os.path.join("data", "markdown_files", folder, f"{subfolder}.md"))
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                docs[md_path] = f.read()
        except Exception as e:
            docs[md_path] = f"Error loading markdown file: {e}"
    return docs
