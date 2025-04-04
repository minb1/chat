import requests

def get_html_file_urls(repo_owner, repo_name, branch="main"):
    """
    Fetches a list of HTML files (exactly 2 folders deep) from a GitHub repository using the GitHub API.
    Returns a list of dictionaries with keys: 'path' and 'raw_url'.
    """
    tree_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/git/trees/{branch}?recursive=1"
    response = requests.get(tree_url)
    response.raise_for_status()
    tree = response.json().get("tree", [])
    html_files = []
    for item in tree:
        if item['type'] == 'blob' and item['path'].endswith(".html"):
            # Expect paths like folder1/folder2/file.html (i.e. 3 parts)
            parts = item['path'].split("/")
            if len(parts) == 3:
                raw_url = f"https://raw.githubusercontent.com/{repo_owner}/{repo_name}/{branch}/{item['path']}"
                html_files.append({
                    "path": item['path'],
                    "raw_url": raw_url
                })
    return html_files

if __name__ == "__main__":
    # Example usage:
    repo_owner = "Logius-standaarden"
    repo_name = "publicatie"
    files = get_html_file_urls(repo_owner, repo_name)
    for f in files:
        print(f)
