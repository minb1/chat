import requests
from bs4 import BeautifulSoup
import os
import re

def fetch_html_content(url):
    """Fetch HTML content from a given URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def chunk_html_to_text_with_context(html_string):
    """
    Chunks HTML into text chunks based on headings, excluding parent chunks.
    Only creates chunks for leaf sections (sections without subsections).
    """
    soup = BeautifulSoup(html_string, 'html.parser')
    headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    chunks = []

    if not headings:
        return [{'context': [], 'chunk_text': html_string, 'heading': 'document', 'level': 0}]

    # Identify parent headings
    is_parent = [False] * len(headings)
    for i, heading in enumerate(headings):
        current_level = int(heading.name[1])
        if i + 1 < len(headings):
            next_level = int(headings[i + 1].name[1])
            if next_level > current_level:
                is_parent[i] = True

    # Create chunks for non-parent (leaf) headings
    for i, heading in enumerate(headings):
        if is_parent[i]:
            continue
        current_level = int(heading.name[1])
        heading_text = heading.get_text(separator=" ", strip=True)
        context = []
        context_levels = []
        for j in range(i - 1, -1, -1):
            prev_level = int(headings[j].name[1])
            if prev_level < current_level:
                context.insert(0, headings[j].get_text(separator=" ", strip=True))
                context_levels.insert(0, prev_level)
                current_level = prev_level
        content = ""
        next_element = heading.find_next_sibling()
        has_table = False
        table_html = ""
        table_element = None

        if next_element and next_element.name == 'table':
            has_table = True
            table_element = next_element
            table_html = str(next_element)

        while next_element:
            if next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                break
            if has_table and next_element == table_element:
                next_element = next_element.find_next_sibling()
                continue
            content += next_element.get_text(separator="\n", strip=True) + "\n"
            next_element = next_element.find_next_sibling()

        chunks.append({
            'context': context,
            'context_levels': context_levels,
            'chunk_text': content.strip(),
            'heading': heading_text,
            'level': int(heading.name[1]),
            'has_table': has_table,
            'table_html': table_html if has_table else None
        })

    return chunks

def extract_section_numbers(heading_text):
    """Extract section numbers from heading text if present."""
    match = re.match(r'^(\d+(\.\d+)*\.?)\s+', heading_text)
    if match:
        return match.group(1).rstrip('.')
    return None

def extract_base_filename_from_url(url):
    """Extracts a base filename from the given URL."""
    parts = url.replace("https://github.com/", "").replace("https://raw.githubusercontent.com/", "") \
               .replace("/blob/main/", "/").replace("/main/", "/").split('/')
    repo_name = parts[1] if len(parts) > 1 else "default_repo"
    document_name = os.path.splitext(parts[-1])[0] if parts else "default_doc"
    base_filename = f"{repo_name}_{document_name}"
    return sanitize_filename_part(base_filename)

def sanitize_filename_part(text):
    """Sanitizes text for filename usage."""
    text = text.strip()
    sanitized = text.lower().replace(" ", "-").replace(".", "-").replace(":", "-")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized

def create_descriptive_filename(url, chunk_context, chunk_heading):
    """
    Creates a descriptive filename that reflects the document hierarchy.
    Format: repo_doc_sectionpath_leafheading.txt
    """
    base_filename = extract_base_filename_from_url(url)
    section_path = []
    for context_heading in chunk_context:
        section_num = extract_section_numbers(context_heading)
        if section_num:
            section_path.append(section_num)
        else:
            section_path.append(sanitize_filename_part(context_heading))
    leaf_section_num = extract_section_numbers(chunk_heading)
    if leaf_section_num:
        leaf_heading = f"{leaf_section_num}-{sanitize_filename_part(chunk_heading.split(' ', 1)[1] if ' ' in chunk_heading else chunk_heading)}"
    else:
        leaf_heading = sanitize_filename_part(chunk_heading)
    filename_parts = [base_filename]
    if section_path:
        filename_parts.append('-'.join(section_path))
    filename_parts.append(leaf_heading)
    return "_".join(filename_parts) + ".txt"

def html_table_to_markdown(table_html):
    """Convert an HTML table to Markdown format."""
    if not table_html:
        return ""
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ""
    markdown_table = []
    headers = []
    header_row = table.find('thead')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    else:
        first_row = table.find('tr')
        if first_row:
            headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]
    if headers:
        markdown_table.append('| ' + ' | '.join(headers) + ' |')
        markdown_table.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
    rows = table.find_all('tr')
    start_index = 1 if headers and not table.find('thead') else 0
    for row in rows[start_index:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            markdown_table.append('| ' + ' | '.join([cell.get_text(strip=True) for cell in cells]) + ' |')
    return '\n'.join(markdown_table)

def get_section_path(context, heading):
    """Generate a dot-separated section path from context and heading."""
    path = []
    for ctx in context:
        section_num = extract_section_numbers(ctx)
        if section_num:
            path.append(section_num)
    section_num = extract_section_numbers(heading)
    if section_num:
        path.append(section_num)
    return '.'.join(path) if path else ''

if __name__ == "__main__":
    # Testing the chunking module
    github_url = "https://github.com/Logius-standaarden/publicatie/blob/main/api/adr-beheer/index.html"
    # Convert to raw URL
    raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/main/", "/main/")
    html_content = fetch_html_content(raw_url)
    if html_content:
        chunks = chunk_html_to_text_with_context(html_content)
        output_directory = "html_chunks_enhanced_rag"
        os.makedirs(output_directory, exist_ok=True)
        for chunk in chunks:
            filename = create_descriptive_filename(raw_url, chunk['context'], chunk['heading'])
            filepath = os.path.join(output_directory, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("---\n")
                f.write(f"path: {get_section_path(chunk['context'], chunk['heading'])}\n")
                f.write(f"parent_sections: {chunk['context']}\n")
                f.write(f"title: \"{chunk['heading']}\"\n")
                f.write("---\n\n")
                for idx, ctx in enumerate(chunk['context']):
                    level = chunk['context_levels'][idx] if 'context_levels' in chunk and idx < len(chunk['context_levels']) else idx + 1
                    f.write(f"{'#'*level} {ctx}\n")
                f.write(f"\n{'#'*chunk['level']} {chunk['heading']}\n\n")
                f.write(chunk['chunk_text'])
                if chunk['has_table']:
                    markdown_table = html_table_to_markdown(chunk['table_html'])
                    if markdown_table:
                        f.write("\n\n" + markdown_table)
            print(f"Saved: {filepath}")
    else:
        print("Failed to fetch HTML content")
