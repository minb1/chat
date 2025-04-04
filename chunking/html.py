import requests
from bs4 import BeautifulSoup
import os
import re


def fetch_html_from_github_url(github_url):
    """Fetches HTML from GitHub URL."""
    try:
        raw_url = github_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/main/", "/main/")
        response = requests.get(raw_url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
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

    # First pass: identify which headings are parents (have direct subsections)
    is_parent = [False] * len(headings)
    for i, heading in enumerate(headings):
        current_level = int(heading.name[1])

        # Check if the next heading is a subsection of the current heading
        if i + 1 < len(headings):
            next_level = int(headings[i + 1].name[1])
            if next_level > current_level:
                is_parent[i] = True

    # Second pass: create chunks only for non-parent headings (leaf sections)
    for i, heading in enumerate(headings):
        if is_parent[i]:
            continue  # Skip parent headings

        current_level = int(heading.name[1])
        heading_text = heading.get_text(separator=" ", strip=True)

        # Build context by finding all parent headings with their levels
        context = []
        context_levels = []
        for j in range(i - 1, -1, -1):
            prev_level = int(headings[j].name[1])
            if prev_level < current_level:
                context.insert(0, headings[j].get_text(separator=" ", strip=True))
                context_levels.insert(0, prev_level)
                current_level = prev_level

        # Extract content for this heading
        content = ""  # Don't include heading in content
        next_element = heading.find_next_sibling()

        # Check if next element contains a table
        has_table = False
        table_html = ""
        table_element = None

        if next_element and next_element.name == 'table':
            has_table = True
            table_element = next_element
            table_html = str(next_element)

        # Collect all text content until next heading
        while next_element:
            if next_element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                break

            # Skip table element as we'll process it separately
            if has_table and next_element == table_element:
                next_element = next_element.find_next_sibling()
                continue

            content += next_element.get_text(separator="\n", strip=True) + "\n"
            next_element = next_element.find_next_sibling()

        chunks.append({
            'context': context,
            'context_levels': context_levels,
            'chunk_text': content.strip(),
            'heading': heading_text,  # Store the current heading text
            'level': int(heading.name[1]),  # Store the heading level
            'has_table': has_table,
            'table_html': table_html if has_table else None
        })

    return chunks


def extract_section_numbers(heading_text):
    """Extract section numbers from heading text if present."""
    # Match patterns like "1.", "1.2.", "1.2.3.", etc.
    match = re.match(r'^(\d+(\.\d+)*\.?)\s+', heading_text)
    if match:
        return match.group(1).rstrip('.')
    return None


def create_descriptive_filename(github_url, chunk_context, chunk_heading):
    """
    Creates a descriptive filename that reflects the document hierarchy and content.
    Format: repo_doc_section-path_leaf-heading.txt
    """
    base_filename = extract_base_filename_from_url(github_url)

    # Extract section numbers if present
    section_path = []
    for context_heading in chunk_context:
        section_num = extract_section_numbers(context_heading)
        if section_num:
            section_path.append(section_num)
        else:
            # If no section number, use a sanitized version of the heading
            section_path.append(sanitize_filename_part(context_heading))

    # Extract section number from the chunk heading itself
    leaf_section_num = extract_section_numbers(chunk_heading)
    if leaf_section_num:
        leaf_heading = f"{leaf_section_num}-{sanitize_filename_part(chunk_heading.split(' ', 1)[1] if ' ' in chunk_heading else chunk_heading)}"
    else:
        leaf_heading = sanitize_filename_part(chunk_heading)

    # Build filename parts
    filename_parts = [base_filename]

    # Add section path if it exists
    if section_path:
        filename_parts.append('-'.join(section_path))

    # Add leaf heading
    filename_parts.append(leaf_heading)

    return "_".join(filename_parts) + ".txt"


def extract_base_filename_from_url(github_url):
    """Extracts base filename from URL path."""
    path_parts = github_url.replace("https://github.com/", "").replace("https://raw.githubusercontent.com/",
                                                                       "").replace("/blob/main/", "/").replace("/main/",
                                                                                                               "/").split(
        '/')
    repo_name = path_parts[1] if len(path_parts) > 1 else "default_repo"
    document_name = os.path.splitext(path_parts[-1])[0] if len(path_parts) > 0 else "default_doc"
    base_filename = f"{repo_name}_{document_name}"
    return sanitize_filename_part(base_filename)


def sanitize_filename_part(text):
    """Sanitizes text for filename."""
    # First, handle the case where there might be a section number at the beginning
    text = text.strip()
    sanitized = text.lower().replace(" ", "-").replace(".", "-").replace(":", "-")
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', sanitized)
    # Limit length to avoid extremely long filenames
    if len(sanitized) > 50:
        sanitized = sanitized[:50]
    return sanitized


def html_table_to_markdown(table_html):
    """Convert HTML table to Markdown format"""
    if not table_html:
        return ""

    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ""

    markdown_table = []

    # Process header row
    headers = []
    header_row = table.find('thead')
    if header_row:
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
    else:
        # Try first row if no thead
        first_row = table.find('tr')
        if first_row:
            headers = [th.get_text(strip=True) for th in first_row.find_all(['th', 'td'])]

    if headers:
        markdown_table.append('| ' + ' | '.join(headers) + ' |')
        markdown_table.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')

    # Process body rows
    rows = table.find_all('tr')
    # Skip first row if we used it as header
    start_index = 1 if headers and not table.find('thead') else 0

    for row in rows[start_index:]:
        cells = row.find_all(['td', 'th'])
        if cells:
            markdown_table.append('| ' + ' | '.join([cell.get_text(strip=True) for cell in cells]) + ' |')

    return '\n'.join(markdown_table)


def get_section_path(context, heading):
    """Generate a path string from context and heading"""
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
    github_html_url = "https://github.com/Logius-standaarden/publicatie/blob/main/api/adr-beheer/index.html"
    html_content = fetch_html_from_github_url(github_html_url)

    if html_content:
        text_chunks = chunk_html_to_text_with_context(html_content)
        output_directory = "html_chunks_enhanced_rag"
        os.makedirs(output_directory, exist_ok=True)

        for i, chunk_data in enumerate(text_chunks):
            filename = create_descriptive_filename(
                github_html_url,
                chunk_data['context'],
                chunk_data['heading']
            )
            filepath = os.path.join(output_directory, filename)

            with open(filepath, 'w', encoding='utf-8') as txt_file:
                # Add metadata section
                section_path = get_section_path(chunk_data['context'], chunk_data['heading'])
                txt_file.write("---\n")
                txt_file.write(f"path: {section_path}\n")
                txt_file.write(f"parent_sections: {chunk_data['context']}\n")
                txt_file.write(f"title: \"{chunk_data['heading']}\"\n")
                txt_file.write("---\n\n")

                # Write context as properly formatted markdown headers
                for idx, ctx in enumerate(chunk_data['context']):
                    level = chunk_data['context_levels'][idx] if 'context_levels' in chunk_data and idx < len(
                        chunk_data['context_levels']) else idx + 1
                    prefix = '#' * level
                    txt_file.write(f"{prefix} {ctx}\n")

                # Write the current heading with proper level
                current_level = chunk_data['level']
                heading_prefix = '#' * current_level
                txt_file.write(f"{heading_prefix} {chunk_data['heading']}\n\n")

                # Write the actual chunk content
                txt_file.write(chunk_data['chunk_text'])

                # If there's a table, convert it to markdown and append
                if chunk_data['has_table']:
                    markdown_table = html_table_to_markdown(chunk_data['table_html'])
                    if markdown_table:
                        txt_file.write("\n\n" + markdown_table)

            print(f"Saved: {filepath}")
            print(f"Context: {chunk_data['context']}")
            print(f"Heading: {chunk_data['heading']}")
            print("-" * 50)

        print(f"\nAll chunks saved to directory: {output_directory}")

    else:
        print("Could not fetch HTML content. Script aborted.")
