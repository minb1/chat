import requests
from bs4 import BeautifulSoup, Comment, Tag
import os
import re
import unicodedata
import traceback
from functools import lru_cache
import concurrent.futures
import time
import yaml

# Performance Optimization Additions
html_soup_cache = {}

@lru_cache(maxsize=128)
def cached_sanitize_filename_part(text):
    """Cached wrapper for sanitize_filename_part."""
    if not text:
        return "untitled_chunk"
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[\s/:]+', '', text)
    text = re.sub(r'[^\w\-.]+', '', text)
    text = text.strip('-')
    text = re.sub(r'[-]{2,}', '', text)
    sanitized = text[:60] or "chunk"
    if not re.match(r'^section_', sanitized) and sanitized and sanitized[0].isdigit():
        sanitized = "" + sanitized
    if not sanitized or sanitized in ["", "__", "chunk"]:
        original_hash_part = abs(hash(text)) % 10000
        return f"chunk_{original_hash_part}"
    return sanitized.lower()

def get_top_level_index_files_per_folder():
    """
    Fetches paths to index.html files that are exactly one level deep
    within a top-level folder (e.g., api/adr/index.html, bomos/beheer/index.html),
    but not deeper ones (e.g., api/adr/1.0/index.html).
    """
    api_url = "https://api.github.com/repos/Logius-standaarden/publicatie/git/trees/main?recursive=1"
    try:
        response = requests.get(api_url, timeout=20)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching repository tree: {e}")
        return []

    try:
        data = response.json()
        if 'tree' not in data:
            print(f"Error: 'tree' key not found in GitHub API response. Response: {data}")
            return []
        tree = data['tree']
    except (KeyError, requests.exceptions.JSONDecodeError) as e:
        print(f"Error parsing repository tree JSON: {e}")
        return []

    index_files = []
    for item in tree:
        if item.get('type') == 'blob' and item.get('path', '').endswith('/index.html'):
            path = item['path']
            parts = path.split('/')
            if len(parts) == 3 and parts[1]:
                index_files.append(path)

    return index_files

def fetch_html_from_path(path):
    """Fetches raw HTML content from a GitHub raw URL."""
    base_url = "https://raw.githubusercontent.com/Logius-standaarden/publicatie/main/"
    raw_url = base_url + path
    try:
        response = requests.get(raw_url, timeout=20)
        response.raise_for_status()
        response.encoding = response.apparent_encoding if response.apparent_encoding else 'utf-8'
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {raw_url}: {e}")
        return None

def fetch_all_html_files_parallel(paths, max_workers=10):
    """Fetch multiple HTML files in parallel."""
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(fetch_html_from_path, path): path for path in paths}
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                html_content = future.result()
                if html_content is not None:
                    results[path] = html_content
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return results

def has_meaningful_content(html_string):
    """Checks if the HTML content has meaningful text after basic cleaning."""
    if not html_string:
        return False
    try:
        quick_check_text = re.sub(r'<script.?</script>', '', html_string, flags=re.DOTALL)
        quick_check_text = re.sub(r'<style.?</style>', '', quick_check_text, flags=re.DOTALL)
        quick_check_text = re.sub(r'<.*?>', ' ', quick_check_text)
        quick_check_text = re.sub(r'\s+', ' ', quick_check_text).strip()

        if len(quick_check_text) < 100:
            return False

        soup = BeautifulSoup(html_string, 'html.parser')

        selectors_to_remove = [
            "script", "style", "head", "meta", "link", "nav", "footer",
            ".sidelabel", ".dfn-panel", "div.head"
        ]

        for selector in selectors_to_remove:
            try:
                elements = soup.select(selector)
                for element in elements:
                    if isinstance(element, Tag):
                        element.decompose()
            except (NotImplementedError, Exception):
                pass

        text = soup.get_text(separator=" ", strip=True)

        min_length = 100
        placeholders = {'placeholder', 'coming soon', 'under construction', 'index', 'to be determined'}

        if not text or len(text) < min_length or text.lower() in placeholders:
            return False

    except Exception as e:
        print(f"Warning: Error during meaningful content check: {e}")
        return False

    return True

def sanitize_filename_part(text):
    """Sanitizes text for use in filenames more robustly."""
    return cached_sanitize_filename_part(text)

def extract_section_numbers(heading):
    """Extracts section numbers from a heading (e.g., '1.', '1.2', 'A.', 'A.1')."""
    if not heading: return None
    match = re.match(r'^[A-Za-z]?\s*(\d+(?:.\d+)*).?\s+', heading)
    return match.group(1) if match else None

def create_chunk_filename(chunk_data, index):
    """Creates a filename for a chunk based on its heading."""
    heading = chunk_data.get('heading', '')
    section_num = extract_section_numbers(heading)

    if section_num:
        heading_text_part = re.sub(r'^[A-Za-z]?\s*(\d+(?:\.\d+)*)\.?\s*', '', heading).strip()
        sanitized_heading = sanitize_filename_part(heading_text_part)
        section_part = section_num.replace('.', '_')
        if not sanitized_heading or sanitized_heading == "chunk":
            return f"section_{section_part}.txt"
        else:
            return f"section_{section_part}_{sanitized_heading}.txt"
    else:
        sanitized_heading = sanitize_filename_part(heading)
        if not sanitized_heading or sanitized_heading == "chunk":
            return f"chunk_{index:03d}.txt"
        elif len(sanitized_heading) < 3:
            return f"{sanitized_heading}_chunk_{index:03d}.txt"
        else:
            return f"{sanitized_heading}.txt"

def preprocess_html(html_string):
    """Optimized preprocessing of HTML content."""
    if html_string in html_soup_cache:
        return html_soup_cache[html_string]

    try:
        soup = BeautifulSoup(html_string, 'html.parser')
    except Exception as e:
        print(f"Error: Failed to parse HTML for preprocessing: {e}")
        return BeautifulSoup("", 'html.parser')

    for tag_name in ['script', 'style', 'head', 'meta', 'link', 'header', 'footer', 'nav']:
        for element in soup.find_all(tag_name):
            if isinstance(element, Tag):
                element.decompose()

    selectors_by_type = {
        'class': ['.sidelabel', '.dfn-panel', '.p-author', '.issue-container-generatedID',
                  '.jump-to-issues', '.respec-info', '.rule .flag', '.rule .rulelab',
                  '.figno', '.caption .fig-title'],
        'id': ['#toc', '#references', '#sotd', '#abstract', '#conformance',
               '#back-to-top', '#respec-ui', '#gh-contributors'],
        'tag': ['p.copyright', 'dl.bibliography', 'section.appendix',
                'section.introductory', 'section.notoc', 'div.head',
                'details.respec-tests-details', '.header-wrapper > a.self-link']
    }

    for selector_type, selectors in selectors_by_type.items():
        for selector in selectors:
            try:
                if selector_type == 'class':
                    class_name = selector[1:]
                    for element in soup.find_all(class_=class_name):
                        if isinstance(element, Tag):
                            element.decompose()
                elif selector_type == 'id':
                    id_name = selector[1:]
                    element = soup.find(id=id_name)
                    if element and isinstance(element, Tag):
                        element.decompose()
                else:
                    for element in soup.select(selector):
                        if isinstance(element, Tag):
                            element.decompose()
            except Exception:
                pass

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    html_soup_cache[html_string] = soup
    return soup

def extract_content_between_v3(soup, start_node, end_node, start_level):
    """Optimized version of content extraction that avoids repetitive DOM traversal."""
    if not isinstance(start_node, Tag):
        return "", []

    content_parts = []
    tables_html = []

    current = start_node.find_next()
    processed_nodes = set()

    while current and current != end_node:
        if current in processed_nodes:
            current = current.find_next()
            continue

        parent_processed = False
        for parent in current.parents:
            if parent in processed_nodes:
                parent_processed = True
                break
        if parent_processed:
            current = current.find_next()
            continue

        node_name = current.name

        if node_name is None or node_name in ['script', 'style', 'head', 'meta', 'link', 'nav']:
            processed_nodes.add(current)
            current = current.find_next()
            continue

        if node_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            try:
                heading_level = int(node_name[1])
                if heading_level > start_level:
                    break
                elif heading_level <= start_level:
                    break
            except (ValueError, IndexError):
                pass

        if node_name == 'table':
            table_html_str = str(current)
            if table_html_str not in tables_html:
                tables_html.append(table_html_str)
            processed_nodes.add(current)
            for desc in current.descendants:
                if isinstance(desc, Tag):
                    processed_nodes.add(desc)

        elif node_name in ['p', 'li', 'dd', 'dt', 'span', 'em', 'strong', 'a', 'code', 'pre', 'td', 'th', 'ul', 'ol', 'dl']:
            if current not in processed_nodes:
                node_texts = list(current.stripped_strings)
                filtered_texts = []
                for text in node_texts:
                    cleaned_text = re.sub(r'^(functional|technical)\s+/[a-zA-Z0-9_/:-]+\s*:\s*', '',
                                          text.strip()).strip()
                    if not re.fullmatch(r'\d+(\.\d+)+', cleaned_text) and cleaned_text:
                        filtered_texts.append(cleaned_text)
                if filtered_texts:
                    content_parts.append(' '.join(filtered_texts))
                processed_nodes.add(current)

        current = current.find_next()

    full_text = ' '.join(content_parts)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text, tables_html

def chunk_html_to_text_with_context_v3(html_string, path):
    """Optimized HTML chunking that reduces redundant operations."""
    if not html_string:
        return []

    try:
        soup = preprocess_html(html_string)
        body = soup.find('body')
        if not body:
            body = soup
    except Exception as e:
        print(f"Error during preprocessing of {path}: {e}")
        traceback.print_exc()
        return []

    headings = body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=True)

    if not headings:
        full_text, tables_html = extract_content_between_v3(soup, body, None, 0)
        if full_text or tables_html:
            title = 'Document Content'
            title_tag = soup.title
            if title_tag and isinstance(title_tag, Tag) and title_tag.string:
                title = title_tag.string.strip()
            elif body.find('h1'):
                h1_tag = body.find('h1')
                if isinstance(h1_tag, Tag):
                    title = h1_tag.get_text(strip=True)
            return [{'context': [], 'context_levels': [], 'chunk_text': full_text,
                     'heading': title, 'level': 1, 'tables_html': tables_html}]
        else:
            return []

    heading_levels = {}
    for heading in headings:
        if isinstance(heading, Tag):
            try:
                heading_levels[heading] = int(heading.name[1])
            except (ValueError, IndexError):
                pass

    next_boundary = {}
    for i, heading in enumerate(headings):
        for j in range(i + 1, len(headings)):
            next_heading = headings[j]
            if heading_levels.get(next_heading, 10) <= heading_levels.get(heading, 0):
                next_boundary[heading] = next_heading
                break

    chunks = []
    for heading in headings:
        if not isinstance(heading, Tag):
            continue

        current_level = heading_levels.get(heading)
        if current_level is None:
            continue

        heading_text = ' '.join(heading.stripped_strings)
        heading_text = re.sub(r'\s+', ' ', heading_text).strip().replace('§', '').strip()

        if not heading_text:
            continue

        # Extract or generate anchor
        anchor = heading.get('id', '')
        if not anchor:
            # Generate anchor from heading text
            anchor = re.sub(r'\W+', '-', heading_text.lower()).strip('-')

        context = []
        context_levels = []
        node = heading

        prev_headings = []
        while True:
            prev_heading = node.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not prev_heading or not isinstance(prev_heading, Tag):
                break
            prev_headings.append(prev_heading)
            node = prev_heading

        current_parent_level = current_level
        for prev_heading in prev_headings:
            prev_level = heading_levels.get(prev_heading)
            if prev_level is None:
                continue

            if prev_level < current_parent_level:
                prev_heading_text = ' '.join(prev_heading.stripped_strings)
                prev_heading_text = re.sub(r'\s+', ' ', prev_heading_text).strip().replace('§', '').strip()
                if prev_heading_text and prev_heading_text != heading_text:
                    context.insert(0, prev_heading_text)
                    context_levels.insert(0, prev_level)
                current_parent_level = prev_level

            if current_parent_level <= 1:
                break

        end_node = next_boundary.get(heading)

        content, tables_html = extract_content_between_v3(soup, heading, end_node, current_level)

        if not content and not tables_html:
            continue

        chunks.append({
            'context': context,
            'context_levels': context_levels,
            'chunk_text': content,
            'heading': heading_text,
            'level': current_level,
            'tables_html': tables_html,
            'anchor': anchor  # Add anchor to chunk data
        })

    return chunks

table_markdown_cache = {}

def html_table_to_markdown(table_html):
    """Cached and optimized HTML table to Markdown conversion."""
    if not table_html:
        return ""

    if table_html in table_markdown_cache:
        return table_markdown_cache[table_html]

    try:
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        if not table:
            table_markdown_cache[table_html] = ""
            return ""

        markdown_table = []

        headers = []
        header_row = table.find('thead')
        start_row_index = 0

        if header_row:
            header_rows_in_thead = header_row.find_all('tr')
            if header_rows_in_thead:
                headers = [th.get_text(separator=' ', strip=True).replace('|', '\\|') for th in
                           header_rows_in_thead[-1].find_all(['th', 'td'])]
        else:
            first_row = table.find('tr')
            if first_row and first_row.find('th'):
                headers = [th.get_text(separator=' ', strip=True).replace('|', '\\|') for th in
                           first_row.find_all(['th', 'td'])]
                start_row_index = 1

        leading_empty = 0
        trailing_empty = 0

        if headers:
            for h in headers:
                if not h:
                    leading_empty += 1
                else:
                    break

            for h in reversed(headers):
                if not h:
                    trailing_empty += 1
                else:
                    break

            headers = headers[leading_empty:len(headers) - trailing_empty]

            if headers:
                markdown_table.append('| ' + ' | '.join(headers) + ' |')
                markdown_table.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
            else:
                leading_empty = 0
                trailing_empty = 0

        body = table.find('tbody')
        rows = body.find_all('tr') if body else table.find_all('tr')[start_row_index:]

        for row in rows:
            cells = [cell.get_text(separator=' ', strip=True).replace('|', '\\|') for cell in
                     row.find_all(['td', 'th'])]

            if headers or leading_empty > 0 or trailing_empty > 0:
                original_cell_count = len(cells)
                cells = cells[leading_empty:original_cell_count - trailing_empty]

                if headers:
                    if len(cells) < len(headers):
                        cells.extend([''] * (len(headers) - len(cells)))
                    elif len(cells) > len(headers):
                        cells = cells[:len(headers)]

            if cells or headers:
                markdown_table.append('| ' + ' | '.join(cells) + ' |')

        result = '\n'.join(markdown_table) if markdown_table else ""
        table_markdown_cache[table_html] = result
        return result

    except Exception as e:
        print(f"Error converting table to Markdown: {e}")
        table_markdown_cache[table_html] = f"<!-- Error converting table: {e} -->\n{table_html}"
        return table_markdown_cache[table_html]

def construct_urls(original_html_path, anchor):
    """Constructs original_url and chunk_url from path and anchor."""
    base_url = "https://gitdocumentatie.logius.nl/publicatie/"
    folder_path = os.path.dirname(original_html_path)
    original_url = base_url + folder_path
    chunk_url = original_url + "#" + anchor if anchor else original_url
    return original_url, chunk_url

def process_and_save_chunks(path, html_content, output_directory, chunk_cache=None):
    """Optimized chunk processing and saving with anchor and URLs in metadata."""
    chunk_cache = chunk_cache or {}

    doc_tag = "unknown_doc"
    path_parts = path.split('/')
    if len(path_parts) == 3 and path_parts[1]:
        doc_tag = path_parts[1]
    elif len(path_parts) > 1:
        doc_tag = path_parts[0]
    print(f"  Derived doc_tag '{doc_tag}' from path '{path}'.")

    output_dir = os.path.join(output_directory, path_parts[0], doc_tag)
    os.makedirs(output_dir, exist_ok=True)

    cache_key = hash(html_content)
    if cache_key in chunk_cache:
        chunks = chunk_cache[cache_key]
    else:
        chunks = chunk_html_to_text_with_context_v3(html_content, path)
        chunk_cache[cache_key] = chunks

    if not chunks:
        print(f"  No processable chunks generated for {path}.")
        return 0

    files_to_write = []
    for index, chunk_data in enumerate(chunks):
        if not chunk_data.get('heading') and not chunk_data.get('chunk_text') and not chunk_data.get('tables_html'):
            continue

        filename = create_chunk_filename(chunk_data, index)
        chunk_relative_path_parts = [path_parts[0], doc_tag, filename]
        chunk_relative_path = '/'.join(chunk_relative_path_parts)
        filepath = os.path.join(output_dir, filename)

        # Construct URLs
        anchor = chunk_data.get('anchor', '')
        original_url, chunk_url = construct_urls(path, anchor)

        content = []
        content.append("---")
        content.append(f"file_path: {chunk_relative_path}")
        content.append(f"original_html_path: {path}")
        content.append(f"doc_tag: {doc_tag}")
        content.append(f"original_url: {original_url}")
        content.append(f"chunk_url: {chunk_url}")
        formatted_context = ["'{}'".format(c.replace("'", "\\'")) for c in chunk_data.get('context', [])]
        context_str = '[' + ', '.join(formatted_context) + ']'
        content.append(f"parent_sections: {context_str}")
        title = chunk_data.get('heading', f'Chunk {index}')
        title_str = title.replace('"', '\\"').replace('---', '- - -')
        content.append(f'title: "{title_str}"')
        anchor_str = anchor.replace('"', '\\"').replace('---', '- - -')
        content.append(f'anchor: "{anchor_str}"')
        content.append("---\n")

        for level, ctx in zip(chunk_data.get('context_levels', []), chunk_data.get('context', [])):
            cleaned_ctx = re.sub(r'\s+', ' ', ctx).strip()
            if cleaned_ctx:
                content.append(f"{'#' * level} {cleaned_ctx}")

        cleaned_heading = re.sub(r'\s+', ' ', title).strip()
        if cleaned_heading:
            content.append(f"{'#' * chunk_data.get('level', 1)} {cleaned_heading}\n")

        if chunk_data.get('chunk_text'):
            cleaned_text = re.sub(r'\n{3,}', '\n\n', chunk_data['chunk_text'])
            cleaned_text = re.sub(r' {2,}', ' ', cleaned_text).strip()
            if cleaned_text:
                content.append(cleaned_text + "\n")

        if chunk_data.get('tables_html'):
            for table_html in chunk_data['tables_html']:
                markdown_table = html_table_to_markdown(table_html)
                if markdown_table:
                    content.append(markdown_table + "\n")

        files_to_write.append((filepath, '\n'.join(content)))

    saved_count = 0
    for filepath, content in files_to_write:
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as txt_file:
                txt_file.write(content)
            saved_count += 1
        except Exception as e:
            print(f"  Error writing file {filepath}: {e}")

    print(f"  Saved {saved_count} chunks for {path} under doc_tag '{doc_tag}'.")
    return saved_count

def fetch_git_and_chunk():
    start_time = time.time()
    output_directory = "chunks_optimized"
    if os.path.exists(output_directory):
        print(f"Cleaning previous output directory: {output_directory}")
        import shutil
        try:
            shutil.rmtree(output_directory)
            print("Previous output directory cleaned.")
        except OSError as e:
            print(f"Error removing directory {output_directory}: {e}. Proceeding might lead to old data.")
    os.makedirs(output_directory, exist_ok=True)

    print("Fetching file list from GitHub...")
    files_to_process = get_top_level_index_files_per_folder()

    if not files_to_process:
        print("No index.html files found matching the criteria (e.g., api/adr/index.html). Exiting.")
        return

    print(f"Found {len(files_to_process)} index.html files to process.")

    print("Fetching HTML content in parallel...")
    html_contents = fetch_all_html_files_parallel(files_to_process)

    processed_count = 0
    skipped_no_content_count = 0
    failed_fetch_count = len(files_to_process) - len(html_contents)
    error_processing_count = 0

    print("Processing HTML content...")
    chunk_cache = {}

    meaningful_files = {}
    for path, html_content in html_contents.items():
        if has_meaningful_content(html_content):
            meaningful_files[path] = html_content
        else:
            print(f"  Skipping {path} due to lack of meaningful content.")
            skipped_no_content_count += 1

    total_chunks = 0
    for path, html_content in meaningful_files.items():
        print(f"\n>>> Processing {path}...")
        try:
            saved_chunks = process_and_save_chunks(path, html_content, output_directory, chunk_cache)
            total_chunks += saved_chunks
            processed_count += 1
        except Exception as e:
            print(f"!!! Top-level error processing {path}: {e}")
            error_processing_count += 1
            traceback.print_exc()

    end_time = time.time()
    execution_time = end_time - start_time

    print("\n--- Processing Summary ---")
    print(f"Successfully processed files:       {processed_count}")
    print(f"Total chunks saved:                 {total_chunks}")
    print(f"Skipped (no meaningful content):    {skipped_no_content_count}")
    print(f"Skipped (error during processing):  {error_processing_count}")
    print(f"Failed to fetch:                    {failed_fetch_count}")
    print(f"Total files attempted:              {len(files_to_process)}")
    print(f"Chunks saved to directory:          '{output_directory}'")
    print(f"Total execution time:               {execution_time:.2f} seconds")