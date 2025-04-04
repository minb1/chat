import requests
from bs4 import BeautifulSoup, Comment, Tag
import os
import re
import unicodedata
import traceback
from functools import lru_cache
import concurrent.futures
import time

# --- Performance Optimization Additions ---
# Cache for BeautifulSoup objects
html_soup_cache = {}


# LRU cache for expensive operations
@lru_cache(maxsize=128)
def cached_sanitize_filename_part(text):
    """Cached wrapper for sanitize_filename_part."""
    if not text:
        return "untitled_chunk"
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[\s/:]+', '_', text)
    text = re.sub(r'[^\w\-.]+', '', text)
    text = text.strip('_-')
    text = re.sub(r'[-_]{2,}', '_', text)
    sanitized = text[:60] or "chunk"
    if not re.match(r'^section_', sanitized) and sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    if not sanitized or sanitized in ["_", "__", "chunk"]:
        original_hash_part = abs(hash(text)) % 10000
        return f"chunk_{original_hash_part}"
    return sanitized.lower()


def get_top_level_index_files_per_folder(): # Renamed for clarity
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
        # Check if it's a file ('blob') and the path ends with '/index.html'
        if item.get('type') == 'blob' and item.get('path', '').endswith('/index.html'):
            path = item['path']
            parts = path.split('/')
            # Check if the path has exactly 3 parts:
            # e.g., ['api', 'adr', 'index.html'] -> length 3 - YES
            # e.g., ['api', 'adr', '1.0', 'index.html'] -> length 4 - NO
            # Also ensure the middle part (sub_folder) is not empty
            if len(parts) == 3 and parts[1]:
                index_files.append(path)

    return index_files


# Parallel fetching of HTML content
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


# --- Content Check and Sanitization ---
def has_meaningful_content(html_string):
    """Checks if the HTML content has meaningful text after basic cleaning."""
    if not html_string:
        return False
    try:
        # Use a simple parsing approach first
        quick_check_text = re.sub(r'<script.*?</script>', '', html_string, flags=re.DOTALL)
        quick_check_text = re.sub(r'<style.*?</style>', '', quick_check_text, flags=re.DOTALL)
        quick_check_text = re.sub(r'<.*?>', ' ', quick_check_text)
        quick_check_text = re.sub(r'\s+', ' ', quick_check_text).strip()

        # Quick length check for efficiency
        if len(quick_check_text) < 100:
            return False

        # Only do full parsing if needed
        soup = BeautifulSoup(html_string, 'html.parser')

        # Selectors for elements to remove
        selectors_to_remove = [
            "script", "style", "head", "meta", "link", "nav", "footer",
            ".sidelabel", ".dfn-panel", "div.head"
        ]

        # Find and decompose elements matching the selectors
        for selector in selectors_to_remove:
            try:
                elements = soup.select(selector)
                for element in elements:
                    if isinstance(element, Tag):
                        element.decompose()
            except (NotImplementedError, Exception) as e:
                pass  # Skip errors in selector processing

        # Get text after removals
        text = soup.get_text(separator=" ", strip=True)

        # Check text length and against common placeholders
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
    # Use cached version for performance
    return cached_sanitize_filename_part(text)


def extract_section_numbers(heading):
    """Extracts section numbers from a heading (e.g., '1.', '1.2', 'A.', 'A.1')."""
    if not heading: return None
    match = re.match(r'^[A-Za-z]?\s*(\d+(?:\.\d+)*)\.?\s+', heading)
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


# --- Optimized HTML Processing ---
def preprocess_html(html_string):
    """Optimized preprocessing of HTML content."""
    # Check if we've already processed this exact HTML
    if html_string in html_soup_cache:
        return html_soup_cache[html_string]

    try:
        soup = BeautifulSoup(html_string, 'html.parser')
    except Exception as e:
        print(f"Error: Failed to parse HTML for preprocessing: {e}")
        return BeautifulSoup("", 'html.parser')

    # Batch removal of elements for better performance
    for tag_name in ['script', 'style', 'head', 'meta', 'link', 'header', 'footer', 'nav']:
        for element in soup.find_all(tag_name):
            if isinstance(element, Tag):
                element.decompose()

    # Optimized selector removal - do in batches by selector type
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

    # Process by selector type
    for selector_type, selectors in selectors_by_type.items():
        for selector in selectors:
            try:
                if selector_type == 'class':
                    class_name = selector[1:]  # Remove the leading dot
                    for element in soup.find_all(class_=class_name):
                        if isinstance(element, Tag):
                            element.decompose()
                elif selector_type == 'id':
                    id_name = selector[1:]  # Remove the leading #
                    element = soup.find(id=id_name)
                    if element and isinstance(element, Tag):
                        element.decompose()
                else:
                    # Use select for complex selectors
                    for element in soup.select(selector):
                        if isinstance(element, Tag):
                            element.decompose()
            except Exception as e:
                pass  # Skip errors in selector processing

    # Remove comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Store in cache
    html_soup_cache[html_string] = soup
    return soup


# Optimized content extraction
# --- Optimized HTML Processing ---

# ... (keep preprocess_html and other preceding functions as they are) ...

# Optimized content extraction
def extract_content_between_v3(soup, start_node, end_node, start_level):
    """Optimized version of content extraction that avoids repetitive DOM traversal
       and stops when a subheading is encountered."""
    if not isinstance(start_node, Tag):
        return "", []

    content_parts = []
    tables_html = []

    # Iterate through sibling and subsequent nodes efficiently
    current = start_node.find_next() # Start processing *after* the heading itself
    processed_nodes = set() # Keep track of nodes already handled (like table descendants)

    while current and current != end_node:
        # If node is already processed (e.g., part of a table we handled), skip
        if current in processed_nodes:
            current = current.find_next()
            continue

        # Check if any parent was already processed (avoids double-counting)
        parent_processed = False
        for parent in current.parents:
            if parent in processed_nodes:
                parent_processed = True
                break
        if parent_processed:
            current = current.find_next()
            continue

        node_name = current.name

        # Skip non-content elements early
        if node_name is None or node_name in ['script', 'style', 'head', 'meta', 'link', 'nav']:
            processed_nodes.add(current)
            current = current.find_next()
            continue

        # Handle headings - check if this is a new section or a SUB-section
        if node_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            try:
                heading_level = int(node_name[1])
                # *** CHANGE HERE ***
                # If we encounter ANY heading at a level deeper than the start_node's level,
                # it signifies the start of a subsection. Stop processing for the current chunk.
                if heading_level > start_level:
                    # print(f"DEBUG: Stopping for {start_node.name} ({start_level}) because found subheading {current.name} ({heading_level})")
                    break # Stop collecting content for the parent section

                # If we encounter a heading at the same level or higher, it's the boundary
                # (This should technically be caught by end_node, but good safety check)
                elif heading_level <= start_level:
                    # print(f"DEBUG: Stopping for {start_node.name} ({start_level}) because found same/higher heading {current.name} ({heading_level})")
                    break # Stop collecting content, new section starts

            except (ValueError, IndexError):
                 # Ignore malformed heading tags like <h7> etc.
                 pass # Continue processing content after it

        # Handle tables - store once and process all descendants
        if node_name == 'table':
            table_html_str = str(current)
            if table_html_str not in tables_html:
                tables_html.append(table_html_str)
            processed_nodes.add(current)
            # Mark all descendants as processed so we don't extract their text separately
            for desc in current.descendants:
                 if isinstance(desc, Tag):
                    processed_nodes.add(desc)

        # Extract text from content tags ONLY if not inside a processed container (like a table)
        elif node_name in ['p', 'li', 'dd', 'dt', 'span', 'em', 'strong', 'a', 'code', 'pre', 'td', 'th', 'ul', 'ol', 'dl']:
             # Check again if this node itself was processed (e.g., a <td> inside a handled <table>)
             if current not in processed_nodes:
                node_texts = list(current.stripped_strings)
                filtered_texts = []
                for text in node_texts:
                    cleaned_text = re.sub(r'^(functional|technical)\s+/[a-zA-Z0-9_/:-]+\s*:\s*', '',
                                          text.strip()).strip()
                    # Avoid adding just section numbers as content
                    if not re.fullmatch(r'\d+(\.\d+)+', cleaned_text) and cleaned_text:
                        filtered_texts.append(cleaned_text)
                if filtered_texts:
                    content_parts.append(' '.join(filtered_texts))
                processed_nodes.add(current) # Mark this node as processed

        # Move to the next node in the document order
        current = current.find_next()

    full_text = ' '.join(content_parts)
    full_text = re.sub(r'\s+', ' ', full_text).strip()
    return full_text, tables_html


# --- (Rest of the code remains the same) ---

# ... (chunk_html_to_text_with_context_v3, html_table_to_markdown, process_and_save_chunks, main, etc.) ...


# Optimized chunking function
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

    # Find all headings at once to avoid repeated searches
    headings = body.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'], recursive=True)

    if not headings:
        # Create single chunk for the entire document
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

    # Create a dictionary to quickly find heading levels
    heading_levels = {}
    for heading in headings:
        if isinstance(heading, Tag):
            try:
                heading_levels[heading] = int(heading.name[1])
            except (ValueError, IndexError):
                pass

    # Create a dictionary to find the next heading at same or higher level
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

        # Build context efficiently
        context = []
        context_levels = []
        node = heading

        # Pre-compute all previous headings at once
        prev_headings = []
        while True:
            prev_heading = node.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not prev_heading or not isinstance(prev_heading, Tag):
                break
            prev_headings.append(prev_heading)
            node = prev_heading

        # Process context in reverse order
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

        # Get end node from our pre-computed boundaries
        end_node = next_boundary.get(heading)

        # Extract content
        content, tables_html = extract_content_between_v3(soup, heading, end_node, current_level)

        # Skip empty chunks
        if not content and not tables_html:
            continue

        chunks.append({
            'context': context,
            'context_levels': context_levels,
            'chunk_text': content,
            'heading': heading_text,
            'level': current_level,
            'tables_html': tables_html
        })

    return chunks


# --- Optimized Table Conversion ---
# Cache for table conversions
table_markdown_cache = {}


def html_table_to_markdown(table_html):
    """Cached and optimized HTML table to Markdown conversion."""
    if not table_html:
        return ""

    # Check cache first
    if table_html in table_markdown_cache:
        return table_markdown_cache[table_html]

    try:
        soup = BeautifulSoup(table_html, 'html.parser')
        table = soup.find('table')
        if not table:
            table_markdown_cache[table_html] = ""
            return ""

        markdown_table = []

        # Extract headers
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

        # Process headers
        leading_empty = 0
        trailing_empty = 0

        if headers:
            # Find leading empty headers
            for h in headers:
                if not h:
                    leading_empty += 1
                else:
                    break

            # Find trailing empty headers
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

        # Process rows efficiently
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


# --- Optimized Saving Function ---
def process_and_save_chunks(path, html_content, output_directory, chunk_cache=None):
    """Optimized chunk processing and saving with optional caching."""
    chunk_cache = chunk_cache or {}

    # Create output directory
    output_dir = os.path.join(output_directory, os.path.dirname(path))
    os.makedirs(output_dir, exist_ok=True)

    # Check if chunks are already cached
    cache_key = hash(html_content)
    if cache_key in chunk_cache:
        chunks = chunk_cache[cache_key]
    else:
        # Generate chunks
        chunks = chunk_html_to_text_with_context_v3(html_content, path)
        chunk_cache[cache_key] = chunks

    if not chunks:
        print(f"  No processable chunks generated for {path}.")
        return 0

    # Prepare all chunk data before writing
    files_to_write = []
    for index, chunk_data in enumerate(chunks):
        if not chunk_data.get('heading') and not chunk_data.get('chunk_text') and not chunk_data.get('tables_html'):
            continue

        filename = create_chunk_filename(chunk_data, index)
        filepath = os.path.join(output_dir, filename)

        # Build content for this file
        content = []

        # YAML Frontmatter
        content.append("---")
        content.append(f"path: {path}")
        formatted_context = ["'{}'".format(c.replace("'", "\\'")) for c in chunk_data.get('context', [])]
        context_str = '[' + ', '.join(formatted_context) + ']'
        content.append(f"parent_sections: {context_str}")
        title = chunk_data.get('heading', f'Chunk {index}')
        title_str = title.replace('"', '\\"')
        content.append(f'title: "{title_str}"')
        content.append("---\n")

        # Markdown Content
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

    # Write files
    saved_count = 0
    for filepath, content in files_to_write:
        try:
            with open(filepath, 'w', encoding='utf-8') as txt_file:
                txt_file.write(content)
            saved_count += 1
        except Exception as e:
            print(f"  Error writing file {filepath}: {e}")

    print(f"  Saved {saved_count} chunks for {path}.")
    return saved_count


# --- Main Execution Block ---
def fetch_git_and_chunk():
    start_time = time.time()
    output_directory = "chunks_optimized"
    os.makedirs(output_directory, exist_ok=True)

    print("Fetching file list from GitHub...")
    files_to_process = get_top_level_index_files_per_folder()

    if not files_to_process:
        print("No index.html files found matching the criteria (e.g., api/adr/index.html). Exiting.")
        return

    print(f"Found {len(files_to_process)} index.html files to process.")

    # Fetch all HTML content in parallel
    print("Fetching HTML content in parallel...")
    html_contents = fetch_all_html_files_parallel(files_to_process)

    processed_count = 0
    skipped_no_content_count = 0
    failed_fetch_count = len(files_to_process) - len(html_contents)
    error_processing_count = 0

    # Process files with meaningful content
    print("Processing HTML content...")
    chunk_cache = {}  # Cache for chunks

    # Filter out files without meaningful content
    meaningful_files = {}
    for path, html_content in html_contents.items():
        if has_meaningful_content(html_content):
            meaningful_files[path] = html_content
        else:
            print(f"  Skipping {path} due to lack of meaningful content.")
            skipped_no_content_count += 1

    # Process meaningful files
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
