- One important challenge is prepare the data for RAG.
- Handle most types of formats.
- Clean noises and unnecessary information. 
- PDF:
- 1.  **Layout Complexity**: PDFs can contain complex layouts, such as multi-column text, tables, images, and intricate formatting. This layout diversity complicates the extraction of structured data.
- 2.  **Font encoding issue**s: PDFs use a variety of font encoding systems, and some of these systems do not map directly to Unicode. This can make it difficult to extract the text accurately.
- 3.  **Non-linear text storage:**  PDFs do not store text in the order it appears on the page. Instead, they store text in objects that can be placed anywhere on the page. This means that the order of the text in the underlying code may not match the order of the text as it appears visually.
- 4.  **Inconsistent use of spaces**: In some PDFs, spaces are not used consistently or are not used at all between words. This can make it difficult to even identify word boundaries.


We need to test llamaprase and also check the information in this article: https://www.llamaindex.ai/blog/mastering-pdfs-extracting-sections-headings-paragraphs-and-tables-with-cutting-edge-parser-faea18870125
