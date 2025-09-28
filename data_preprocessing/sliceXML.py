import datetime
from lxml import etree
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
POSTS_INPUT_XML_FILE = os.getenv("POSTS_XML_FILE_PATH")
POSTS_OUTPUT_XML_FILE = os.getenv("POSTS_SLICED_XML_FILE_PATH")
COMMENTS_INPUT_XML_FILE = os.getenv("COMMENTS_XML_FILE_PATH")
COMMENTS_OUTPUT_XML_FILE = os.getenv("COMMENTS_SLICED_XML_FILE_PATH")

def filter_by_creation_date(input_file, output_file, element_tag, min_date_str, max_date_str):
    min_date = datetime.datetime.strptime(min_date_str, "%Y-%m-%d")
    max_date = datetime.datetime.strptime(max_date_str, "%Y-%m-%d")
    context = etree.iterparse(input_file, events=('end',), tag=element_tag)
    with open(output_file, 'wb') as f:
        f.write(b'<posts>\n')
        for _, elem in context:
            creation_date = elem.get('CreationDate')
            if creation_date:
                elem_date = datetime.datetime.strptime(creation_date[:10], "%Y-%m-%d")
                if min_date <= elem_date <= max_date:
                    f.write(etree.tostring(elem) + b'\n')
            elem.clear()
        f.write(b'</posts>\n')

if __name__ == '__main__':
    ROW_TAG = 'row'
    MIN_DATE = '2024-01-01'
    MAX_DATE = '2024-01-10'

    print(f"Filtering '{POSTS_INPUT_XML_FILE}' for rows from {MIN_DATE}...")
    try:
        filter_by_creation_date(POSTS_INPUT_XML_FILE, POSTS_OUTPUT_XML_FILE, ROW_TAG, MIN_DATE,MAX_DATE)
        print(f"Successfully created '{POSTS_OUTPUT_XML_FILE}' with rows from {MIN_DATE}.")
    except FileNotFoundError:
        print(f"Error: The file '{POSTS_INPUT_XML_FILE}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    print(f"Filtering '{COMMENTS_INPUT_XML_FILE}' for rows from {MIN_DATE}...")
    try:
        filter_by_creation_date(COMMENTS_INPUT_XML_FILE, COMMENTS_OUTPUT_XML_FILE, ROW_TAG, MIN_DATE,MAX_DATE)
        print(f"Successfully created '{COMMENTS_OUTPUT_XML_FILE}' with rows from {MIN_DATE}.")
    except FileNotFoundError:
        print(f"Error: The file '{COMMENTS_INPUT_XML_FILE}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")