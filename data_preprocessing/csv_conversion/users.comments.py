import xml.etree.ElementTree as ET
import csv

# Root Directory
ROOT_DIR = "../.."

def extract_post_data(posts_file_path):
    """
    Extract post information including Tags and PostTypeId, resolving both for questions and answers.

    Args:
        posts_file_path (str): Path to the Posts.xml file.

    Returns:
        dict: Mapping of PostId to a dictionary with Tags and PostTypeId.
              For example: { "101": {"tags": "neural-networks, deep-learning", "postTypeId": "1"} }
    """
    question_tags = {}  # Store tags for questions
    post_mapping = {}  # Final mapping of PostId to its Tags and PostTypeId

    # Helper function to clean and parse Tags
    def clean_tags(raw_tags):
        """
        Cleans and parses the raw Tags string into a human-readable format.

        Args:
            raw_tags (str): Raw Tags string from Posts.xml (e.g., "|tag1|tag2|tag3|").

        Returns:
            str: Comma-separated list of tags (e.g., "tag1, tag2, tag3").
        """
        if not raw_tags:
            return "null"
        # Remove leading and trailing | and split by |
        return ", ".join(raw_tags.strip("|").split("|"))

    # First Pass: Collect all question tags
    context = ET.iterparse(posts_file_path, events=("start", "end"))
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            post_id = elem.get("Id")
            post_type = elem.get("PostTypeId")
            tags = elem.get("Tags")

            if post_type == "1":  # It's a question
                # Clean and store question tags
                cleaned_tags = clean_tags(tags)
                question_tags[post_id] = cleaned_tags

                # Add to post_mapping with postTypeId
                post_mapping[post_id] = {"tags": cleaned_tags, "postTypeId": "1"}

            elem.clear()  # Clear memory

    # Second Pass: Map answers to questions and finalize mapping
    context = ET.iterparse(posts_file_path, events=("start", "end"))
    for event, elem in context:
        if event == "end" and elem.tag == "row":
            post_id = elem.get("Id")
            post_type = elem.get("PostTypeId")
            parent_id = elem.get("ParentId")

            if post_type == "2":  # It's an answer
                # Resolve answer tags using the parent question's tags
                parent_tags = question_tags.get(parent_id, "null")
                post_mapping[post_id] = {"tags": parent_tags, "postTypeId": "2"}

            elem.clear()  # Clear memory

    return post_mapping


def convert_comments_to_csv(comments_file_path, posts_file_path, output_csv):
    """
    Converts Comments.xml into a CSV file with each tag of a comment stored in a new row. Adds PostTypeId column.

    Args:
        comments_file_path (str): Path to the Comments.xml file.
        posts_file_path (str): Path to the Posts.xml file.
        output_csv (str): Path to output the CSV file.

    Returns:
        None
    """
    # Build post-to-tags and postTypeId mapping
    post_mapping = extract_post_data(posts_file_path)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Writing the header row
        writer.writerow([
            "UserId",
            "CommentId",
            "PostId",
            "PostTypeId",
            "Tag",
            "CreationDate"
        ])

        # Process Comments.xml
        context = ET.iterparse(comments_file_path, events=("start", "end"))
        for event, elem in context:
            if event == "end" and elem.tag == "row":  # Process each <row> element
                comment_id = elem.get("Id")
                post_id = elem.get("PostId")
                user_id = elem.get("UserId", "null")
                creation_date = elem.get("CreationDate")

                # Get resolved tags and postTypeId from post_mapping
                post_data = post_mapping.get(post_id, {"tags": "null", "postTypeId": "null"})
                tags = post_data["tags"]
                post_type_id = post_data["postTypeId"]

                # If tags exist, write one row for each tag
                if tags != "null":
                    tags_list = tags.split(", ")  # Split tags into a list (comma-separated values)
                    for tag in tags_list:
                        writer.writerow([
                            user_id,
                            comment_id,
                            post_id,
                            post_type_id,
                            tag,
                            creation_date
                        ])
                else:
                    # If no tags, write a single row with "null" for the tag
                    writer.writerow([
                        user_id,
                        comment_id,
                        post_id,
                        post_type_id,
                        "null",
                        creation_date
                    ])

                # Clear element from memory
                elem.clear()



# File paths
comments_file = f"/media/muazul-islam/Transcend/StackOverflow/FilteredComments_2024_JAN_1-10.xml"  # Replace with actual file path
posts_file = f"/media/muazul-islam/Transcend/StackOverflow/FilteredPosts_2024_JAN_1-10.xml"  # Replace with actual file path
output_file = f"{ROOT_DIR}/output/stackoverflow/dump_2024/users.comments.table.csv"

# Generate the comments table with tags
convert_comments_to_csv(comments_file, posts_file, output_file)
print(f"Comments table with tags saved to {output_file}")