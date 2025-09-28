import xml.etree.ElementTree as ET
import csv

# Root Directory
ROOT_DIR = "../.."

def generate_answers_table(file_path, output_csv):
    """
    Generate a table for answers provided by users, with one row for each tag associated with a question.

    Args:
        file_path (str): Path to the large XML file (Posts.xml).
        output_csv (str): Path to output the CSV file.

    Returns:
        None
    """
    # Dictionary to store question tags
    question_tags = {}

    # Open the output CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Writing the header row
        writer.writerow([
            "UserId",
            "AnswerId",
            "QuestionId",
            "IsAcceptedAnswer",
            "CreationDate",
            "Score",
            "ViewCount",
            "IsCommunityOwned",
            "Tag",
            "CommentCount",
            "FavouriteCount"
        ])

        # Use iterparse for efficient XML processing
        context = ET.iterparse(file_path, events=("start", "end"))
        for event, elem in context:
            if event == "end" and elem.tag == "row":  # Process only <row> elements
                post_type = elem.get("PostTypeId")

                if post_type == "1":  # Question
                    question_id = elem.get("Id")
                    tags = elem.get("Tags", "")
                    question_tags[question_id] = tags.strip('|').split('|')

                elif post_type == "2":  # Answer
                    # Extract necessary fields
                    answer_id = elem.get("Id")
                    owner_user_id = elem.get("OwnerUserId")
                    parent_id = elem.get("ParentId")
                    is_accepted = parent_id and elem.get("Id") == elem.get("AcceptedAnswerId")
                    creation_date = elem.get("CreationDate")
                    score = elem.get("Score")
                    view_count = elem.get("ViewCount", "null")
                    community_owned_date = elem.get("CommunityOwnedDate")
                    is_community_owned = "true" if community_owned_date else "false"
                    comment_count = elem.get("CommentCount", "null")
                    favourite_count = elem.get("FavouriteCount", "null")

                    # Retrieve parent question's tags
                    tags_for_answer = question_tags.get(parent_id, [])

                    # Write a row for each tag associated with the answer
                    for tag in tags_for_answer:
                        writer.writerow([
                            owner_user_id,
                            answer_id,
                            parent_id,
                            "true" if is_accepted else "false",
                            creation_date,
                            score,
                            view_count,
                            is_community_owned,
                            tag,
                            comment_count,
                            favourite_count
                        ])

                # Clear element from memory to reduce memory usage
                elem.clear()


# File paths
# input_file = f"{ROOT_DIR}/data/genai.stackexchange.com/Posts.xml" # Replace with actual file path
# output_file = f"{ROOT_DIR}/output/dump/users.answers.table.csv"
input_file = f"/media/muazul-islam/Transcend/StackOverflow/FilteredPosts_2024_JAN_1-10.xml"  # Replace with actual file path
output_file = f"{ROOT_DIR}/output/stackoverflow/dump_2024/users.answers.table.csv"

# Generate the answers table
generate_answers_table(input_file, output_file)
print(f"Answers table saved to {output_file}")