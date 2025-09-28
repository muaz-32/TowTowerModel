import xml.etree.ElementTree as ET
import csv
import os

# Root Directory
ROOT_DIR = "../.."

def generate_questions_table_with_tags(file_path, output_csv):
    """
    Generate a table for questions asked by users, storing one row per tag for each question.

    Args:
        file_path (str): Path to the large XML file (Posts.xml).
        output_csv (str): Path to output the CSV file.

    Returns:
        None
    """
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Open the output CSV file
    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Writing the header row
        writer.writerow([
            "UserId",
            "QuestionId",
            "AcceptedAnswerId",
            "CreationDate",
            "Score",
            "ViewCount",
            "IsCommunityOwned",
            "Tag",
            "AnswerCount",
            "CommentCount",
            "FavouriteCount"
        ])

        # Use iterparse for efficient XML processing
        context = ET.iterparse(file_path, events=("start", "end"))
        for event, elem in context:
            if event == "end" and elem.tag == "row":  # Process only <row> elements
                post_type = elem.get("PostTypeId")

                if post_type == "1":  # Question
                    # Extract necessary fields
                    question_id = elem.get("Id")
                    owner_user_id = elem.get("OwnerUserId", "null")
                    accepted_answer_id = elem.get("AcceptedAnswerId", "null")
                    creation_date = elem.get("CreationDate")
                    score = elem.get("Score", "null")
                    view_count = elem.get("ViewCount", "null")
                    community_owned_date = elem.get("CommunityOwnedDate")
                    is_community_owned = "true" if community_owned_date else "false"
                    tags = elem.get("Tags", "").strip('|')
                    answer_count = elem.get("AnswerCount", "null")
                    comment_count = elem.get("CommentCount", "null")
                    favourite_count = elem.get("FavouriteCount", "null")

                    # Split tags by the "|" character and create a row for each tag
                    tag_list = tags.split('|')
                    for tag in tag_list:
                        writer.writerow([
                            owner_user_id,
                            question_id,
                            accepted_answer_id,
                            creation_date,
                            score,
                            view_count,
                            is_community_owned,
                            tag,
                            answer_count,
                            comment_count,
                            favourite_count
                        ])

                # Clear element from memory to reduce memory usage
                elem.clear()


# File paths
input_file = f"/media/muazul-islam/Transcend/StackOverflow/FilteredPosts_2024_JAN_1-10.xml"  # Replace with actual file path
output_file = f"{ROOT_DIR}/output/stackoverflow/dump_2024/users.questions.table.csv"

# Generate the questions table
generate_questions_table_with_tags(input_file, output_file)
print(f"Questions table saved to {output_file}")