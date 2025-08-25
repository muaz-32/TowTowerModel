import pandas as pd
import numpy as np
from collections import defaultdict

def main():
    # Read the users.questions.table.csv file
    questions_file = pd.read_csv('./data/dump/users.questions.table.csv', header=0,
                                 names=['UserId', 'QuestionId', 'AcceptedAnswerId', 'CreationDate',
                                        'Score', 'ViewCount', 'IsCommunityOwned', 'Tag',
                                        'AnswerCount', 'CommentCount', 'FavoriteCount'])

    # Read and process users.answers.table.csv
    answers_file = pd.read_csv('./data/dump/users.answers.table.csv', header=0,
                               names=['UserId', 'AnswerId', 'QuestionId', 'IsAcceptedAnswer',
                                      'CreationDate', 'Score', 'ViewCount', 'IsCommunityOwned',
                                      'Tag', 'CommentCount', 'FavoriteCount'])

    # Read and process users.comments.table.csv
    comments_file = pd.read_csv('./data/dump/users.comments.table.csv', header=0,
                                names=['UserId', 'CommentId', 'PostId', 'PostTypeId', 'Tag', 'CreationDate'])

    user_tags = defaultdict(list)
    question_tags = defaultdict(list)

    # User tags ------------------------------------------------------------
    for _, row in questions_file.iterrows():
        user_id = row['UserId']
        tag = row['Tag']
        if pd.notna(user_id):  # Check if UserId is not null
            user_tags[user_id].append(tag)

    for _, row in answers_file.iterrows():
        user_id = row['UserId']
        tag = row['Tag']
        if pd.notna(user_id):  # Check if UserId is not null
            user_tags[user_id].append(tag)

    for _, row in comments_file.iterrows():
        user_id = row['UserId']
        tag = row['Tag']
        if pd.notna(user_id):  # Check if UserId is not null
            user_tags[user_id].append(tag)

    print(f"Number of users with tags: {len(user_tags)}")
    print(f"Sample user tags:")
    for i, (uid, tags) in enumerate(list(user_tags.items())[:5]):
        print(f"User {uid}: {tags[:20]}...")  # Show first 20 tags
        print(f"Total tags for user {uid}: {len(tags)}")

    # Question tags --------------------------------------------------------
    for _, row in questions_file.iterrows():
        question_id = row['QuestionId']
        tag = row['Tag']
        question_tags[question_id].append(tag)

    print(f"Number of questions with tags: {len(question_tags)}")
    print(f"Sample question tags:")
    for i, (qid, tags) in enumerate(list(question_tags.items())[:5]):
        print(f"Question {qid}: {tags}")

    # Labels matrix --------------------------------------------------------
    answers_file = answers_file.dropna(subset=['UserId'])  # Remove rows with NaN UserId

    unique_users = answers_file['UserId'].unique()
    unique_questions = answers_file['QuestionId'].unique()
    users_count = len(unique_users)
    questions_count = len(unique_questions)

    # Create mappings for indexing
    user_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    question_to_index = {q_id: idx for idx, q_id in enumerate(unique_questions)}

    # Initialize the label matrix (users x questions)
    label_matrix = np.zeros((users_count, questions_count), dtype=int)

    # Fill the matrix: 1 if user answered the question, 0 otherwise
    for _, row in answers_file.iterrows():
        user_idx = user_to_index[row['UserId']]
        question_idx = question_to_index[row['QuestionId']]
        label_matrix[user_idx][question_idx] = 1

    print(f"Label matrix shape: {label_matrix.shape}")
    print(f"Sample label matrix (first 5 users and questions):\n{label_matrix[:5, :5]}")

if __name__ == "__main__":
    main()
