import pandas as pd
import numpy as np
from collections import defaultdict

def extract_data():
    # Read the users.questions.table.csv file
    questions_file = pd.read_csv('./data/dump/users.questions.table.csv', header=0,
                                 names=['UserId', 'QuestionId', 'AcceptedAnswerId', 'CreationDate',
                                        'Score', 'ViewCount', 'IsCommunityOwned', 'Tag',
                                        'AnswerCount', 'CommentCount', 'FavoriteCount'])

    # Convert UserId to int, handling NaN values
    questions_file = questions_file.dropna(subset=['UserId'])
    questions_file['UserId'] = questions_file['UserId'].astype(int)

    # Read and process users.answers.table.csv
    answers_file = pd.read_csv('./data/dump/users.answers.table.csv', header=0,
                               names=['UserId', 'AnswerId', 'QuestionId', 'IsAcceptedAnswer',
                                      'CreationDate', 'Score', 'ViewCount', 'IsCommunityOwned',
                                      'Tag', 'CommentCount', 'FavoriteCount'])

    # Convert UserId to int, handling NaN values
    answers_file = answers_file.dropna(subset=['UserId'])
    answers_file['UserId'] = answers_file['UserId'].astype(int)

    # Read and process users.comments.table.csv
    comments_file = pd.read_csv('./data/dump/users.comments.table.csv', header=0,
                                names=['UserId', 'CommentId', 'PostId', 'PostTypeId', 'Tag', 'CreationDate'])

    # Convert UserId to int, handling NaN values
    comments_file = comments_file.dropna(subset=['UserId'])
    comments_file['UserId'] = comments_file['UserId'].astype(int)

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

    # Question tags --------------------------------------------------------
    for _, row in questions_file.iterrows():
        question_id = row['QuestionId']
        tag = row['Tag']
        question_tags[question_id].append(tag)

    # Get unique users and questions from answers (interactions)
    unique_users = list(user_tags.keys())
    unique_questions = list(question_tags.keys())

    # Create user and question ID mappings
    user_id_to_index = {user_id: idx for idx, user_id in enumerate(unique_users)}
    question_id_to_index = {q_id: idx for idx, q_id in enumerate(unique_questions)}

    # Convert user tags to lists (matching generate.py format)
    user_topic_lists = []
    for user_id in unique_users:
        if user_id in user_tags:
            user_topic_lists.append(list(user_tags[user_id]))
        else:
            user_topic_lists.append([])

    # Convert question tags to lists (items in generate.py format)
    item_topic_lists = []
    for question_id in unique_questions:
        if question_id in question_tags:
            item_topic_lists.append(list(question_tags[question_id]))
        else:
            item_topic_lists.append([])

    # Create interactions from answers (user answered question = positive interaction)
    interactions = []

    for _, row in answers_file.iterrows():
        user_id = row['UserId']
        question_id = row['QuestionId']

        if user_id in user_id_to_index and question_id in question_id_to_index:
            user_idx = user_id_to_index[user_id]
            question_idx = question_id_to_index[question_id]
            interactions.append([user_idx, question_idx, 1])  # Positive interaction

    # Create some negative interactions (users who didn't answer certain questions)
    # Sample negative interactions to balance the dataset
    positive_count = len(interactions)
    negative_interactions = []

    import random
    random.seed(42)

    while len(negative_interactions) < positive_count:
        user_idx = random.randint(0, len(unique_users) - 1)
        question_idx = random.randint(0, len(unique_questions) - 1)

        # Check if this combination doesn't exist in positive interactions
        if [user_idx, question_idx, 1] not in interactions:
            negative_interactions.append([user_idx, question_idx, 0])

    interactions.extend(negative_interactions)
    interactions_df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'label'])

    # Get all unique topics for vocabulary
    all_topics = set()
    for tags in user_tags.values():
        all_topics.update(tags)
    for tags in question_tags.values():
        all_topics.update(tags)

    topic_vocab = list(all_topics)

    print(f"Generated {len(interactions_df)} interactions")
    print(f"Positive interactions: {sum(interactions_df['label'])}")
    print(f"Negative interactions: {len(interactions_df) - sum(interactions_df['label'])}")
    print(f"Number of users: {len(unique_users)}")
    print(f"Number of items (questions): {len(unique_questions)}")
    print(f"Topic vocabulary size: {len(topic_vocab)}")

    return {
        'user_topics': user_topic_lists,
        'item_topics': item_topic_lists,
        'interactions': interactions_df,
        'n_users': len(unique_users),
        'n_items': len(unique_questions),
        'topic_vocab': topic_vocab
    }


def main():
    extract_data()

if __name__ == "__main__":
    main()
