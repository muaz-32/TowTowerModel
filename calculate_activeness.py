import csv
import json
from collections import defaultdict

ROOT_DIR = '.'
questions_input_file = f"{ROOT_DIR}/data/dump/users.questions.table.csv"
answers_input_file = f"{ROOT_DIR}/data/dump/users.answers.table.csv"
comments_input_file = f"{ROOT_DIR}/data/dump/users.comments.table.csv"

def get_activeness():
    user_dates = defaultdict(list)

    with open(questions_input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 4:
                continue
            user_id = row[0]
            date = row[3]
            if user_id and date:
                user_dates[user_id].append(date)

    with open(answers_input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 5:
                continue
            user_id = row[0]
            date = row[4]
            if user_id and date:
                user_dates[user_id].append(date)

    with open(comments_input_file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # Skip header
        for row in reader:
            if len(row) < 6:
                continue
            user_id = row[0]
            date = row[5]
            if user_id and date:
                user_dates[user_id].append(date)

    return user_dates


import pandas as pd
from datetime import datetime, timedelta
import numpy as np


def extract_activeness_features(user_datetimes: dict[int, str]):
    """
    Extract user activeness features from datetime data

    Args:
        user_datetimes: List of datetime strings in format YYYY-MM-DDTHH:MM:SS.SSS

    Returns:
        Dictionary of activeness features
    """
    if not user_datetimes:
        return {
            'total_activities': 0,
            'days_active': 0,
            'avg_daily_activity': 0,
            'activity_variance': 0,
            'most_active_hour': 0,
            'weekend_activity_ratio': 0,
            'recent_activity_days': 0
        }

    # Convert strings to datetime objects
    datetimes = [datetime.fromisoformat(dt.replace('T', ' ')) for dt in user_datetimes]
    datetimes.sort()

    # Basic activity metrics
    total_activities = len(datetimes)

    # Activity span and frequency
    date_range = (datetimes[-1] - datetimes[0]).days + 1
    unique_days = len(set(dt.date() for dt in datetimes))
    avg_daily_activity = total_activities / max(date_range, 1)

    # Activity distribution by day
    daily_counts = {}
    for dt in datetimes:
        day = dt.date()
        daily_counts[day] = daily_counts.get(day, 0) + 1

    activity_variance = np.var(list(daily_counts.values())) if daily_counts else 0

    # Time patterns
    hours = [dt.hour for dt in datetimes]
    most_active_hour = max(set(hours), key=hours.count) if hours else 0

    # Weekend activity
    weekend_activities = sum(1 for dt in datetimes if dt.weekday() >= 5)
    weekend_activity_ratio = weekend_activities / total_activities if total_activities > 0 else 0

    # Recent activity (last 30 days)
    recent_cutoff = datetime.now() - timedelta(days=30)
    recent_activities = sum(1 for dt in datetimes if dt >= recent_cutoff)
    recent_activity_days = len(set(dt.date() for dt in datetimes if dt >= recent_cutoff))

    return {
        'total_activities': total_activities,
        'days_active': unique_days,
        'avg_daily_activity': avg_daily_activity,
        'activity_variance': activity_variance,
        'most_active_hour': most_active_hour,
        'weekend_activity_ratio': weekend_activity_ratio,
        'recent_activity_days': recent_activity_days
    }


def append_activeness_features_to_embeddings(user_topic_embeddings: np.ndarray):
    """
    Append activeness features to user topic embeddings

    Args:
        user_topic_embeddings: numpy array with user embeddings

    Returns:
        Updated numpy array with activeness features
    """
    activeness_features = []
    user_datetime_data = get_activeness()

    for user_id in range(user_topic_embeddings.shape[0]):
        user_datetimes = user_datetime_data.get(str(user_id), [])  # Convert to string for lookup
        features = extract_activeness_features(user_datetimes)
        activeness_features.append(list(features.values()))

    # Convert to numpy array
    activeness_array = np.array(activeness_features)

    # Concatenate with existing embeddings
    enhanced_embeddings = np.concatenate([user_topic_embeddings, activeness_array], axis=1)

    return enhanced_embeddings
