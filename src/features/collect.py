import time
from src.features.game_complexity import calculate_complexity_scores


def get_timestamp():
    return time.time()


def get_game_complexity(game_id):
    return calculate_complexity_scores(game_id)
