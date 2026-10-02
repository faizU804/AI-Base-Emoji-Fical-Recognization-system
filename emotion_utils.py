EMOTION_DATA = {
    "happy": {
        "emoji": "😊",
        "name": "HAPPY"
    },
    "sad": {
        "emoji": "😢",
        "name": "SAD"
    },
    "angry": {
        "emoji": "😡",
        "name": "ANGRY"
    },
    "surprise": {
        "emoji": "😮",
        "name": "SURPRISE"
    },
    "fear": {
        "emoji": "😨",
        "name": "FEAR"
    },
    "disgust": {
        "emoji": "🤢",
        "name": "DISGUST"
    },
    "neutral": {
        "emoji": "😐",
        "name": "NEUTRAL"
    }
}


def get_emoji(emotion):
    return EMOTION_DATA.get(
        emotion.lower(),
        {"emoji": "🙂"}
    )["emoji"]


def get_emotion_name(emotion):
    return EMOTION_DATA.get(
        emotion.lower(),
        {"name": emotion.upper()}
    )["name"]