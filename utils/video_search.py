def get_video(topic):

    query = topic.replace(" ", "+")

    videos = [
        f"https://www.youtube.com/results?search_query={query}+tutorial",
        f"https://www.youtube.com/results?search_query={query}+explained",
        f"https://www.youtube.com/results?search_query={query}+lecture"
    ]

    return videos