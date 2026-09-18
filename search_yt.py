import yt_dlp


def search_youtube(query: str, max_results: int = 10) -> list[dict]:
    options = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
    }

    search_query = f"ytsearch{max_results}:{query}"

    with yt_dlp.YoutubeDL(options) as ydl:
        result = ydl.extract_info(search_query, download=False)

    videos = []

    for entry in result.get("entries", []):
        if not entry:
            continue

        videos.append({
            "id": entry.get("id"),
            "title": entry.get("title"),
            "url": entry.get("url"),
            "channel": entry.get("channel"),
            "duration": entry.get("duration"),
            "view_count": entry.get("view_count"),
        })

    return videos