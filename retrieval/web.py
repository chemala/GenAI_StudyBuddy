from tavily import TavilyClient

def tavily_search(query, api_key, k=3):
    client = TavilyClient(api_key=api_key)
    res = client.search(query, limit=k)
    return [r.get("content", "") for r in res["results"]]