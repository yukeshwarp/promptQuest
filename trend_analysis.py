from cloud_config import llmclient

def analyze_trends(processed_chat_titles):
    return llmclient.chat.completions.create(
                            model="gpt-4.1",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert data analyst analyzing trends from user interaction data.",
                                },
                                {
                                    "role": "user",
                                    "content": f"""
                                    Analyze the following chat titles for trends, topics, and insights based on user interactions. 
                                    Provide a summary of key trends and observations.
                                    Return only the text.
                                    
                                    Chat Titles:
                                    {processed_chat_titles}
                                """,
                                },
                            ],
                            temperature=0.7,
                            stream=False,  # We want a complete response, not a stream
                        )