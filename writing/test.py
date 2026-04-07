from openai import OpenAI
import os

client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="636b7b0b-2",
)

json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "trump_analysis",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "birth_date": {"type": "string"},
                "world_wealth_rank": {"type": "integer"},
                "net_worth": {"type": "string"},
                "occupations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age", "world_wealth_rank"]
        }
    }
}
completion = client.chat.completions.create(
    model="doubao-seed-1-6-251015",
    messages = [
        {"role": "user", "content": "分析一个特朗普的生平，如年龄、世界富豪排名，按照json格式输出"},
    ],
    # response_format={"type": "json_object"}
    # response_format={"type": "text"}
    response_format=json_schema
)
print(completion.choices[0].message.content)