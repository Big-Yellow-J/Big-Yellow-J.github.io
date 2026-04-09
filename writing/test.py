from openai import OpenAI
client = OpenAI(
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key="",
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
                "net_worth": {"type": "string",
                              },
                "occupations": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["name", "age", "world_wealth_rank"]
        }
    }
}
completion = client.chat.completions.create(
    model="doubao-seed-2-0-lite-260215",
    messages = [
        {"role": "user", "content": "分析一个特朗普的生平，如年龄、世界富豪排名，按照json格式输出"},
    ],
    response_format=json_schema,
    extra_body={
        "guided_json": json_schema,
        "guided_decoding_backend": "xgrammar",
        "temperature": 0.3
    }
)
print(completion.choices[0].message.content)

# import torch
# from torchviz import make_dot
# from torchvision.models import resnet18
# """
# apt-get install -y graphviz
# pip install torchviz
# """
# model = resnet18()
# model.eval()
# x = torch.randn(1, 3, 224, 224)
# y = model(x)

# # dot = make_dot(y, params=dict(model.named_parameters()))
# # dot.format = 'png'
# # dot.render("model_graph")

# onnx_file_path = "resnet18_model.onnx"
# torch.onnx.export(
#     model,                      # 要导出的模型
#     x,                          # 虚拟输入
#     onnx_file_path,             # 导出文件路径
#     export_params=True,         # 是否导出权重参数
#     opset_version=17,           # ONNX 算子版本，建议 12 及以上
#     do_constant_folding=True,   # 是否执行常量折叠优化
#     input_names=['input'],      # 输入节点的名称
#     output_names=['output'],    # 输出节点的名称
#     dynamic_axes={              # 可选：支持动态 Batch Size
#         'input': {0: 'batch_size'},
#         'output': {0: 'batch_size'}
#     }
# )
# print(f"模型已成功导出至: {onnx_file_path}")