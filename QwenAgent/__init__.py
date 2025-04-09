from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch, json, os
from openai import OpenAI
from pathlib import Path


class QwenModel(torch.nn.Module):
    def __init__(self, model_path=None, api_key=None, model_id="qwen-max"):
        super().__init__()
        if model_path is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.model = None
            self.tokenizer = None
        self.api_key = api_key
        self.model_id = model_id

    def generate_local(self, system_prompt, prompt, *args, **kwargs):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=9999999
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def generate_api(self, system_prompt, prompt, *args, **kwargs):
        client = OpenAI(
            api_key=self.api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ],
        )
        data = completion.model_dump_json()
        data = json.loads(data)
        data = data["choices"][0]["message"]["content"]
        return data
    
    def generate_api_deepseek(self, system_prompt, prompt, *args, **kwargs):
        client = OpenAI(
            api_key=self.api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        reasoning_content = ""  # 定义完整思考过程
        answer_content = ""     # 定义完整回复
        is_answering = False   # 判断是否结束思考过程并开始回复

        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称
            messages=[
                {"role": "user", "content": system_prompt + prompt}
            ],
            stream=True,
            # 解除以下注释会在最后一个chunk返回Token使用量
            # stream_options={
            #     "include_usage": True
            # }
        )

        print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

        for chunk in completion:
            # 如果chunk.choices为空，则打印usage
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                # 打印思考过程
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    print(delta.reasoning_content, end='', flush=True)
                    reasoning_content += delta.reasoning_content
                else:
                    # 开始回复
                    if delta.content != "" and is_answering == False:
                        print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                        is_answering = True
                    # 打印回复过程
                    print(delta.content, end='', flush=True)
                    answer_content += delta.content
        return answer_content
    
    def generate_api_stream(self, system_prompt, prompt, doc_path):
        client = OpenAI(
            api_key=self.api_key, 
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        answer_content = ""     # 定义完整回复

        messages = []
        if os.path.exists(doc_path) and doc_path.endswith(".pdf"):
            print("Got pdf file.")
            file_object = client.files.create(file=Path(doc_path), purpose="file-extract")
            messages.append({'role': 'system', 'content': f'fileid://{file_object.id}'})
        if len(system_prompt) > 1:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        # 创建聊天完成请求
        completion = client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True}
        )
        for chunk in completion:
            if not chunk.choices:
                print("\nUsage:")
                print(chunk.usage)
            else:
                delta = chunk.choices[0].delta
                print(delta.content, end='', flush=True)
                answer_content += delta.content
        return answer_content
    
    def generate(self, system_prompt, prompt, doc_path):
        if self.api_key is not None and self.model_id == "deepseek-r1":
            return self.generate_api_deepseek(system_prompt, prompt, doc_path)
        elif self.api_key is not None:
            return self.generate_api_stream(system_prompt, prompt, doc_path)
        else:
            return self.generate_local(system_prompt, prompt, doc_path)


class QwenModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "model_path": ("STRING", {"default": ""}),
                "api_key": ("STRING", {"default": ""}),
                "model_id": ("STRING", {"default": "qwen-max"}),
            }
        }

    RETURN_TYPES = ("QWENMODEL",)
    RETURN_NAMES = ("model", )
    FUNCTION = "loadmodel"
    CATEGORY = "PaperWriter"

    def loadmodel(self, model_path=None, api_key=None, model_id=None):
        if len(model_path) == 0: model_path = None
        if len(api_key) == 0: api_key = None
        model = QwenModel(model_path, api_key, model_id)
        return (model,)
    

class QwenAgent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("QWENMODEL",),
                "system_prompt": ("STRING", {"multiline": True, "default": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},),
                "prompt": ("PROMPT",),
                "doc_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "generate"
    CATEGORY = "PaperWriter"

    def generate(self, model, system_prompt, prompt, doc_path):
        return (model.generate(system_prompt, prompt, doc_path),)


class TextInput:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""},),
            },
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("result",)
    FUNCTION = "generate"
    CATEGORY = "PaperWriter"

    def generate(self, text):
        return (text,)


class TextReader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "path": ("STRING", {"default": ""},),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "load"
    CATEGORY = "PaperWriter"

    def load(self, path):
        with open(path, "r", encoding="utf-8-sig") as f:
            text = f.read()
        return (text,)
    

class TextFormater:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("PROMPT",),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "format"
    CATEGORY = "PaperWriter"

    def format(self, text):
        text = text.strip()
        if text.startswith("```latex"):
            text = text[len("```latex"):]
        if text.endswith("```"):
            text = text[:-len("```")]
        return (text,)
    

class TextMerger:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "text_1": ("PROMPT", {"default": ""}),
                "text_2": ("PROMPT", {"default": ""}),
                "text_3": ("PROMPT", {"default": ""}),
                "text_4": ("PROMPT", {"default": ""}),
                "text_5": ("PROMPT", {"default": ""}),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "merge"
    CATEGORY = "PaperWriter"

    def merge(self, text_1="", text_2="", text_3="", text_4="", text_5=""):
        text_list = [text_1, text_2, text_3, text_4, text_5]
        return ("\n\n".join(text_list),)


class TextWriter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("PROMPT",),
                "path": ("STRING", {"default": ""},),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save"
    CATEGORY = "PaperWriter"
    OUTPUT_NODE = True

    def save(self, text, path):
        with open(path, "w", encoding="utf-8-sig") as f:
            f.write(text)
        return text


NODE_CLASS_MAPPINGS = {
    "QwenModelLoader": QwenModelLoader,
    "QwenAgent": QwenAgent,
    "TextReader": TextReader,
    "TextWriter": TextWriter,
    "TextFormater": TextFormater,
    "TextMerger": TextMerger,
    "TextInput": TextInput,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenModelLoader": "Qwen Model Loader",
    "QwenAgent": "Qwen Agent",
    "TextReader": "TextReader",
    "TextWriter": "TextWriter",
    "TextFormater": "TextFormater",
    "TextMerger": "TextMerger",
    "TextInput": "TextInput",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

