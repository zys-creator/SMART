from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
class QwenTool:
    def __init__(self, args, model_path="/home/zys/Qwen72B", device="cuda:0"):
        """
        args: 对象，需要包含 temperature 等属性
        model_path: LLaMA 模型路径
        device: 指定 GPU，如 "cuda:0" 或 "cpu"
        """
        self.args = args
        self.model_path = model_path
        self.device = device
        self.model_name = "Qwen"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        # print(self.model.generation_config)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # print(self.tokenizer.chat_template)

    def generate(self, prompt, system_instruction="You are a helpful AI bot."):
        messages = [
            {"role": "system", "content": system_instruction},
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
            max_new_tokens=8192,
            temperature=0.1
            # top_k=1,
            # top_p=1.0,
            # repetition_penalty=1.05
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
