# Requires transformers>=4.51.0

from transformers import AutoTokenizer, AutoModel
import torch
from torch import Tensor


class Qwen3Embeddings:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 设置为评估模式
        self.max_length = 8192
        if torch.cuda.is_available():
            self.model = self.model.cuda()  # 如果有 GPU，加载到 GPU 上

        self.TASK = "Given a search query related to the field of oral medicine and dentistry, retrieve relevant knowledge that answers the query. "
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery:{query}'

    def embed(self, texts: list) -> torch.Tensor:
        """
        生成文本的嵌入。
        Args:
            texts (list): 文本列表。
        Returns:
            torch.Tensor: 嵌入向量。
        """
        texts = [self.get_detailed_instruct(self.TASK, text) for text in texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length  # 根据模型的最大长度调整
        )
        inputs.to(self.model.device)

        with torch.no_grad():  # 禁用梯度计算
            outputs = self.model(**inputs)
            # embeddings = outputs.last_hidden_state.mean(dim=1)  # 平均池化生成句子嵌入
            embeddings = self.last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])

        return embeddings
    
    def embed_documents(self, documents: list) -> list:
        """
        生成文档的嵌入。
        Args:
            documents (list): 包含 str 对象的列表。
        Returns:
            list: 每个文档的嵌入向量（列表形式）。
        """
        # 提取文档的主要内容（如 page_content）
        embeddings = self.embed(documents)  # 调用 embed 方法生成嵌入
        return embeddings.cpu().tolist()  # 转换为列表形式返回

    def embed_query(self, query: str) -> list:
        """
        生成查询的嵌入。
        Args:
            query (str): 查询文本。
        Returns:
            list: 查询的嵌入向量（列表形式）。
        """
        embeddings = self.embed([query])
        return embeddings[0].cpu().tolist()

    def last_token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

