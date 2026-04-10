from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Qwen3Reranker:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B"):
        """
        初始化 Qwen3 Reranker 模型。
        Args:
            model_name (str): Qwen3 reranker 模型名称。
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_name).eval()
        # We recommend enabling flash_attention_2 for better acceleration and memory saving.
        # model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
        
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.TASK = "Given a search query related to the field of oral medicine and dentistry, retrieve relevant knowledge that answers the query. "

        if torch.cuda.is_available():
            self.model = self.model.cuda()  # 如果有 GPU，加载到 GPU 上
            

    def rerank(self, query: list, documents: list) -> list:
        """
        对候选项进行 reranking。
        Args:
            query (list): 查询文本。
            documents (list): 候选文本列表。
        Returns:
            list: 包含 (candidate, score) 的元组列表，按分数降序排序。
        """

        pairs = [
                self.format_instruction(self.TASK, query, doc) 
                    for doc in documents
                 ]

        inputs = self.process_inputs(pairs)
        scores = self.compute_logits(inputs)

        # Print scores for debugging
        print("Scores:", scores)

        # Pair scores with document indices
        scored_documents = [
            {"index": idx, "score": score}
            for idx, score in enumerate(scores)
        ]

        # Sort by score in descending order
        scored_documents.sort(key=lambda x: x["score"], reverse=True)

        return scored_documents


    @torch.no_grad()
    def compute_logits(self, inputs, **kwargs):
        batch_scores = self.model(**inputs).logits[:, -1, :]
        
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    def process_inputs(self, pairs):
        prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    def format_instruction(self, instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
        return output