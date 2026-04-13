import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class RAGEvaluator:
    """
    RAG 自動化評估模組
    功能：整合語意相似度與知識忠實度分析，量化 RAG 輸出品質。
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # 採用輕量化嵌入模型進行語意計算，符合成本效能平衡策略
        self.embed_model = SentenceTransformer(model_name)

    def get_embedding(self, text):
        return self.embed_model.encode([text])

    def calculate_semantic_similarity(self, prediction, reference):
        """
        [指標 1: Semantic Similarity]
        評估 AI 答案與標準答案之間的語意接近程度。
        """
        emb1 = self.get_embedding(prediction)
        emb2 = self.get_embedding(reference)
        score = cosine_similarity(emb1, emb2)[0][0]
        return round(float(score), 4)

    def calculate_faithfulness(self, prediction, context):
        """
        [指標 2: Faithfulness / Groundedness]
        評估 AI 答案是否忠實於檢索到的原始資料，用於監控「幻覺」現象。
        邏輯：計算生成答案與檢索上下文的語意重合度。
        """
        emb_pred = self.get_embedding(prediction)
        emb_ctx = self.get_embedding(context)
        score = cosine_similarity(emb_pred, emb_ctx)[0][0]
        return round(float(score), 4)

    def run_suite(self, prediction, reference, context):
        """
        執行全量評測套件
        """
        semantic_score = self.calculate_semantic_similarity(prediction, reference)
        faithfulness_score = self.calculate_faithfulness(prediction, context)
        
        # 整體得分計算（可根據 PM 策略加權調整）
        overall_score = round((semantic_score + faithfulness_score) / 2, 4)
        
        return {
            "semantic_similarity": semantic_score,
            "faithfulness": faithfulness_score,
            "overall_score": overall_score
        }

# --- 範例測試 (用於驗證 V6 版本邏輯) ---
if __name__ == "__main__":
    evaluator = RAGEvaluator()
    
    # 模擬數據
    sample_context = "根據 2024 資安報告，SQL 注入攻擊比例上升了 15%。"
    sample_ref = "SQL 注入攻擊上升了 15%。"
    sample_pred = "報告指出 SQL 注入有 15% 的成長。"
    
    results = evaluator.run_suite(sample_pred, sample_ref, sample_context)
    print(f"V6 評測結果: {results}")
