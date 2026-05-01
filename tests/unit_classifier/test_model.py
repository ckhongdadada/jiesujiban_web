from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch


class TestBertCNNAttentionModel(unittest.TestCase):
    """测试BertCNNAttention模型"""

    def test_model_initialization(self):
        """测试模型初始化"""
        from src.jsjb.unit_classifier.model import BertCNNAttention, HybridClassifierConfig
        
        config = HybridClassifierConfig(
            model_name="bert-base-chinese",
            num_classes=32,
            use_tfidf=False,
        )
        
        with patch("src.jsjb.unit_classifier.model.AutoModel.from_pretrained") as mock_bert:
            mock_bert.return_value.config.hidden_size = 768
            model = BertCNNAttention(config)
            
            # 验证模型结构
            self.assertIsNotNone(model.bert)
            self.assertEqual(len(model.convs), 3)  # [2, 3, 4] filter sizes
            self.assertIsNotNone(model.attention)
            self.assertIsNone(model.tfidf_net)  # use_tfidf=False

    def test_model_with_tfidf(self):
        """测试带TF-IDF的模型"""
        from src.jsjb.unit_classifier.model import BertCNNAttention, HybridClassifierConfig
        
        config = HybridClassifierConfig(
            model_name="bert-base-chinese",
            num_classes=32,
            use_tfidf=True,
            tfidf_dim=3000,
            tfidf_hidden=64,
        )
        
        with patch("src.jsjb.unit_classifier.model.AutoModel.from_pretrained") as mock_bert:
            mock_bert.return_value.config.hidden_size = 768
            model = BertCNNAttention(config)
            
            self.assertIsNotNone(model.tfidf_net)

    def test_conv_and_pool(self):
        """测试卷积和池化"""
        import torch
        from src.jsjb.unit_classifier.model import BertCNNAttention, HybridClassifierConfig
        
        config = HybridClassifierConfig(
            model_name="bert-base-chinese",
            num_classes=32,
            use_tfidf=False,
        )
        
        with patch("src.jsjb.unit_classifier.model.AutoModel.from_pretrained") as mock_bert:
            mock_bert.return_value.config.hidden_size = 768
            model = BertCNNAttention(config)
            
            # 测试conv_and_pool静态方法
            x = torch.randn(2, 768, 128)  # batch_size, hidden_size, seq_len
            conv = model.convs[0]
            result = BertCNNAttention.conv_and_pool(x, conv)
            
            self.assertEqual(result.shape, (2, 256))  # batch_size, num_filters


class TestHybridClassifierConfig(unittest.TestCase):
    """测试混合分类器配置"""

    def test_config_defaults(self):
        """测试配置默认值"""
        from src.jsjb.unit_classifier.model import HybridClassifierConfig
        
        config = HybridClassifierConfig(
            model_name="bert-base-chinese",
            num_classes=32,
        )
        
        self.assertEqual(config.use_tfidf, False)
        self.assertEqual(config.tfidf_dim, 3000)
        self.assertEqual(config.tfidf_hidden, 64)
        self.assertEqual(config.num_filters, 256)

    def test_config_custom_values(self):
        """测试自定义配置值"""
        from src.jsjb.unit_classifier.model import HybridClassifierConfig
        
        config = HybridClassifierConfig(
            model_name="bert-base-chinese",
            num_classes=64,
            use_tfidf=True,
            tfidf_dim=5000,
            tfidf_hidden=128,
            num_filters=512,
        )
        
        self.assertEqual(config.num_classes, 64)
        self.assertTrue(config.use_tfidf)
        self.assertEqual(config.tfidf_dim, 5000)
        self.assertEqual(config.tfidf_hidden, 128)
        self.assertEqual(config.num_filters, 512)


if __name__ == "__main__":
    unittest.main()