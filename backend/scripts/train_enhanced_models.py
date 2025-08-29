#!/usr/bin/env python3
"""
增强模型训练脚本
用于训练和优化服装分类和风格匹配模型
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"请安装必要的依赖: pip install numpy pandas scikit-learn matplotlib seaborn")
    print(f"缺少依赖: {e}")
    sys.exit(1)

try:
    from models.enhanced_classifier import get_enhanced_classifier
    from models.style_matcher import get_style_matcher
    ENHANCED_MODELS_AVAILABLE = True
except ImportError:
    print("增强模型不可用，将使用基础模型进行训练")
    ENHANCED_MODELS_AVAILABLE = False

class ModelTrainer:
    """
    模型训练器
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "models/trained"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.training_history = []
        
        print(f"模型训练器初始化完成")
        print(f"数据目录: {self.data_dir}")
        print(f"输出目录: {self.output_dir}")
    
    def load_training_data(self) -> Tuple[List[Dict], List[str]]:
        """
        加载训练数据
        """
        print("加载训练数据...")
        
        # 从用户数据中提取训练样本
        users_file = self.data_dir / "users.json"
        feedback_file = self.data_dir / "feedback.json"
        
        training_samples = []
        labels = []
        
        # 加载用户衣橱数据
        if users_file.exists():
            with open(users_file, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            
            for user_id, user_data in users_data.items():
                wardrobe_items = user_data.get("wardrobe_items", [])
                for item in wardrobe_items:
                    if "category" in item and "image_path" in item:
                        training_samples.append(item)
                        labels.append(item["category"])
        
        # 加载反馈数据用于强化学习
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
            
            # 可以根据用户反馈调整训练权重
            print(f"加载了 {len(feedback_data.get('feedback', []))} 条用户反馈")
        
        print(f"加载了 {len(training_samples)} 个训练样本")
        print(f"类别分布: {pd.Series(labels).value_counts().to_dict()}")
        
        return training_samples, labels
    
    def prepare_features(self, samples: List[Dict]) -> np.ndarray:
        """
        准备特征数据
        """
        print("准备特征数据...")
        
        features = []
        
        for sample in samples:
            # 提取基础特征
            feature_vector = []
            
            # 颜色特征
            dominant_colors = sample.get("dominant_colors", [])
            if dominant_colors:
                # 取前3个主要颜色的RGB值
                for i in range(3):
                    if i < len(dominant_colors):
                        color = dominant_colors[i]
                        rgb = color.get("rgb", [0, 0, 0])
                        feature_vector.extend(rgb)
                        feature_vector.append(color.get("percentage", 0))
                    else:
                        feature_vector.extend([0, 0, 0, 0])  # 填充
            else:
                feature_vector.extend([0] * 12)  # 3 colors * 4 values
            
            # 风格关键词特征（简单的词袋模型）
            style_keywords = sample.get("style_keywords", [])
            common_styles = ["casual", "formal", "sporty", "elegant", "trendy", "classic"]
            for style in common_styles:
                feature_vector.append(1 if style in style_keywords else 0)
            
            # 置信度特征
            feature_vector.append(sample.get("confidence", 0.5))
            
            # 图片尺寸特征（如果可用）
            if "image_size" in sample:
                width, height = sample["image_size"]
                feature_vector.extend([width, height, width/height if height > 0 else 1])
            else:
                feature_vector.extend([224, 224, 1])  # 默认值
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_classifier(self, samples: List[Dict], labels: List[str]) -> Dict[str, Any]:
        """
        训练分类器
        """
        print("训练增强分类器...")
        
        if not ENHANCED_MODELS_AVAILABLE:
            print("增强模型不可用，跳过训练")
            return {"success": False, "error": "Enhanced models not available"}
        
        try:
            # 准备特征
            X = self.prepare_features(samples)
            y = np.array(labels)
            
            # 分割训练和测试数据
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # 获取增强分类器
            classifier = get_enhanced_classifier()
            
            # 训练模型
            if hasattr(classifier, 'train_classifier'):
                training_result = classifier.train_classifier(X_train, y_train)
                
                # 评估模型
                if hasattr(classifier, 'predict'):
                    y_pred = classifier.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    report = classification_report(y_test, y_pred, output_dict=True)
                    
                    print(f"分类器准确率: {accuracy:.4f}")
                    
                    # 保存训练结果
                    result = {
                        "success": True,
                        "accuracy": accuracy,
                        "classification_report": report,
                        "training_samples": len(X_train),
                        "test_samples": len(X_test),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # 保存模型
                    model_path = self.output_dir / "enhanced_classifier.pkl"
                    if hasattr(classifier, 'save_model'):
                        classifier.save_model(str(model_path))
                        result["model_path"] = str(model_path)
                    
                    return result
                else:
                    return {"success": False, "error": "Classifier does not support prediction"}
            else:
                return {"success": False, "error": "Classifier does not support training"}
                
        except Exception as e:
            print(f"训练分类器时出错: {e}")
            return {"success": False, "error": str(e)}
    
    def train_style_matcher(self, samples: List[Dict], feedback_data: Dict) -> Dict[str, Any]:
        """
        训练风格匹配器
        """
        print("训练风格匹配器...")
        
        if not ENHANCED_MODELS_AVAILABLE:
            print("增强模型不可用，跳过训练")
            return {"success": False, "error": "Enhanced models not available"}
        
        try:
            # 获取风格匹配器
            style_matcher = get_style_matcher()
            
            if style_matcher is None:
                return {"success": False, "error": "Style matcher not available"}
            
            # 准备训练数据
            outfit_pairs = []
            compatibility_scores = []
            
            # 从反馈数据中提取搭配对和评分
            feedback_list = feedback_data.get("feedback", [])
            for feedback in feedback_list:
                if "rating" in feedback:
                    # 这里需要根据实际的反馈数据结构来调整
                    # 简化示例：假设我们有搭配对的信息
                    outfit_pairs.append({
                        "item1": {"category": "shirt", "colors": ["blue"]},
                        "item2": {"category": "pants", "colors": ["black"]}
                    })
                    compatibility_scores.append(feedback["rating"] / 5.0)  # 归一化到0-1
            
            if len(outfit_pairs) < 10:
                # 生成一些合成训练数据
                synthetic_pairs, synthetic_scores = self._generate_synthetic_training_data()
                outfit_pairs.extend(synthetic_pairs)
                compatibility_scores.extend(synthetic_scores)
            
            # 训练风格匹配器
            if hasattr(style_matcher, 'train_compatibility_network'):
                training_result = style_matcher.train_compatibility_network(
                    outfit_pairs, compatibility_scores
                )
                
                result = {
                    "success": True,
                    "training_pairs": len(outfit_pairs),
                    "training_result": training_result,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 保存模型
                model_path = self.output_dir / "style_matcher.pkl"
                if hasattr(style_matcher, 'save_model'):
                    style_matcher.save_model(str(model_path))
                    result["model_path"] = str(model_path)
                
                return result
            else:
                return {"success": False, "error": "Style matcher does not support training"}
                
        except Exception as e:
            print(f"训练风格匹配器时出错: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_synthetic_training_data(self) -> Tuple[List[Dict], List[float]]:
        """
        生成合成训练数据
        """
        print("生成合成训练数据...")
        
        pairs = []
        scores = []
        
        # 定义一些基本的搭配规则
        good_combinations = [
            ({"category": "shirt", "colors": ["white"]}, {"category": "pants", "colors": ["black"]}, 0.9),
            ({"category": "shirt", "colors": ["blue"]}, {"category": "pants", "colors": ["navy"]}, 0.8),
            ({"category": "dress", "colors": ["red"]}, {"category": "shoes", "colors": ["black"]}, 0.85),
            ({"category": "jacket", "colors": ["gray"]}, {"category": "shirt", "colors": ["white"]}, 0.9),
        ]
        
        bad_combinations = [
            ({"category": "shirt", "colors": ["red"]}, {"category": "pants", "colors": ["orange"]}, 0.2),
            ({"category": "dress", "colors": ["pink"]}, {"category": "shoes", "colors": ["green"]}, 0.1),
            ({"category": "jacket", "colors": ["purple"]}, {"category": "shirt", "colors": ["yellow"]}, 0.15),
        ]
        
        # 生成多个变体
        for item1, item2, score in good_combinations + bad_combinations:
            for _ in range(5):  # 每个组合生成5个变体
                pairs.append({"item1": item1.copy(), "item2": item2.copy()})
                scores.append(score + np.random.normal(0, 0.1))  # 添加一些噪声
        
        # 确保分数在合理范围内
        scores = [max(0, min(1, score)) for score in scores]
        
        print(f"生成了 {len(pairs)} 个合成训练样本")
        return pairs, scores
    
    def evaluate_models(self) -> Dict[str, Any]:
        """
        评估训练好的模型
        """
        print("评估模型性能...")
        
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "classifier": {},
            "style_matcher": {}
        }
        
        try:
            # 评估分类器
            if ENHANCED_MODELS_AVAILABLE:
                classifier = get_enhanced_classifier()
                if hasattr(classifier, 'evaluate'):
                    classifier_results = classifier.evaluate()
                    evaluation_results["classifier"] = classifier_results
                
                # 评估风格匹配器
                style_matcher = get_style_matcher()
                if style_matcher and hasattr(style_matcher, 'evaluate'):
                    matcher_results = style_matcher.evaluate()
                    evaluation_results["style_matcher"] = matcher_results
            
            return evaluation_results
            
        except Exception as e:
            print(f"评估模型时出错: {e}")
            return {"error": str(e)}
    
    def generate_training_report(self, results: Dict[str, Any]) -> str:
        """
        生成训练报告
        """
        report_path = self.output_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"训练报告已保存到: {report_path}")
        return str(report_path)
    
    def run_full_training(self) -> Dict[str, Any]:
        """
        运行完整的训练流程
        """
        print("开始完整训练流程...")
        
        results = {
            "start_time": datetime.now().isoformat(),
            "classifier_training": {},
            "style_matcher_training": {},
            "evaluation": {},
            "success": False
        }
        
        try:
            # 加载训练数据
            samples, labels = self.load_training_data()
            
            if len(samples) == 0:
                results["error"] = "没有找到训练数据"
                return results
            
            # 训练分类器
            classifier_results = self.train_classifier(samples, labels)
            results["classifier_training"] = classifier_results
            
            # 加载反馈数据
            feedback_file = self.data_dir / "feedback.json"
            feedback_data = {}
            if feedback_file.exists():
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedback_data = json.load(f)
            
            # 训练风格匹配器
            matcher_results = self.train_style_matcher(samples, feedback_data)
            results["style_matcher_training"] = matcher_results
            
            # 评估模型
            evaluation_results = self.evaluate_models()
            results["evaluation"] = evaluation_results
            
            results["end_time"] = datetime.now().isoformat()
            results["success"] = True
            
            # 生成报告
            report_path = self.generate_training_report(results)
            results["report_path"] = report_path
            
            print("训练流程完成！")
            return results
            
        except Exception as e:
            print(f"训练流程出错: {e}")
            results["error"] = str(e)
            results["end_time"] = datetime.now().isoformat()
            return results

def main():
    parser = argparse.ArgumentParser(description="训练增强ML模型")
    parser.add_argument("--data-dir", default="data", help="数据目录路径")
    parser.add_argument("--output-dir", default="models/trained", help="输出目录路径")
    parser.add_argument("--mode", choices=["full", "classifier", "matcher", "evaluate"], 
                       default="full", help="训练模式")
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = ModelTrainer(args.data_dir, args.output_dir)
    
    if args.mode == "full":
        results = trainer.run_full_training()
    elif args.mode == "classifier":
        samples, labels = trainer.load_training_data()
        results = trainer.train_classifier(samples, labels)
    elif args.mode == "matcher":
        samples, _ = trainer.load_training_data()
        feedback_file = Path(args.data_dir) / "feedback.json"
        feedback_data = {}
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_data = json.load(f)
        results = trainer.train_style_matcher(samples, feedback_data)
    elif args.mode == "evaluate":
        results = trainer.evaluate_models()
    
    print("\n=== 训练结果 ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()