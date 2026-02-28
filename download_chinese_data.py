"""下载中文训练数据集"""
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_wikipedia_zh():
    """下载中文维基百科数据"""
    print("📥 下载中文维基百科数据...")
    
    # 创建data目录
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    output_file = data_dir / "zh_wiki.txt"
    
    try:
        # 加载中文维基百科数据集
        dataset = load_dataset("wikipedia", "20220301.zh", split="train")
        
        print(f"✓ 数据集大小: {len(dataset)} 篇文章")
        print(f"📝 保存到: {output_file}")
        
        # 写入文件
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc="处理文章"):
                text = item["text"].strip()
                if len(text) > 100:  # 过滤太短的文本
                    f.write(text + "\n\n")
        
        # 统计信息
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        with open(output_file, "r", encoding="utf-8") as f:
            lines = sum(1 for _ in f)
        
        print(f"\n✅ 下载完成!")
        print(f"   文件: {output_file}")
        print(f"   大小: {file_size:.2f} MB")
        print(f"   行数: {lines:,}")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n尝试备用方案: 下载中文新闻数据...")
        return download_news_zh()


def download_news_zh():
    """下载中文新闻数据（备用方案）"""
    print("📥 下载中文新闻数据...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "zh_news.txt"
    
    try:
        # 使用THUCNews数据集
        dataset = load_dataset("thu-coai/THUCNews", split="train")
        
        print(f"✓ 数据集大小: {len(dataset)} 篇新闻")
        print(f"📝 保存到: {output_file}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for item in tqdm(dataset, desc="处理新闻"):
                text = item["content"].strip()
                if len(text) > 50:
                    f.write(text + "\n\n")
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✅ 下载完成!")
        print(f"   文件: {output_file}")
        print(f"   大小: {file_size:.2f} MB")
        
        return str(output_file)
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("\n尝试最后方案: 创建示例数据集...")
        return create_sample_corpus()


def create_sample_corpus():
    """创建扩展的示例语料库（最后备用方案）"""
    print("📝 创建扩展示例语料库...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "zh_sample.txt"
    
    # 扩展的中文示例文本
    sample_texts = [
        # 历史文化
        "中国是世界四大文明古国之一，拥有五千年悠久的历史。从夏商周到秦汉唐宋，历代王朝都留下了丰富的文化遗产。",
        "长城是中国古代最伟大的建筑之一，东起山海关，西至嘉峪关，全长超过两万公里，被誉为世界七大奇迹之一。",
        "故宫位于北京市中心，是明清两代的皇家宫殿，现在是世界上规模最大、保存最完整的木质结构古建筑群。",
        
        # 地理自然
        "中国地域辽阔，地形多样，有高山、平原、盆地、高原等多种地貌。从东部沿海到西部高原，呈现出三级阶梯状分布。",
        "长江是中国第一大河，全长6300多公里，流经11个省市，是中华民族的母亲河之一。",
        "青藏高原被称为世界屋脊，平均海拔超过4000米，是世界上海拔最高的高原。",
        
        # 科技发展
        "人工智能技术正在深刻改变人类社会，从医疗诊断到自动驾驶，从语音识别到机器翻译，AI应用无处不在。",
        "量子计算机利用量子力学原理进行信息处理，在某些特定问题上具有远超传统计算机的计算能力。",
        "5G移动通信技术提供了更高的数据传输速度和更低的延迟，为物联网、智慧城市等应用提供了基础设施。",
        
        # 经济社会
        "改革开放以来，中国经济持续快速发展，已成为世界第二大经济体，对全球经济增长的贡献率超过30%。",
        "教育是国家发展的基石，通过普及义务教育、发展高等教育，培养了大量人才，为经济社会发展提供智力支持。",
        "可持续发展理念强调经济发展、社会进步和环境保护的协调统一，是人类未来发展的必由之路。",
        
        # 文学艺术
        "中国古代文学源远流长，从《诗经》到唐诗宋词，从四大名著到现代文学，构成了璀璨的文学宝库。",
        "京剧是中国传统戏曲的代表，融合了唱、念、做、打等多种表演形式，被誉为国粹。",
        "中国书法是一门独特的艺术形式，通过毛笔、墨汁和宣纸的结合，表达书写者的情感和审美追求。",
        
        # 哲学思想
        "儒家思想强调仁、义、礼、智、信等道德准则，对中国社会产生了深远影响。",
        "道家哲学主张道法自然、无为而治，追求人与自然的和谐统一。",
        "佛教传入中国后，与本土文化融合，形成了独具特色的中国佛教文化。",
        
        # 现代生活
        "互联网的普及彻底改变了人们的生活方式，从购物、支付到社交、娱乐，都可以在线上完成。",
        "环境保护已成为全球共识，减少碳排放、保护生物多样性、推广清洁能源是应对气候变化的重要措施。",
        "健康饮食和适量运动是保持身体健康的关键，均衡的营养摄入和规律的锻炼能够提高生活质量。",
    ] * 50  # 重复50次，生成1000条文本
    
    with open(output_file, "w", encoding="utf-8") as f:
        for text in sample_texts:
            f.write(text + "\n")
    
    file_size = output_file.stat().st_size / 1024  # KB
    print(f"\n✅ 创建完成!")
    print(f"   文件: {output_file}")
    print(f"   大小: {file_size:.2f} KB")
    print(f"   条数: {len(sample_texts)}")
    
    return str(output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中文训练数据下载工具")
    parser.add_argument(
        "--type", 
        choices=["wiki", "news", "sample"], 
        default="sample",
        help="数据类型: wiki(维基百科), news(新闻), sample(示例数据)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("中文训练数据下载工具")
    print("=" * 60)
    
    if args.type == "wiki":
        print("\n📥 下载中文维基百科...")
        result = download_wikipedia_zh()
    elif args.type == "news":
        print("\n📥 下载中文新闻数据...")
        result = download_news_zh()
    else:
        print("\n📝 创建扩展示例数据...")
        result = create_sample_corpus()
    
    if result:
        print(f"\n📊 可以使用以下命令训练:")
        print(f"   python train_chinese.py --data-file {result}")
        print(f"   或者:")
        print(f"   make train CHINESE_DATA_FILE={result}")
