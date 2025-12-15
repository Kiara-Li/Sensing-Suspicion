import praw
import pandas as pd
import os


REDDIT_CLIENT_ID = "RIZNxu1_2-gcAmIv3-Piwg" 
REDDIT_CLIENT_SECRET = "Sp3-sf4Bjk2Bq-pAzvPMfdvLt6Z_DQ"
REDDIT_USER_AGENT = "desktop:OwO:v1.0 (by u/Chemical-Arm2227)"

def scrape_data(limit_per_sub=1000):
    """
    抓取数据并保存为 CSV。
    limit_per_sub: 每个版块抓取的数量，建议 500-1000 以保证模型能学到东西。
    """
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    
    # 检查只读模式
    print(f"Read Only Mode: {reddit.read_only}")

    all_posts = []

    # 1. 定义正样本（恐怖/Creepy） -> Label 1
    creepy_subreddits = ["TwoSentenceHorror", "LetsNotMeet"]
    
    for sub_name in creepy_subreddits:
        print(f"Scraping Creepy (Label 1): r/{sub_name}...")
        try:
            subreddit = reddit.subreddit(sub_name)
            # time_filter="all" 抓取有史以来最好的帖子，保证质量
            for submission in subreddit.top(limit=limit_per_sub, time_filter="all"):
                content = submission.title + "\n" + (submission.selftext if submission.selftext else "")
                all_posts.append({
                    "text": content,
                    "label": 1,
                    "subreddit": sub_name
                })
        except Exception as e:
            print(f"Error scraping {sub_name}: {e}")

    # 2. 定义负样本（正常/Normal） -> Label 0
    # 增加 r/LifeStories 丰富数据
    normal_subreddits = ["CasualConversation", "PointlessStories", "BenignExistence", "LifeStories"]
    
    for sub_name in normal_subreddits:
        print(f"Scraping Normal (Label 0): r/{sub_name}...")
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.top(limit=limit_per_sub, time_filter="all"):
                content = submission.title + "\n" + (submission.selftext if submission.selftext else "")
                all_posts.append({
                    "text": content,
                    "label": 0,
                    "subreddit": sub_name
                })
        except Exception as e:
            print(f"Error scraping {sub_name}: {e}")

    # 3. 数据处理与清洗 (关键步骤!)
    df = pd.DataFrame(all_posts)
    print(f"原始抓取数量: {len(df)}")
    
    # a. 去除空值
    df = df.dropna(subset=['text'])
    
    # b. 去除 [removed] 和 [deleted] 的无效数据
    # 这是导致模型准确率低的最大原因
    filter_pattern = r'\[removed\]|\[deleted\]'
    df = df[~df['text'].str.contains(filter_pattern, case=False, regex=True)]
    
    # c. 去除太短的文本 (比如少于 20 个字符的垃圾内容)
    df = df[df['text'].str.len() > 20]
    
    # d. 去重 (防止同一篇文章出现多次)
    df = df.drop_duplicates(subset=['text'])

    # 这里的路径对应我们之前的目录结构
    output_path = "data/raw/reddit_posts.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False)
    
    print("-" * 30)
    print(f"✅ 完成！清洗后剩余有效数据: {len(df)} 条")
    print(f"数据已保存至: {output_path}")
    print("各类别数据分布:")
    print(df['label'].value_counts())
    print("-" * 30)

if __name__ == "__main__":
    scrape_data()