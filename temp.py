import json
import numpy as np

def select_top_n_keyframes(item, nframes=8):
    """
    根据 keyframe_distribution 选择 top n 的索引。
    """
    scores = item.get("keyframe_distribution", [])
    if not scores:
        return []

    scores_np = np.array(scores)
    top_indices = np.argsort(scores_np)[-nframes:][::-1]
    return top_indices.tolist()


def sample_n_keyframe_indices(item, nframes=8, replace=False):
    """
    根据 keyframe_distribution 的概率分布采样索引。
    """
    scores = item.get("keyframe_distribution", [])
    if not scores:
        return []

    scores_np = np.array(scores, dtype=np.float32)
    prob_dist = scores_np / (scores_np.sum() + 1e-8)

    if not replace:
        nframes = min(nframes, len(prob_dist))

    sampled_indices = np.random.choice(len(prob_dist), size=nframes, replace=replace, p=prob_dist)
    return sampled_indices.tolist()


def select_top_n_keyframes(item, nframes=8):
    """
    根据 score_distribution 选择 top n 的索引，并生成新的 nkeyframe_timestamps 字段。
    
    Args:
        item (dict): 单个条目，包含 "score_distribution" 和 "keyframe_timestamps"。
        top_n (int): 要选择的 top n 数量。
    
    Returns:
        List[float]: 新的 nkeyframe_timestamps 列表。
    """
    # 获取分数和对应的时间戳列表
    scores = item.get("keyframe_distribution", [])

    # 将 scores 转为 numpy 数组
    scores_np = np.array(scores)
    
    # 获取 top_n 的索引（降序）
    # np.argsort 返回从小到大的索引，所以取后 top_n 个，并逆序排序
    top_indices = np.argsort(scores_np)[-nframes:][::-1]
    
    # 根据 top_indices 选取对应的 timestamps
    top_timestamps = [i for i in top_indices]
    return top_timestamps.tolist()
def sample_n_keyframe_indices(item, nframes=8, replace=False):
    """
    根据 score_distribution 的概率分布采样索引（不使用 keyframe_timestamps）。

    Args:
        item (dict): 包含 "score_distribution"。
        nframes (int): 要采样的索引数量。
        replace (bool): 是否允许重复采样。

    Returns:
        List[int]: 被采样到的索引列表。
    """
    scores = item.get("keyframe_distribution", [])

    scores_np = np.array(scores, dtype=np.float32)
    prob_dist = scores_np / (scores_np.sum() + 1e-8)

    if not replace:
        nframes = min(nframes, len(prob_dist))

    sampled_indices = np.random.choice(len(prob_dist), size=nframes, replace=replace, p=prob_dist)
    return sampled_indices.tolist()

def process_json_file(input_json, nframe=8, mode="sample"):
    """
    处理 JSON 文件，为每个条目添加 top 或 sample 的 keyframe 索引字段，并保存回原文件。
    
    Args:
        input_json (str): JSON 文件路径
        nframe (int): 选择的帧数
        mode (str): "top" 或 "sample"
    
    Returns:
        List[dict]: 处理后的数据
    """
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if mode == "top":
            new_keyframes = select_top_n_keyframes(item, nframes=nframe)
        elif mode == "sample":
            new_keyframes = sample_n_keyframe_indices(item, nframes=nframe)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'top' or 'sample'.")
        new_keyframes.sort()
        item[f"{nframe}keyframe_indices"] = new_keyframes

    # 写回原始文件
    with open(input_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data


if __name__ == "__main__":
    input_json = "/data/guoweiyu/LV-Haystack/results/frame_search/TStar_LongVideoHaystack_tiny.json"
    
    process_json_file(input_json=input_json, nframe=32, mode="sample")
    pass
