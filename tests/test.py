import torch

def load_training_args(file_path):
    try:
        # 尝试使用 torch 加载
        data = torch.load(file_path, map_location="cpu")
        print("Loaded data:", data)
        return data
    except Exception as e:
        print(f"Failed to load using torch: {e}")
        return None

if __name__ == "__main__":
    file_path = "saves/full/Llama-33-70b-ins-sft_AIME_gpqa-diamond_HLE_usamo_20250404/training_args.bin"
    load_training_args(file_path)