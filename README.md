# viscose
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    similarities = []
    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i:i+args.batch_size]
        emb1 = model.encode(batch['text1'].tolist(), convert_to_tensor=True, device=device)
        emb2 = model.encode(batch['text2'].tolist(), convert_to_tensor=True, device=device)
        cos_sim = util.pytorch_cos_sim(emb1, emb2).diagonal().cpu().numpy()  # get pairwise sims
        similarities.extend(cos_sim)

    df['cosine_similarity'] = similarities
    df.to_csv(args.output, index=False)
