"""Compare P(B|A) at first vs second occurrence of bigram AB.

If the model has induction heads, it should assign higher probability
to B after A the second time it sees the bigram, since it can copy
from the earlier context.
"""
import torch
import torch.nn.functional as F
import yaml
import numpy as np
from lib import AttentionLM


def run_induction_boost(checkpoint_path, config_path, data_path, max_stories=2000):
    cfg = yaml.safe_load(open(config_path))
    model = AttentionLM.from_config(cfg).cuda()
    ckpt = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint (step {ckpt['step']})")

    data = torch.load(data_path, weights_only=True).to(torch.long)

    first_logprobs = []
    second_logprobs = []
    first_ranks = []
    second_ranks = []

    with torch.no_grad():
        for batch_start in range(0, min(max_stories, data.shape[0]), 32):
            batch = data[batch_start:batch_start+32].cuda()
            logits = model(batch)

            for b in range(batch.shape[0]):
                seq = batch[b].tolist()
                seq_len = sum(1 for t in seq if t != 0)

                bigrams = {}
                found = False
                for j in range(seq_len - 2):
                    bg = (seq[j], seq[j+1])
                    if bg in bigrams and bigrams[bg] is not None:
                        first_pos = bigrams[bg]
                        second_pos = j
                        if second_pos - first_pos > 10:
                            target = seq[j+1]

                            # Log-prob at first occurrence
                            p1 = F.softmax(logits[b, first_pos], dim=-1)
                            lp1 = torch.log(p1[target] + 1e-10).item()
                            r1 = (logits[b, first_pos].argsort(descending=True) == target).nonzero(as_tuple=True)[0].item()

                            # Log-prob at second occurrence
                            p2 = F.softmax(logits[b, second_pos], dim=-1)
                            lp2 = torch.log(p2[target] + 1e-10).item()
                            r2 = (logits[b, second_pos].argsort(descending=True) == target).nonzero(as_tuple=True)[0].item()

                            first_logprobs.append(lp1)
                            second_logprobs.append(lp2)
                            first_ranks.append(r1)
                            second_ranks.append(r2)

                            bigrams[bg] = None
                            found = True
                            break
                    elif bg not in bigrams:
                        bigrams[bg] = j

    print(f"\nInduction boost test: P(B|A) at 1st vs 2nd occurrence")
    print(f"=====================================================")
    print(f"N examples:           {len(first_logprobs)}")
    print()
    print(f"                      1st occurrence    2nd occurrence    Boost")
    print(f"Mean log-prob:        {np.mean(first_logprobs):>10.3f}       {np.mean(second_logprobs):>10.3f}       {np.mean(second_logprobs)-np.mean(first_logprobs):>+.3f}")
    print(f"Mean rank:            {np.mean(first_ranks):>10.1f}       {np.mean(second_ranks):>10.1f}       {np.mean(second_ranks)-np.mean(first_ranks):>+.1f}")
    print(f"Median rank:          {np.median(first_ranks):>10.0f}       {np.median(second_ranks):>10.0f}")
    print(f"Top-1 accuracy:       {sum(1 for r in first_ranks if r==0)/len(first_ranks):>10.1%}       {sum(1 for r in second_ranks if r==0)/len(second_ranks):>10.1%}")
    print(f"Top-10 rate:          {sum(1 for r in first_ranks if r<10)/len(first_ranks):>10.1%}       {sum(1 for r in second_ranks if r<10)/len(second_ranks):>10.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default="/workspace/tensor-mars/workspaces/logan/cached_tokens/test_perstory.pt")
    parser.add_argument("--max-stories", type=int, default=2000)
    args = parser.parse_args()
    run_induction_boost(args.checkpoint, args.config, args.data, args.max_stories)
