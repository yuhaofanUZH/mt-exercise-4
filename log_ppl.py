import re
import pandas as pd

def extract_perplexities(log_file_path):
    ppl_pattern = r"ppl: *([0-9.]+)"
    step_pattern = r"Step: *(\d+)"
    data = {"Step": [], "Perplexity": []}
    last_step = None

    with open(log_file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            step_match = re.search(step_pattern, line)
            if step_match:
                last_step = int(step_match.group(1))

            ppl_match = re.search(ppl_pattern, line)
            if ppl_match and last_step is not None:
                ppl = float(ppl_match.group(1))
                data["Perplexity"].append(ppl)
                data["Step"].append(last_step)


    return pd.DataFrame(data)


if __name__ == "__main__":
    log_files = {
        "Baseline": "baseline.log",
        "Prenorm": "models/deen_transformer_pre/train.log",
        "Postnorm": "models/deen_transformer_post/train.log"
    }

    results = {}
    for model, path in log_files.items():
        results[model] = extract_perplexities(path)

    final_df = pd.merge(results["Baseline"], results["Prenorm"], on="Step", suffixes=('_Baseline', '_Prenorm'))
    final_df = pd.merge(final_df, results["Postnorm"], on="Step")
    final_df.rename(columns={"Perplexity": "Postnorm"}, inplace=True)
    final_df.columns = ['Step', 'Baseline', 'Prenorm', 'Postnorm']

    final_df.to_csv("model_ppl_log.csv", index=False)
    print("Data saved to model_perplexities_comparison.csv")
    print(final_df)
