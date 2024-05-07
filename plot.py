import pandas as pd
import matplotlib.pyplot as plt


def plot_perplexity(input_file, output_file):
    df = pd.read_csv(input_file)

    plt.figure(figsize=(10, 5))

    plt.plot(df['Step'], df['Baseline'], label='Baseline', color='blue')
    plt.plot(df['Step'], df['Prenorm'], label='Prenorm', color='green')
    plt.plot(df['Step'], df['Postnorm'], label='Postnorm', color='red')

    plt.title('Model Perplexity over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Validation Perplexity')
    plt.legend(title="Model Types")
    plt.grid(True)

    plt.savefig(output_file)
    plt.show()
    plt.close()


plot_perplexity('model_ppl_log.csv', 'model_ppl_plot.png')
