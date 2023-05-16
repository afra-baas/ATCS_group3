# this for each charactistic we compare

import matplotlib.pyplot as plt
import numpy as np


class AccuracyVisualizer:
    def __init__(self, data):
        self.data = data

    def visualize(self):
        models = ['bloom SA', 'bloom NLI', 'bloomz SA',
                  'bloomz NLI', 'flan SA', 'flan NLI']
        languages = ['English', 'German']
        prompts = ['active', 'passive']

        bar_width = 0.2
        opacity = 0.8
        x_pos = np.arange(len(models))

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, language in enumerate(languages):
            for j, prompt in enumerate(prompts):
                accuracies = self.data[(language, prompt)]
                offsets = x_pos + \
                    (i * bar_width * len(prompts)) + (j * bar_width)
                ax.bar(offsets, accuracies, bar_width, alpha=opacity,
                       label=f'{language} - {prompt} prompt')

        ax.set_xticks(x_pos + (bar_width * len(prompts) / 2))
        ax.set_xticklabels(models, rotation=45)
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison')
        ax.legend()

        plt.tight_layout()
        plt.show()

# TO DO: get accuracies from job_logs?


# example data
data = {
    ('English', 'active'): [0.58, 0.28, 0.86, 0.55, 0.85, 0.57],
    ('English', 'passive'): [0.80, 0.68, 0.78, 0.35, 0.72, 0.47],
    ('German', 'active'): [0.58, 0.28, 0.58, 0.55, 0.82, 0.57],
    ('German', 'passive'): [0.47, 0.48, 0.68, 0.35, 0.52, 0.37],
}

visualizer = AccuracyVisualizer(data)
visualizer.visualize()
