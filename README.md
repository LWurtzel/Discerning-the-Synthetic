Discerning the Synthetic

Cross-generator detection of AI-generated images, with an interactive human-vs-model game.

Repository contents


CIFAKE_CNN.ipynb — from-scratch CNN trained on the CIFAKE dataset (single-generator baseline).
Genimage_CNN.ipynb — ResNet50V2 transfer-learning model trained across multiple GenImage generators.
spot_the_ai_game_all_generators.ipynb — the human-vs-model game as a self-contained notebook (auto-downloads the model and images).
docs/ — the browser-based game (served via GitHub Pages).
cifake_vs_genimage_generalization.png, cifake_to_genimage_collapse.png — result figures.
Slides — presentation covering the problem, approach, and findings.


Models

Trained models are published as assets under Releases (too large for the repo). The notebooks and the website download them automatically.

Datasets


CIFAKE — kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
GenImage — github.com/GenImage-Dataset/GenImage · per-generator Kaggle mirror: github.com/vtphatt2/GenImage-mirror



Claude Opus 4.8 assisted in creating several of the games, figures, and the GenImage model.
