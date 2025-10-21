We start by running the humor detection models with each seed, which store results in `../data/humor_detection`. As an example, to store predictions for the BL dataset using the `humor-detection-comb` model with seed `23`, we would run

```bash
python predict.py hr_roberta --datasets bulwer --model_args humor-detection-comb-23
```

The included notebook reads in those humor detection results, averages across seeds, and plots.