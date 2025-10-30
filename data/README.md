# Bulwer-Lytton sentences

`Bulwer-Lytton.tsv` contains all entries highlighted on the Bulwer-Lytton Fiction Contest website's [archive](https://www.bulwer-lytton.com/winners) between 1996 and 2024. It has 4 columns: year, sentence, category (or genre, as listed on the website) and position (winner, runner-up, dishonorable mention).

The remaining `txt` files are synthetic BL sentences generated using the scripts and settings listed in the `oneshot-generation` folder in this repository from 4 different LLMs - `DeepSeek-V3.1`, `gpt-5-2025-08-07`, `gpt-4.1-2025-04-14` and `gpt-oss-120b`.

## List of changes to Raw Bulwer Lytton sentence data

Sentences in the dataset are copied directly from the BLFC archive, except for the following changes:

- All single and double quotes were converted to **straight** single and double quotes.
- All 3 dot ellipses were converted to the ellipsis unicode character &hellip; (codepoint: U+2026)
- In 2017, the sentence "As he lay dying on the smoke-wreathed battlefield..." was listed under both *Adventure* and *Historical Fiction*. We listed the category as only *Adventure*.

**Note**: When loading `Bulwer-Lytton.tsv` with pandas, set the `quoting` parameter to `csv.QUOTE_NONE`. This ensures that python loads the text as is without inserting any quote characters that weren't present initially in the original sentence.

```
import pandas
import csv

df = pd.read_csv('Bulwer-Lytton.tsv', sep='\t', quoting=csv.QUOTE_NONE)
```