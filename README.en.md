# Seasonal adjustment: notebooks, module and data

An open set of teaching and working materials on the seasonal adjustment of time series: from what
seasonality is to a ready template into which you drop your own indicator.

Everything runs on ordinary Excel files in this folder — no database and no environment setup is
required. The notebooks are bilingual: the language is switched by one line, `LANG = "ru"` or
`LANG = "en"`.

[Русский](README.md) · **English**

---

## What is in the repository

| File | What it is |
|---|---|
| `x13_seasonal_full_pgh.ipynb` | The full walkthrough: ~160 cells. What seasonality is, how to measure it, the evolution of the methods from Census Method I (1954) to X-13ARIMA-SEATS, a comparison with TRAMO/SEATS and JDemetra+, methods outside that line (STL, MSTL, TBATS, Prophet, UCM, wavelets), applied metrics. Every claim is checked by code on data where the right answer is known in advance. |
| `x13_seasonal_short_pgh.ipynb` | An hour-long overview: the same ideas without derivations and experiments. A good entry point. |
| `indicator_template_pgh.ipynb` | A working template for your own series. Copy it for each new indicator; after every step it prints a check on whether your decision agrees with the data. The settings are saved as JSON next to the notebook. |
| `seasonal_toolbox.py` | The module: about 160 functions. Data loading, seasonality diagnostics, calendar regressors, outlier search, five decomposition implementations, a wrapper around the real X-13, applied metrics and charts. |
| `seasonal_adjustment_en.pdf` | A presentation, 32 slides. |
| `seasonal_adjustment_ru.pdf` | The same in Russian. |
| `RU_M2_M_CB.xlsx` | Example data: money supply M2, Bank of Russia, from 1992. |
| `RU_TDAYSWKPC_M_GV.xlsx` | The Russian working calendar: working days by month. |
| `requirements.txt` | Dependencies. |
| [`LICENSE`](LICENSE) | MIT for the code, CC BY 4.0 for the materials. |
| [`CITATION.cff`](CITATION.cff) | Data for the GitHub “Cite this repository” button. |

The data format is simple: two columns, `Date` and `Value`, dates at the end of the period. Put
your own series next to them in the same form.

---

## Installation

```bash
git clone https://github.com/MikeMike13/SEASONAL.git
cd SEASONAL
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Python 3.10 or newer is required.

---

## The X-13ARIMA-SEATS binary: downloaded separately

It is **not** in this repository and cannot be: it is a separate program by the US Census Bureau
with its own distribution terms. Everything works without it — the notebooks use the built-in
implementations, and the sections that need the real engine are skipped with a message.

For published figures, however, the official binary is what you want. On Russian M2 the built-in
implementations differ from the Bank of Russia estimate by 0.3–0.6% and the official engine by
0.3%. The gap is small on this series but may be larger on another, and the official engine
reproduces the methodology used by statistical offices.

**1. Download** it from the Census Bureau: https://www.census.gov/data/software/x13as.html

Pick the build for your system. You need the executable, usually called `x13as` (Linux, macOS) or
`x13as.exe` (Windows).

**2. Allow it to run** (Linux, macOS):

```bash
chmod +x /path/to/x13as
```

On macOS the system may block an unsigned program. Then: System Settings → Privacy & Security →
click "Allow" next to the block message, or

```bash
xattr -d com.apple.quarantine /path/to/x13as
```

**3. Put the path into the `PATH_X13` environment variable.** The notebooks read it themselves.

Linux and macOS — add to `~/.zshrc` or `~/.bashrc`:

```bash
export PATH_X13="/usr/local/bin/x13as"
```

Windows — once, in PowerShell:

```powershell
[Environment]::SetEnvironmentVariable("PATH_X13", "C:\x13as\x13as.exe", "User")
```

Or, if you would rather not touch the environment, directly in the notebook:

```python
sx.X13_BIN = "/path/to/x13as"
```

**4. Check.** The first cell of a notebook should print the file name rather than "not found":

```python
import seasonal_toolbox as sx
print(sx.X13_BIN, sx.x13_available())
```

---

## Where to start

If the topic is new — `x13_seasonal_short_pgh.ipynb`, about an hour.

If you want to understand it properly — `x13_seasonal_full_pgh.ipynb`. It is self-contained: every
claim is checked by code, and the data are synthetic with a known right answer, so you see not only
that a method works but how well.

If you need to adjust your own series — `indicator_template_pgh.ipynb`. Put the data file next to
it, change the settings in the first cell and run the whole notebook.

---

## What to know before applying it

**Seasonal adjustment is not a filter but a set of decisions.** Whether to take logs, whether to
account for the calendar, what counts as an outlier, which method to decompose with. These
decisions affect the result more than the choice between X-11 and SEATS, and they are made by a
person, not by a program.

**The end of the series is always unreliable.** The last months are estimated with an asymmetric
filter and will be revised as new data arrive. That is a property of the method, not a bug.

**The built-in implementations in the module are for teaching.** They reproduce the logic of X-11,
SEATS and TRAMO but are simpler than the official ones. That is enough to understand the ideas, not
enough for published figures.

---

## Licence and citation

The repository carries two licences, and the split is deliberate.

**The code** — `seasonal_toolbox.py` and the code cells in the notebooks — is under
**[MIT](https://opensource.org/licenses/MIT)**: use it freely, including in closed commercial
products, keeping the copyright notice.

**The materials** — the explanatory text of the notebooks, the presentations, the README and the
charts — are under **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**: copy, remix and
use them, commercially too, giving credit and indicating whether changes were made.

MIT is a licence for a program, and stretching it over explanations and slides would be a poor fit.
CC BY was made for text and illustrations and asks for attribution in any use, including a retelling
in an article or a lecture.

The full text of both licences is in [LICENSE](LICENSE). It also states the terms the data come
under and why the X-13 binary is not part of this repository.

If the materials were useful in a publication or a study:

```
exit10.ru. Seasonal adjustment: notebooks, module and data. 2026.
https://github.com/MikeMike13/SEASONAL
```

On the repository page GitHub shows a “Cite this repository” button: it reads `CITATION.cff`
and produces a ready reference in several formats.

The X-13ARIMA-SEATS binary and its algorithms are a development of the US Census Bureau and are
distributed on their terms.

---

## Feedback

If you find an error in the calculations or the text, open an issue. Methodological remarks are
especially welcome: there are several places where we simplify deliberately, all of them marked in
the text, but some may have been left unmarked.

[exit10.ru](https://exit10.ru)
