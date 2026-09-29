# -*- coding: utf-8 -*-
# ============================================================================
#  seasonal_toolbox -- инструментарий для сезонной корректировки временных рядов
#
#  Автор:       exit10.ru
#  Репозиторий: https://github.com/MikeMike13/SEASONAL
#  Лицензия:    MIT (код). Пояснительные материалы репозитория -- CC BY 4.0.
#
#  Author:      exit10.ru
#  Repository:  https://github.com/MikeMike13/SEASONAL
#  Licence:     MIT (code). The explanatory materials -- CC BY 4.0.
#
#  Эта шапка едет вместе с файлом: если модуль скопируют в другой проект,
#  авторство останется видимым там, где его действительно читают.
#  This header travels with the file: if the module is copied elsewhere, the
#  attribution stays where people actually look.
# ============================================================================
"""seasonal_toolbox -- a toolkit for seasonal adjustment.  Version 1.0.

Everything that the notebook "Evolution of seasonal adjustment methods" used to define
inline, cell by cell, lives here in one place. The notebook now only EXPLAINS and CALLS;
it no longer redefines functions for the third time. That removes the main reproducibility
hazard: results used to depend on which cells had happened to run, because easter_date,
loess_1d, stl_lite, henderson_weights and x11_lite existed in two or three slightly
different variants.

Usage:

    import seasonal_toolbox as sx
    from seasonal_toolbox import *          # to call without the prefix

    sx.X13_BIN = "/usr/local/bin/x13as"     # path to the X-13ARIMA-SEATS binary
    sx.DEFAULT_EASTER_TRADITION = "orthodox"

Contents:
  * plotting style       -- econ_style, chart_frame, set_section, panel, cycle_colors
  * data loading         -- load_indicator, convert_to_level, decumulate_ytd
  * diagnostics          -- acf, periodogram, seasonal_dummy_ftest[_detrended],
                            seasonal_power_share, ljung_box, evaluate_seasonality
  * calendar & outliers  -- easter_date, easter_regressor, trading_day_regressor,
                            auto_outlier_search (with GLS whitening)
  * model choice         -- choose_transform (additive vs multiplicative)
  * decompositions       -- naive_decompose, x11_lite, stl_lite, mstl_lite,
                            ucm_lite (formerly seats_lite), trig_seasonal_fit,
                            prophet_lite, fit_ucm_mle, haar_dwt
  * applied metrics      -- three_month_ma_saar, forecast_deterministic, backtest_forecast
  * the real X-13        -- run_x13, make_spec, parse_*, read_d11
  * assembly             -- seasonal_adjustment_pipeline, analyze_series

Language: every comment and docstring is written in English and immediately followed by
its Russian translation. Every user-facing string (chart titles, labels, legends,
warnings, the verbal quality interpretation) switches with sx.set_lang("ru" | "en").

[RU] Модуль -- инструментарий сезонной корректировки, версия 1.0. Всё, что блокнот
«Эволюция методов сезонной корректировки» раньше определял прямо в ячейках, собрано здесь.
Блокнот теперь только объясняет и вызывает. Каждый комментарий и докстринг написан
по-английски, и сразу за ним идёт русский перевод. Все строки, которые видит
пользователь (заголовки и подписи графиков, легенды, предупреждения, словесная
интерпретация качества), переключаются через sx.set_lang("ru" | "en").
"""

__version__ = "1.0"

import os
import re
import stat
import datetime
import sqlite3
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
from scipy.optimize import minimize

try:
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.preprocessing import StandardScaler
except ImportError:            # prophet_lite is then unavailable | prophet_lite тогда недоступен
    Lasso = LinearRegression = StandardScaler = None

__all__ = [n for n in dir() if not n.startswith("_")]   # rebuilt at the end of the file | будет пересобран в конце файла

# ======================================================================
#  GLOBAL SETTINGS
#  Overridden from the notebook: sx.X13_BIN = "...", sx.DEFAULT_TRANSFORM = "log"
#  ГЛОБАЛЬНЫЕ НАСТРОЙКИ
#  Переопределяются из блокнота: sx.X13_BIN = "...", sx.DEFAULT_TRANSFORM = "log"
# ======================================================================
X13_BIN = os.getenv("PATH_X13") or os.path.abspath("./x13as")

DEFAULT_PERIOD = 12              # 12 for monthly data, 4 for quarterly | 12 -- месячные данные, 4 -- квартальные
DEFAULT_SMOOTH_WINDOW = 10       # window of the auxiliary centred-rolling smoothing used on charts | окно вспомогательных centered-rolling сглаживаний на графиках
DEFAULT_OUTLIER_CV = 3.5         # critical t-statistic for outlier detection | критическое значение t-статистики при поиске выбросов
DEFAULT_OUTLIER_TYPES = ("AO", "LS", "TC")
DEFAULT_EASTER_WINDOW = 8        # Easter regressor window in days (the X-13/regARIMA default) | окно пасхального регрессора в днях (дефолт X-13/regARIMA)
DEFAULT_EASTER_TRADITION = "orthodox"   # "orthodox" for Russian series, "catholic" for Western ones | "orthodox" для рядов РФ, "catholic" для западных
DEFAULT_TRANSFORM = "auto"       # "auto" | "log" | "none"
DEFAULT_GLS = True               # whether to whiten the residual before outlier search (see VIII.2) | отбеливать ли остаток перед поиском выбросов (см. VIII.2)

# ======================================================================
#  LANGUAGE
#  Every user-facing string (chart titles, labels, legends, warnings, the verbal
#  quality interpretation) exists in Russian and English. The notebook calls
#  sx.set_lang("ru" | "en") once, right after LANG is chosen; everything drawn or
#  printed afterwards follows it.
#  ЯЗЫК
#  Каждая строка, которую видит пользователь (заголовки и подписи графиков, легенды,
#  предупреждения, словесная интерпретация качества), существует на русском и на
#  английском. Блокнот один раз вызывает sx.set_lang("ru" | "en") сразу после выбора
#  LANG; всё, что рисуется или печатается дальше, следует этому выбору.
# ======================================================================
LANG = "ru"


def set_lang(lang):
    """Set the language of every user-facing string: "ru" or "en".

    [RU] Устанавливает язык всех пользовательских строк: "ru" или "en".
    """
    global LANG
    if lang not in ("ru", "en"):
        raise ValueError('LANG must be "ru" or "en" / LANG должен быть "ru" или "en"')
    LANG = lang
    return LANG


def tr(ru, en):
    """Return the variant for the current language: `en` when LANG == "en", else `ru`.
    Both arguments may be f-strings -- they are evaluated before the call.

    [RU] Возвращает вариант для текущего языка: `en` при LANG == "en", иначе `ru`.
    Оба аргумента могут быть f-строками -- они вычисляются до вызова.
    """
    return en if LANG == "en" else ru


def tx(entry):
    """Pick the current-language item from a {"ru": ..., "en": ...} entry. This is how
    notebook cells read the TXT dictionaries declared at their top.

    [RU] Выбирает элемент текущего языка из записи {"ru": ..., "en": ...}. Так ячейки
    блокнота читают словари TXT, объявленные в их начале.
    """
    return entry["en"] if LANG == "en" else entry["ru"]


def show_md(entry):
    """Render a {"ru": ..., "en": ...} Markdown block in the current language -- the
    notebook's replacement for static Markdown cells, which cannot switch language.

    [RU] Отображает Markdown-блок {"ru": ..., "en": ...} на текущем языке -- замена
    статическим Markdown-ячейкам блокнота, которые не умеют переключать язык.
    """
    from IPython.display import Markdown, display
    display(Markdown(tx(entry)))


# ======================================================================
#  PLOTTING STYLE
#  СТИЛЬ ОТРИСОВКИ
# ======================================================================
PALETTE_LIGHT = {
    "bg": "#FAF5E9", "text": "#16283B",
    "lazur": "#1D4E7E", "zoloto": "#D9A13B", "volna": "#3E7E7A",
    "korall": "#C05A3E", "sliva": "#7C6A9C",
    "grid": "#16283B", "muted": "#16283B",
}
PALETTE_DARK = {
    "bg": "#171613", "text": "#F0EDE4",
    "lazur": "#79B6D0", "zoloto": "#E8BA5E", "volna": "#6FBBA8",
    "korall": "#E0836A", "sliva": "#B3A0CE",
    "grid": "#F0EDE4", "muted": "#F0EDE4",
}
CYCLE_ORDER = ["lazur", "zoloto", "volna", "korall", "sliva"]

# The active palette. Updated by econ_style(); the plotting helpers in this module
# fall back to it when no pal is passed explicitly.
# Активная палитра. Обновляется econ_style(); функции модуля, рисующие графики,
# берут цвета отсюда, если им не передали pal явно.
pal = PALETTE_LIGHT

_installed = {f.name for f in fm.fontManager.ttflist}
FONT_TITLE = "Lora" if "Lora" in _installed else "serif"
FONT_LABEL = "DejaVu Sans" if "DejaVu Sans" in _installed else "sans-serif"
FONT_MONO = "DejaVu Sans Mono" if "DejaVu Sans Mono" in _installed else "monospace"

# Chart kicker = number and name of the current subsection; updated by set_section().
# Тема графика = номер и название текущего подраздела; обновляется set_section().
CURRENT_SECTION = ("I", {"ru": "Стиль отрисовки", "en": "Plotting style"})


def econ_style(mode="light"):
    """Apply the single plotting style: warm light or graphite dark background,
    azure as the main series colour, gold as the accent. No logo, no footer --
    only colours, grid and (when installed) the title font. Also updates the
    module-level `pal`, which the plotting helpers fall back to.

    [RU] Единый стиль отрисовки: тёплый светлый или графитовый тёмный фон, лазурь как
    основной цвет ряда, золото как акцент. Без логотипа и футера -- только цвета, сетка и
    (если установлен) заголовочный шрифт. Также обновляет модульную `pal`, на которую
    опираются функции отрисовки.
    """
    global pal
    pal = PALETTE_DARK if mode == "dark" else PALETTE_LIGHT
    mpl.rcParams.update({
        "figure.facecolor": pal["bg"], "axes.facecolor": pal["bg"], "savefig.facecolor": pal["bg"],
        "text.color": pal["text"], "axes.labelcolor": pal["text"],
        "xtick.color": pal["text"], "ytick.color": pal["text"], "axes.edgecolor": pal["text"],
        "axes.grid": True, "axes.grid.axis": "y",
        "grid.color": pal["grid"], "grid.alpha": 0.28,
        "grid.linewidth": 0.75, "grid.linestyle": (0, (1, 2)),
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": True, "axes.linewidth": 0.7,
        "font.family": FONT_LABEL, "font.size": 9,
        "axes.titlesize": 14, "axes.titleweight": "regular", "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "figure.figsize": [11, 5.2], "lines.linewidth": 2.0,
        "axes.prop_cycle": mpl.cycler(color=[pal[k] for k in CYCLE_ORDER]),
        "xtick.bottom": True, "ytick.left": False,
    })
    return pal

def annotate_source(ax, text=None, pal=None):
    """Write the source credit under the axes.

    [RU] Подпись источника под осями.
    """
    text = text if text is not None else tr("Источник: exit10.ru", "Source: exit10.ru")
    ax.text(0.0, -0.14, text, transform=ax.transAxes, fontsize=7.5,
             color=(pal or PALETTE_LIGHT)["text"], alpha=0.6, family=FONT_LABEL, ha="left")

def cycle_colors(n, pal=None):
    """Colours for n series -- simply cycle through the palette (azure, gold,
    wave, coral, plum, ...) without singling out one "hero" series.

    [RU] Цвета для n рядов -- просто циклически проходим по палитре (лазурь, золото,
    волна, коралл, слива, ...), не выделяя один ряд как «главный».
    """
    pal = pal or PALETTE_LIGHT
    return [pal[CYCLE_ORDER[i % len(CYCLE_ORDER)]] for i in range(n)]

def set_section(number, name):
    """Declare that the charts that follow belong to subsection {number}. {name}
    is what appears as the kicker above every subsequent chart and in the name of
    the auto-saved file. `name` may be a plain string or a {"ru": ..., "en": ...} entry;
    an entry is resolved at drawing time, so the kicker follows the current LANG.

    [RU] Объявляет, что следующие графики относятся к подразделу {number}. {name} --
    это появится рубрикой над каждым следующим графиком и в имени автосохранённого файла.
    `name` может быть строкой или записью {"ru": ..., "en": ...}; запись разрешается в
    момент отрисовки, поэтому рубрика следует текущему LANG.
    """
    global CURRENT_SECTION
    CURRENT_SECTION = (number, name)


def _slugify(text):
    """Transliterate Cyrillic and reduce the text to a filename-safe form.

    [RU] Транслитерация кириллицы и приведение текста к безопасному для имени файла виду.
    """
    table = str.maketrans({
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'e','ж':'zh','з':'z','и':'i','й':'i',
        'к':'k','л':'l','м':'m','н':'n','о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f',
        'х':'h','ц':'c','ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e','ю':'yu','я':'ya'
    })
    text = text.lower().translate(table)
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')[:60]

def _fit_text_to_axis(ax, text, fontsize=7.5, family=None, fontweight="bold", y=1.14, max_frac=0.98):
    """Truncate `text` (adding an ellipsis) until its ACTUAL rendered width fits
    the width of the axes. Without this a long subsection name stretches the saved
    image, because bbox_inches='tight' expands the figure around text that overflows.

    [RU] Обрезает `text` (добавляя многоточие), пока его РЕАЛЬНАЯ отрисованная ширина не
    впишется в ширину осей. Без этого длинное название подраздела растягивает сохраняемую
    картинку: bbox_inches='tight' раздвигает фигуру под вылезающий текст.
    """
    family = family or FONT_LABEL
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    target_px = ax.get_window_extent(renderer=renderer).width * max_frac

    def render_width(s):
        t = ax.text(0, y, s, transform=ax.transAxes, fontsize=fontsize,
                     fontweight=fontweight, family=family)
        w = t.get_window_extent(renderer=renderer).width
        t.remove()
        return w

    if render_width(text) <= target_px:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if render_width(text[:mid].rstrip() + "…") <= target_px:
            lo = mid
        else:
            hi = mid - 1
    return (text[:lo].rstrip() + "…") if lo > 0 else "…"

def chart_frame(ax, title, subtitle=None, kicker=None, source=None, pal=None, custody=None):
    """Kicker (number and name of the current subsection) + title (a conclusion,
    not a label) + subtitle (units / coverage / method) + source -- identical for
    every chart. The kicker is clipped to the real axes width so that a long
    subsection name cannot stretch the saved image.
    When one FIGURE holds several subplots (chart_frame is called once per subplot),
    the kicker is drawn only on the FIRST call for that figure.

    [RU] Рубрика (номер и название текущего подраздела) + заголовок (вывод, а не ярлык) +
    подзаголовок (единицы / охват / метод) + источник -- одинаково для каждого графика.
    Рубрика обрезается по реальной ширине осей, чтобы длинное название не растягивало
    сохраняемую картинку.
    Если на одной ФИГУРЕ несколько подграфиков (chart_frame вызывается по разу на каждый),
    рубрика рисуется только при ПЕРВОМ вызове для этой фигуры.
    """
    pal = pal or PALETTE_LIGHT
    fig = ax.figure
    if not getattr(fig, "_kicker_drawn", False):
        sec_num, sec_name = CURRENT_SECTION
        sec_name = tx(sec_name) if isinstance(sec_name, dict) else sec_name
        kicker_text = _fit_text_to_axis(ax, f"{sec_num}. {sec_name}".upper(), fontsize=7.5, fontweight="bold")
        ax.text(0, 1.14, kicker_text, transform=ax.transAxes, fontsize=7.5,
                 fontweight="bold", color=pal["zoloto"], family=FONT_LABEL, ha="left")
        fig._kicker_drawn = True
    ax.set_title(title, loc="left", fontfamily=FONT_TITLE, fontsize=14,
                 color=pal["text"], pad=18 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=8.5,
                 color=pal["text"], alpha=0.65, family=FONT_LABEL, ha="left")
    annotate_source(ax, source, pal=pal)

def panel(dates, series_dict, title, ylabel="", true_dict=None):
    """Quick plotting helper: several series on one axis, plus optional dashed
    "ground truth" lines to compare against.

    [RU] Быстрый хелпер отрисовки: несколько рядов на одной оси плюс, при желании,
    пунктирные «истинные» линии для сравнения с эталоном.
    """
    fig, ax = plt.subplots()
    for label, vals in series_dict.items():
        ax.plot(dates, vals, label=label, linewidth=1.8)
    if true_dict:
        for label, vals in true_dict.items():
            ax.plot(dates, vals, "--", color="gray", linewidth=1.2, alpha=0.8, label=label)
    chart_frame(ax, title, pal=pal)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
    plt.tight_layout()
    plt.show()

def acf(x, nlags):
    """Autocorrelation function from scratch: ACF[k] is the correlation of the
    series with itself shifted by k steps.

    [RU] Автокорреляционная функция с нуля: ACF[k] -- корреляция ряда с самим собой,
    сдвинутым на k шагов.
    """
    x = np.asarray(x, dtype=float); x = x - np.mean(x)
    return np.array([1.0] + [np.sum(x[k:]*x[:-k])/np.sum(x**2) for k in range(1, nlags+1)])

def periodogram_demo(x):
    """Raw periodogram for the illustrations in section IV.2: power by frequency,
    expressed in cycles per year for monthly data.

    [RU] Сырая периодограмма для иллюстраций раздела IV.2: мощность по частотам в циклах
    за год для месячных данных.
    """
    x = np.asarray(x, dtype=float); x = x - np.mean(x)
    fft = np.fft.rfft(x)
    power = (np.abs(fft)**2) / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0) * 12   # cycles per year for monthly data | циклы/год для месячных данных
    return freqs, power

def _share_at(freqs, power, targets, tol=0.15):
    """Share of total spectral power sitting at the given target frequencies
    (peak-based version used in the section IV.3 illustration).

    [RU] Доля полной спектральной мощности на заданных целевых частотах (вариант «по пикам»,
    используется в иллюстрации раздела IV.3).
    """
    total = power.sum() + 1e-12
    at = sum(power[np.abs(freqs-f) < tol].max() if (np.abs(freqs-f) < tol).any() else 0 for f in targets)
    return at / total

def periodogram(x):
    """Raw periodogram: frequencies (cycles per observation) and power, mean removed.

    [RU] Сырая периодограмма: частоты (циклы за наблюдение) и мощность; среднее вычтено.
    """
    x = np.asarray(x, dtype=float); x = np.nan_to_num(x - np.nanmean(x))
    n = len(x)
    fft = np.fft.rfft(x)
    power = (np.abs(fft)**2)/n
    freqs = np.fft.rfftfreq(n, d=1.0)
    return freqs, power

def seasonal_dummy_ftest(x, dates, period=12):
    """F-test on seasonal dummies WITHOUT detrending -- the naive variant, kept for
    the comparison in IV.5. In production use seasonal_dummy_ftest_detrended().

    [RU] F-тест на сезонные дамми БЕЗ детрендирования -- наивный вариант, оставлен для
    сравнения в IV.5. В продакшене используйте seasonal_dummy_ftest_detrended().
    """
    x = np.asarray(x, dtype=float); n = len(x)
    month_idx = (pd.DatetimeIndex(dates).month - 1) if period == 12 else (np.arange(n) % period)
    D = np.zeros((n, period)); D[np.arange(n), month_idx] = 1.0
    beta_f,*_ = np.linalg.lstsq(D, x, rcond=None); rss_f = np.sum((x-D@beta_f)**2)
    X_r = np.ones((n,1)); beta_r,*_ = np.linalg.lstsq(X_r, x, rcond=None); rss_r = np.sum((x-X_r@beta_r)**2)
    dof1, dof2 = period-1, n-period
    if dof2 <= 0 or rss_f <= 0: return {"F": np.nan, "p_value": np.nan}
    F = ((rss_r-rss_f)/dof1)/(rss_f/dof2)
    return {"F": F, "p_value": stats.f.sf(F, dof1, dof2)}

def seasonal_dummy_ftest_detrended(x, dates, period=12, trend_order=1):
    """F-test on seasonal dummies WITH detrending -- the default variant.

    A polynomial trend of degree `trend_order` is added to BOTH models (full and
    restricted). Without it the trend sits entirely in the residuals of both models,
    inflates the RSS and depresses F: on a strongly trending series the test loses
    power and may report "no seasonality" where seasonality is plainly there. The
    official counterpart in X-12/X-13 is computed on SI ratios, i.e. also on a
    detrended series.

    [RU] F-тест на сезонные дамми С детрендированием -- вариант по умолчанию.

    В ОБЕ модели (полную и ограниченную) добавлен полиномиальный тренд степени
    `trend_order`. Без этого тренд целиком сидит в остатках обеих моделей, раздувает RSS и
    занижает F: на ряде с сильным трендом тест теряет мощность и может отрапортовать
    «сезонности нет» там, где она очевидна. Официальный аналог в X-12/X-13 считается по
    SI-отношениям, то есть тоже по детрендированному ряду.
    """
    x = np.asarray(x, dtype=float); n = len(x)
    t = np.arange(n)
    trend_cols = [t**k for k in range(trend_order + 1)]
    month_idx = (pd.DatetimeIndex(dates).month - 1) if period == 12 else (np.arange(n) % period)
    D = np.zeros((n, period - 1))
    for m in range(period - 1):
        D[month_idx == m, m] = 1.0
    X_r = np.column_stack(trend_cols)
    X_f = np.column_stack(trend_cols + [D])
    b_r, *_ = np.linalg.lstsq(X_r, x, rcond=None); rss_r = np.sum((x - X_r @ b_r)**2)
    b_f, *_ = np.linalg.lstsq(X_f, x, rcond=None); rss_f = np.sum((x - X_f @ b_f)**2)
    dof1, dof2 = period - 1, n - X_f.shape[1]
    if dof2 <= 0 or rss_f <= 0:
        return {"F": np.nan, "p_value": np.nan}
    F = ((rss_r - rss_f)/dof1)/(rss_f/dof2)
    return {"F": F, "p_value": stats.f.sf(F, dof1, dof2)}

def seasonal_power_share(x, period=12, tol=0.15, cycles=None, mode="band"):
    """Share of spectral power that sits at the seasonal frequencies.

    mode="band" (default) -- the SUM of power in a band of +-tol around each seasonal
        frequency. This accounts for spectral leakage (in a finite sample the peak is
        smeared across several neighbouring bins), so the number may legitimately be
        quoted as "seasonality explains X% of the variance of the series".
    mode="peak" -- the maximum of a single bin in the same neighbourhood. As a
        before/after metric it behaves the same way, but it systematically
        understates the absolute share. Kept for the comparison in IV.5.
    cycles -- an explicit list of target frequencies in cycles per period (default
        1..period//2). For multi-seasonal series (daily data) pass only the
        frequencies you actually care about: with period=365 the default averages
        182 harmonics and the metric stops meaning anything.

    [RU] Доля спектральной мощности, приходящаяся на сезонные частоты.

    mode="band" (по умолчанию) -- СУММА мощности в полосе +-tol вокруг каждой сезонной
        частоты. Учитывает утечку спектра (у конечной выборки пик размазан по соседним
        бинам), поэтому число можно цитировать как «сезонность объясняет X% дисперсии ряда».
    mode="peak" -- максимум одного бина в той же окрестности. Как метрика «до/после» ведёт
        себя так же, но абсолютную долю систематически занижает. Оставлен для сравнения в IV.5.
    cycles -- явный список целевых частот в циклах за период (по умолчанию 1..period//2).
        Для многосезонных рядов (дневные данные) передавайте только интересующие частоты:
        при period=365 значение по умолчанию усредняет 182 гармоники и теряет смысл.
    """
    x = np.asarray(x, dtype=float); x = np.nan_to_num(x - np.nanmean(x))
    power = (np.abs(np.fft.rfft(x))**2)/len(x)
    cpy = np.fft.rfftfreq(len(x), d=1.0)*period
    cycles = np.arange(1, period//2 + 1) if cycles is None else np.asarray(cycles, dtype=float)
    if mode == "band":
        mask = np.zeros_like(cpy, dtype=bool)
        for f in cycles:
            mask |= np.abs(cpy - f) < tol
        num = power[mask].sum()
    else:
        num = 0.0
        for f in cycles:
            m = np.abs(cpy - f) < tol
            num += power[m].max() if m.any() else 0.0
    return num/(power.sum() + 1e-12)

def ljung_box(resid, lags):
    """Ljung-Box Q statistic at the given lags, implemented from scratch.
    Lags at or beyond the sample length are dropped (short or daily series).

    [RU] Статистика Q Льюнга-Бокса на заданных лагах, реализация с нуля. Лаги, не меньшие
    длины выборки, отбрасываются (короткие или дневные ряды).
    """
    x = np.asarray(resid, dtype=float); n = len(x); x = x - np.mean(x)
    lags = [k for k in lags if 0 < k < n]   # guard against lags >= sample length (short or daily series) | защита от лагов >= длины ряда (короткие/дневные ряды)
    if not lags or np.sum(x**2) == 0:
        return {"Q": np.nan, "p_value": np.nan}
    acf_vals = np.array([np.sum(x[k:]*x[:-k])/np.sum(x**2) for k in lags])
    Q = n*(n+2)*np.sum([acf_vals[i]**2/(n-lags[i]) for i in range(len(lags))])
    return {"Q": Q, "p_value": stats.chi2.sf(Q, len(lags))}

def evaluate_seasonality(original, adjusted, dates, period=12, seasonal=None, irregular=None,
                          outliers=None, method_name="", show_chart=False, dark=False,
                          detrend=True, power_mode="band", target_cycles=None):
    """UNIVERSAL diagnostic: applied identically to the output of X-11 / UCM / STL /
    anything else -- it only needs the original and the adjusted series.

    detrend=True        -- the F-test includes a trend in the model (see IV.5);
                           False gives the earlier, naive variant.
    power_mode="band"   -- power share as the sum over a band around each seasonal
                           frequency; "peak" is the earlier single-bin maximum.
    target_cycles       -- explicit list of seasonal frequencies (cycles per period).
                           Mandatory for multi-seasonal series: with period=365 the
                           default sweeps 182 harmonics and the metric loses meaning.
    irregular           -- pass it explicitly whenever the method returns it. Otherwise
                           the Ljung-Box test runs on np.diff(adjusted), and
                           differencing itself injects an MA(1) structure and changes
                           the autocorrelations at every lag.

    [RU] УНИВЕРСАЛЬНАЯ диагностика: применяется одинаково к результату X-11 / UCM / STL /
    чего угодно ещё -- нужны только исходный и скорректированный ряд.

    detrend=True        -- F-тест считается с трендом в модели (см. IV.5); False даёт прежний,
                           наивный вариант.
    power_mode="band"   -- доля мощности как сумма по полосе вокруг каждой сезонной частоты;
                           "peak" -- прежний максимум одного бина.
    target_cycles       -- явный список сезонных частот (циклов за период). Обязателен для
                           многосезонных рядов: при period=365 значение по умолчанию
                           перебирает 182 гармоники, и метрика теряет смысл.
    irregular           -- передавайте явно, когда метод его возвращает. Иначе тест
                           Льюнга-Бокса считается по np.diff(adjusted), а дифференцирование само
                           вносит MA(1)-структуру и меняет автокорреляции на всех лагах.
    """
    original = np.asarray(original, dtype=float); adjusted = np.asarray(adjusted, dtype=float)
    _ftest = seasonal_dummy_ftest_detrended if detrend else seasonal_dummy_ftest
    ftest_before = _ftest(original, dates, period)
    ftest_after  = _ftest(adjusted, dates, period)
    lb_input = irregular if irregular is not None else np.diff(adjusted)
    lb = ljung_box(lb_input - np.nanmean(lb_input), lags=[period, 2*period])

    freqs_b, power_b = periodogram(original); freqs_a, power_a = periodogram(adjusted)
    cpy_b, cpy_a = freqs_b*period, freqs_a*period
    seasonal_freqs = np.arange(1, period//2+1) if target_cycles is None else np.asarray(target_cycles, dtype=float)
    share_b = seasonal_power_share(original, period, cycles=seasonal_freqs, mode=power_mode)
    share_a = seasonal_power_share(adjusted, period, cycles=seasonal_freqs, mode=power_mode)

    metrics = {"method": method_name, "transform_scale": "work",
               "F_pvalue_before": ftest_before["p_value"], "F_pvalue_after": ftest_after["p_value"],
               "ljung_box_p_after": lb["p_value"],
               "seasonal_power_share_before": share_b, "seasonal_power_share_after": share_a,
               "seasonal_power_reduction_x": share_b/max(share_a,1e-9),
               "n_outliers_found": len(outliers) if outliers is not None else None}

    if show_chart:
        pal_ = econ_style("dark" if dark else "light")
        fig, axes = plt.subplots(2,2, figsize=(13,8))
        axes[0,0].plot(dates, original, color=pal_["korall"], label=tr("Исходный", "Original"))
        axes[0,0].plot(dates, adjusted, color=pal_["lazur"], label=tr("Скорректированный", "Adjusted"))
        chart_frame(axes[0,0], tr(f"{method_name}: исходный vs скорректированный", f"{method_name}: original vs adjusted"), pal=pal); axes[0,0].legend(fontsize=8)
        axes[0,1].stem(cpy_b, power_b, linefmt=pal_["korall"], markerfmt=" ", basefmt=" ")
        for f in seasonal_freqs: axes[0,1].axvline(f, color=pal_["muted"], linestyle=":", linewidth=0.8)
        axes[0,1].set_xlim(0, period/2+0.5); chart_frame(axes[0,1], tr("Периодограмма ДО", "Periodogram BEFORE"), pal=pal)
        axes[1,0].stem(cpy_a, power_a, linefmt=pal_["lazur"], markerfmt=" ", basefmt=" ")
        for f in seasonal_freqs: axes[1,0].axvline(f, color=pal_["muted"], linestyle=":", linewidth=0.8)
        axes[1,0].set_xlim(0, period/2+0.5); chart_frame(axes[1,0], tr("Периодограмма ПОСЛЕ -- пики должны исчезнуть", "Periodogram AFTER -- the peaks should be gone"), pal=pal)
        axes[1,1].bar([tr("До", "Before"), tr("После", "After")], [share_b*100, share_a*100], color=[pal_["korall"],pal_["lazur"]])
        chart_frame(axes[1,1], tr("Доля мощности на сезонных частотах, %", "Share of power at seasonal frequencies, %"), pal=pal)
        fig.suptitle(tr(f"Диагностика качества -- {method_name}", f"Quality diagnostics -- {method_name}"), y=1.02); plt.tight_layout(); plt.show()
    return metrics


def loess_1d(x, y, xout=None, frac=0.3, degree=1, robust_weights=None):
    """Locally weighted regression (Cleveland, 1979): for every point in xout take
    the nearest frac*n observations, weight them with a tricube kernel by distance,
    solve a weighted polynomial regression of degree `degree`, and predict at that point.

    [RU] Локально взвешенная регрессия (Cleveland, 1979): для каждой точки xout берём
    ближайшие frac*n наблюдений, взвешиваем их трикубическим ядром по расстоянию, решаем
    взвешенную полиномиальную регрессию степени `degree` и предсказываем в этой точке.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    n = len(x)
    if xout is None: xout = x
    k = max(int(np.ceil(frac * n)), degree + 2)
    out = np.empty(len(xout))
    for i, x0 in enumerate(xout):
        d = np.abs(x - x0)
        idx = np.argsort(d)[:k]
        dmax = d[idx].max() + 1e-12
        w = np.clip((1 - (d[idx] / dmax) ** 3) ** 3, 0, None)
        if robust_weights is not None:
            w = w * robust_weights[idx]
        X = np.vander(x[idx] - x0, degree + 1, increasing=True)
        W = np.diag(w)
        beta = np.linalg.solve(X.T @ W @ X + 1e-10*np.eye(degree+1), X.T @ W @ y[idx])
        out[i] = beta[0]
    return out

def _cycle_subseries_smooth(detrended, period, frac):
    """Smooth each cycle-subseries (all Januaries, all Februaries, ...) along the
    year axis with loess -- the seasonal step of STL.

    [RU] Сглаживает каждый цикл-подряд (все январи, все феврали, ...) вдоль оси лет с
    помощью loess -- сезонный шаг STL.
    """
    n = len(detrended); seasonal = np.empty(n)
    for m in range(period):
        idx = np.arange(m, n, period)
        seasonal[idx] = loess_1d(np.arange(len(idx)), detrended[idx], frac=frac, degree=1)
    return seasonal

def _moving_average(x, length):
    """Centred moving average of the given length, with truncated windows at the edges.

    [RU] Центрированная скользящая средняя заданной длины, с усечёнными окнами на краях.
    """
    n = len(x); out = np.full(n, np.nan); half = length // 2
    for i in range(n):
        lo, hi = max(i-half,0), min(i+half+(1 if length%2==1 else 0), n)
        out[i] = np.mean(x[lo:hi])
    return out

def stl_lite(y, period, n_inner=2, n_outer=2, seasonal_frac=0.4, trend_frac=None):
    """Pedagogically simplified STL: iterative backfitting (cycle-subseries smoothing
    -> low-pass filtering -> loess trend smoothing) with an outer loop of robustness
    weights that damp outliers.

    [RU] Педагогически упрощённая STL: итеративный backfitting (сглаживание цикл-подрядов ->
    низкочастотная фильтрация -> сглаживание тренда loess'ом) с внешним циклом робастных
    весов, подавляющих выбросы.
    """
    y = np.asarray(y, dtype=float); n = len(y)
    if trend_frac is None: trend_frac = 1.5*period/n
    t = np.arange(n); trend = np.zeros(n); robust_w = np.ones(n); seasonal = np.zeros(n)
    for outer in range(n_outer):
        for inner in range(n_inner):
            detrended = y - trend
            C = _cycle_subseries_smooth(detrended, period, seasonal_frac)
            L = loess_1d(t, np.nan_to_num(_moving_average(_moving_average(C, period), 3), nan=0.0),
                         frac=trend_frac, degree=1)
            seasonal = C - L
            trend = loess_1d(t, y - seasonal, frac=trend_frac, degree=1, robust_weights=robust_w)
        resid = y - trend - seasonal
        s = np.median(np.abs(resid))*6 + 1e-9
        robust_w = (1 - np.clip(np.abs(resid)/s, 0, 0.999)**2)**2
    return {"trend": trend, "seasonal": seasonal, "sa": y-seasonal, "irregular": y-trend-seasonal, "robust_w": robust_w}

def mstl_lite(y, periods, iterations=2, inner=1):
    """MSTL: extract the seasonal component of each period in turn with STL, applied to
    the residual left by the other components; repeat until it settles.

    [RU] MSTL: по очереди выделяем сезонную компоненту каждого периода с помощью STL,
    применённой к остатку от прочих компонент; повторяем до стабилизации.
    """
    n = len(y); t = np.arange(n)
    seasonal = {p: np.zeros(n) for p in periods}
    trend = np.zeros(n)
    for it in range(iterations):
        for p in periods:
            other = sum(seasonal[p2] for p2 in periods if p2 != p)
            res = stl_lite(y - trend - other, period=p, n_inner=inner, n_outer=1,
                            trend_frac=max(1.5*p/n, 0.05))
            seasonal[p] = res["seasonal"]
        trend = loess_1d(t, y - sum(seasonal.values()), frac=max(2*max(periods)/n, 0.15), degree=1)
    total = sum(seasonal.values())
    return {"trend": trend, "seasonal": seasonal, "sa": y - total,
            "irregular": y - trend - total}

def easter_date(year, tradition="catholic"):
    """Easter date in the GREGORIAN calendar.

    tradition="catholic" -- Western Easter, Meeus/Jones/Butcher algorithm (the default
        of X-13/TRAMO and of the whole Western literature).
    tradition="orthodox" -- Orthodox Easter: computed in the JULIAN calendar (Gauss
        algorithm) and converted to Gregorian by adding the calendar difference
        (13 days over 1900-2099). This is the one Russian series need -- see VII.1.

    [RU] Дата Пасхи в ГРИГОРИАНСКОМ календаре.

    tradition="catholic" -- западная Пасха, алгоритм Meeus/Jones/Butcher (дефолт X-13/TRAMO
        и всей западной литературы).
    tradition="orthodox" -- православная Пасха: считается в ЮЛИАНСКОМ календаре (алгоритм
        Гаусса) и переводится в григорианский прибавлением разницы календарей (13 суток в
        1900-2099). Для российских рядов нужна именно она -- см. VII.1.
    """
    if tradition == "orthodox":
        a = year % 19; b = year % 4; c = year % 7
        d = (19*a + 15) % 30
        e = (2*b + 4*c + 6*d + 6) % 7
        julian_march_day = 22 + d + e          # day of March in the JULIAN calendar (may exceed 31) | день марта по ЮЛИАНСКОМУ календарю (может быть > 31)
        jd = datetime.date(year, 3, 1) + datetime.timedelta(days=julian_march_day - 1)
        shift = year//100 - year//400 - 2      # difference between the Julian and Gregorian calendars | разница юлианского и григорианского календарей
        return jd + datetime.timedelta(days=shift)
    if tradition != "catholic":
        raise ValueError(tr("tradition должен быть 'catholic' или 'orthodox'",
                            "tradition must be 'catholic' or 'orthodox'"))
    a = year % 19; b = year // 100; c = year % 100
    d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4; k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    return datetime.date(year, month, day)

def easter_regressor_yoy(dates, L=8):
    """Easter regressor used by the year-on-year illustration in VI.2.2 (Western
    Easter, uncentred, March/April only) -- kept separate so that section stays
    exactly reproducible.

    [RU] Пасхальный регрессор для иллюстрации «год к году» в VI.2.2 (западная Пасха, без
    центрирования, только март/апрель) -- вынесен отдельно, чтобы этот раздел
    воспроизводился в точности.
    """
    dts = [d.date() if hasattr(d, "date") else d for d in dates]
    reg = np.zeros(len(dts))
    for y in sorted(set(d.year for d in dts)):
        ed = easter_date(y)
        window = [ed - datetime.timedelta(days=k) for k in range(1, L+1)]
        for idx, d in enumerate(dts):
            if d.year == y and d.month in (3, 4):
                cnt = sum(1 for w in window if w.month == d.month and w.year == d.year)
                reg[idx] += cnt / L
    return reg

def yoy(x, per=12):
    """Year-on-year growth in percent for the given periodicity.

    [RU] Темп год к году в процентах для заданной периодичности.
    """
    x = pd.Series(x)
    return (x/x.shift(per) - 1)*100

def decumulate_ytd(series):
    """Split a series reported as a YEAR-TO-DATE CUMULATIVE TOTAL (the usual Rosstat
    format for fixed investment, exports/imports and others) into individual period
    values. Requires a DatetimeIndex/PeriodIndex so that the function can tell where
    each year starts, i.e. where the accumulation resets.

    [RU] Разбивает ряд, публикуемый НАРАСТАЮЩИМ ИТОГОМ С НАЧАЛА ГОДА (обычный формат Росстата
    для инвестиций, экспорта/импорта и др.), на значения отдельных периодов. Нужен
    DatetimeIndex/PeriodIndex, чтобы определить, где начинается каждый год, то есть где
    накопление сбрасывается.
    """
    if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        raise ValueError(tr("decumulate_ytd требует DatetimeIndex или PeriodIndex у входного ряда",
                            "decumulate_ytd requires a DatetimeIndex or PeriodIndex on the input"))
    years = series.index.year
    is_year_start = years != np.roll(years, 1)
    is_year_start[0] = True   # the first observation is always a 'start' (we cannot know what came before) | первое наблюдение ряда -- всегда 'старт' (не знаем, что было раньше)
    diffs = series.diff()
    return series.where(is_year_start, diffs)

def choose_transform(y, period=12, threshold=0.5, min_level_range=0.1, return_detail=False):
    """Choose between an additive and a multiplicative (logarithmic) model -- the
    "range against level" (range-mean) test.

    The series is cut into blocks of `period` observations (a year). For every block the
    median level and the interquartile range are computed. If the seasonal amplitude is
    PROPORTIONAL to the level (multiplicative model), then log(range) grows together with
    log(median) with a slope near 1; if the amplitude is constant (additive model), the
    slope is near zero. The threshold=0.5 separates the two cases.

    Why this and not a comparison of the AICC of two regressions (on y and on log y).
    Formally the latter is more correct -- and it is exactly what X-13 does in
    transform{function=auto}, bringing the likelihood of the log model back to the original
    scale with the Jacobian correction (-sum log y). But that test requires a CORRECTLY
    SPECIFIED mean model: an unmodelled LS/TC or a bend in the trend produces huge
    residuals that are relatively smaller on the logarithmic scale, and the test then picks
    the logarithm because of the outlier rather than because of multiplicativity. Inside
    X-13 the problem does not arise, because its aictest runs within regARIMA where the
    outliers and the calendar are already in the model. The range-against-level test does
    not depend on the specification of the trend at all, and the median and the IQR are
    insensitive to individual outliers -- which makes it the more reliable choice for a
    standalone function.

    extra_cols -- additional regressor columns (typically dummies for outliers already
    found); kept for the AICC-style variant and for experiments.

    Returns "log" or "none"; with return_detail=True also the slope and an explanation.
    Series with zeros or negative values always return "none" (the logarithm is undefined),
    as do series whose level barely moves over the sample -- there the two models are
    practically indistinguishable and the additive one is the safer default.

    [RU] Выбор между аддитивной и мультипликативной (логарифмической) моделью -- тест
    «размах против уровня» (range-mean).

    Ряд режется на блоки по `period` наблюдений (год). Для каждого блока считаются медиана
    уровня и межквартильный размах. Если сезонная амплитуда ПРОПОРЦИОНАЛЬНА уровню
    (мультипликативная модель), log(размаха) растёт вместе с log(медианы) с наклоном около 1;
    если амплитуда постоянна (аддитивная модель) -- наклон около нуля. Порог threshold=0.5
    разделяет эти два случая.

    Почему так, а не сравнением AICC двух регрессий (на y и на log y). Формально второе
    правильнее -- именно это делает X-13 в transform{function=auto}, приводя правдоподобие
    лог-модели к исходной шкале поправкой на якобиан (-sum log y). Но такой тест требует
    ПРАВИЛЬНО СПЕЦИФИЦИРОВАННОЙ модели среднего: неучтённый LS/TC или изгиб тренда дают
    огромные остатки, которые на логарифмической шкале относительно меньше, и тест выбирает
    логарифм из-за выброса, а не из-за мультипликативности. Внутри X-13 проблемы нет, потому
    что его aictest выполняется внутри regARIMA, где выбросы и календарь уже в модели. Тест
    «размах против уровня» от спецификации тренда не зависит вовсе, а медиана и IQR
    нечувствительны к отдельным выбросам -- поэтому для автономной функции он надёжнее.

    extra_cols -- дополнительные столбцы регрессоров (обычно дамми уже найденных выбросов);
    оставлены для AICC-варианта и экспериментов.

    Возвращает "log" или "none"; при return_detail=True -- ещё наклон и пояснение.
    Ряды с нулями или отрицательными значениями всегда дают "none" (логарифм не определён),
    как и ряды, уровень которых за выборку почти не меняется: там две модели практически
    неразличимы и аддитивная безопаснее.
    """
    y = np.asarray(y, dtype=float)
    def _out(res, slope, why):
        return (res, slope, why) if return_detail else res
    if not np.isfinite(y).all() or np.nanmin(y) <= 0:
        return _out("none", np.nan, tr("в ряде есть неположительные значения", "the series has non-positive values"))
    n_blocks = len(y)//period
    if n_blocks < 4:
        return _out("none", np.nan, tr("меньше 4 полных периодов -- тест неинформативен", "fewer than 4 full periods -- the test is uninformative"))
    blocks = y[:n_blocks*period].reshape(n_blocks, period)
    med = np.median(blocks, axis=1)
    iqr = np.percentile(blocks, 75, axis=1) - np.percentile(blocks, 25, axis=1)
    ok = (iqr > 0) & (med > 0)
    if ok.sum() < 4:
        return _out("none", np.nan, tr("слишком мало пригодных блоков", "too few usable blocks"))
    log_med = np.log(med[ok])
    if log_med.max() - log_med.min() < min_level_range:
        return _out("none", np.nan, tr("уровень ряда почти не меняется -- модели неразличимы", "the level barely moves -- the models are indistinguishable"))
    X = np.column_stack([np.ones(ok.sum()), log_med])
    coef, *_ = np.linalg.lstsq(X, np.log(iqr[ok]), rcond=None)
    slope = float(coef[1])
    return _out("log" if slope > threshold else "none", slope, "ok")

def easter_regressor(dates, L=8, tradition="catholic", center=False):
    """Share of the L-day window BEFORE Easter that falls into each calendar month.

    tradition -- 'catholic' or 'orthodox' (see easter_date above and section VII.1).
    center    -- subtract the mean of each calendar month, the way X-13 does.
                 An uncentred regressor carries a seasonal wave of its own (it is
                 non-zero almost always in the same months) and partly substitutes
                 for the seasonal factor, stealing explained variance from it.
                 Use True in production; the illustrations keep False so the regressor
                 reads literally as "the share of the window".

    The months are NOT hard-coded: the window belongs to whichever month its days
    fall into. That matters for Orthodox Easter, which also occurs in May, whereas
    the earlier version searched only March and April.

    [RU] Доля L-дневного окна ПЕРЕД Пасхой, попавшая в каждый календарный месяц.

    tradition -- 'catholic' или 'orthodox' (см. easter_date выше и раздел VII.1).
    center    -- вычесть среднее по каждому календарному месяцу, как это делает X-13.
                 Нецентрированный регрессор сам несёт сезонную волну (он ненулевой почти
                 всегда в одних и тех же месяцах) и частично подменяет сезонный фактор,
                 отбирая у него объяснённую дисперсию. В продакшене -- True; в иллюстрациях
                 оставлено False, чтобы регрессор читался буквально как «доля окна».

    Месяцы НЕ зашиты в код: окно относится к тому месяцу, куда попали его дни. Это важно для
    православной Пасхи, которая бывает и в мае, -- прежняя версия искала только март и апрель.
    """
    dts = [d.date() if hasattr(d, "date") else d for d in dates]
    reg = np.zeros(len(dts))
    easters = {y: easter_date(y, tradition) for y in sorted({d.year for d in dts})}
    for idx, d in enumerate(dts):
        window = [easters[d.year] - datetime.timedelta(days=k) for k in range(1, L+1)]
        reg[idx] = sum(1 for w in window if (w.year, w.month) == (d.year, d.month))/L
    if center:
        months = np.array([d.month for d in dts])
        for m_ in np.unique(months):
            sel = months == m_
            reg[sel] = reg[sel] - reg[sel].mean()
    return reg

def trading_day_regressor(dates, working_days_calendar=None):
    """Regressor for the number of working / trading days.

    Without arguments -- the simplified version: the deviation of the number of
    weekdays (Mon-Fri) in the month from its expected average (5/7 of the month
    length), WITHOUT national holidays (i.e. assuming every weekday is worked).

    With working_days_calendar -- the OFFICIAL WORKING CALENDAR of any country: a
    dict {(year, month): number_of_working_days} that accounts for real holidays.
    That is the general case; the simplified version above is just its special case
    with no holidays at all.

    [RU] Регрессор числа рабочих / торговых дней.

    Без аргументов -- упрощённая версия: отклонение числа будних дней (Пн-Пт) в месяце от
    ожидаемого среднего (5/7 длины месяца), БЕЗ национальных праздников (то есть считаем,
    что все будни рабочие).

    С working_days_calendar -- ПРОИЗВОДСТВЕННЫЙ КАЛЕНДАРЬ любой страны: словарь
    {(год, месяц): число_рабочих_дней} с учётом реальных праздников. Это общий случай;
    упрощённая версия выше -- лишь его частный случай без праздников вообще.
    """
    dts = pd.DatetimeIndex(dates)
    if working_days_calendar is not None:
        wd = np.array([working_days_calendar.get((d.year, d.month)) for d in dts], dtype=float)
        if np.isnan(wd).any():
            missing = dts[np.isnan(wd)][:5].tolist()
            raise ValueError(tr(f"В working_days_calendar не хватает записей для: {missing}",
                               f"working_days_calendar has no entries for: {missing}"))
        return wd - wd.mean()
    reg = np.zeros(len(dts))
    for i, d in enumerate(dts):
        days_in_month = pd.Period(d, freq="M").days_in_month
        month_days = pd.date_range(d.replace(day=1), periods=days_in_month, freq="D")
        weekdays = (month_days.weekday < 5).sum()
        reg[i] = weekdays - (5/7)*days_in_month
    return reg

def _harmonic_trend_cols(n, period=12, n_harm=2):
    """Build the base regressors: constant + linear trend + n_harm sine/cosine pairs
    of seasonal harmonics -- the backbone of the regression in auto_outlier_search.

    [RU] Базовые регрессоры: константа + линейный тренд + n_harm пар синус/косинус сезонных
    гармоник -- основа регрессии в auto_outlier_search.
    """
    t = np.arange(n)
    cols = [np.ones(n), t]
    for k in range(1, n_harm+1):
        cols += [np.sin(2*np.pi*k*t/period), np.cos(2*np.pi*k*t/period)]
    return cols, t

def _outlier_col(t, idx0, kind, alpha=0.7):
    """Build one outlier dummy: AO (single point), LS (permanent level shift),
    TC (temporary change decaying at the fixed rate alpha).

    [RU] Один регрессор-дамми выброса: AO (точка), LS (сдвиг уровня навсегда),
    TC (временное изменение, затухающее с фиксированной скоростью alpha).
    """
    n = len(t); col = np.zeros(n)
    if kind == "AO": col[idx0] = 1.0
    elif kind == "LS": col[idx0:] = 1.0
    elif kind == "TC": col[idx0:] = alpha ** np.arange(n - idx0)
    return col

def ar_coefficients(resid, order=1, max_persistence=0.98):
    """AR(p) coefficients of a residual series (OLS on its own lags).

    Needed for the whitening step in auto_outlier_search(gls=True). The sum of |phi|
    is shrunk towards max_persistence when necessary, so that the whitening filter
    stays stable even close to a unit root.

    [RU] Коэффициенты AR(p) остатка (МНК по собственным лагам).

    Нужны для отбеливания в auto_outlier_search(gls=True). Сумма |phi| при необходимости
    поджимается к max_persistence, чтобы отбеливающий фильтр оставался устойчивым даже
    вблизи единичного корня.
    """
    r = np.asarray(resid, dtype=float); r = r - np.mean(r)
    n = len(r)
    if order < 1 or n <= order + 2:
        return np.zeros(0)
    Xl = np.column_stack([r[order-k-1: n-k-1] for k in range(order)])
    phi, *_ = np.linalg.lstsq(Xl, r[order:], rcond=None)
    total = np.abs(phi).sum()
    if total >= max_persistence:
        phi = phi*(max_persistence/total)
    return phi


def whiten(v, phi):
    """Whitening filter: v_t - phi_1 v_{t-1} - ... - phi_p v_{t-p}.
    The result is p observations shorter than the input (the first p points have
    nothing to be differenced against).

    [RU] Отбеливающий фильтр: v_t - phi_1 v_{t-1} - ... - phi_p v_{t-p}.
    Результат короче входа на p наблюдений (первым p точкам не с чем разностить).
    """
    v = np.asarray(v, dtype=float); p = len(phi)
    if p == 0:
        return v
    out = v[p:].copy()
    for k in range(p):
        out = out - phi[k]*v[p-k-1: len(v)-k-1]
    return out


def auto_outlier_search(y, base_cols, period=12, cv=3.5, types=("AO","LS","TC"),
                         alpha=None, min_gap=3, max_outliers=8, skip_edge=6,
                         calendar_cols=(), backward=True, gls=None, ar_order=1):
    """Automatic outlier detection: whitening (GLS) -> forward pass -> backward deletion.

    WHY WHITENING. An outlier is selected on the t-statistic of its coefficient. Plain
    OLS computes that statistic assuming independent residuals: Var(beta) = sigma^2 (X'X)^-1.
    The residual of a macro series is almost always autocorrelated (business-cycle
    inertia), and then this formula UNDERSTATES the standard error -- neighbouring
    observations carry less new information than OLS assumes, yet the formula treats
    them as independent pieces of evidence. An understated SE means an inflated |t|,
    and at one and the same threshold cv=3.5 the algorithm starts declaring ordinary
    fluctuations to be outliers. The typical symptom is a cluster of spurious LS
    wherever the trend merely curved.

    Real regARIMA (X-13) and TRAMO solve this by searching for outliers not in an OLS
    regression but inside a model with ARIMA errors. What is implemented here is the
    same idea, simpler: the residual of the base model is described by an AR(p), and a
    whitening filter (Cochrane-Orcutt transformation) is applied to the series AND to
    every column of the regressor matrix. After that the residual is close to white
    noise, the OLS formula for the SE is valid again, and the t-statistics become
    comparable with the critical values those thresholds were designed for.

    WHAT CHANGES IN THE OUTPUT. The coefficient estimates were unbiased under OLS too --
    what changes is the STANDARD ERRORS and, through them, the set of detected outliers:
    far fewer false positives, and t-statistics in the report that drop, sometimes
    several-fold. That is not a worse result, it is the removal of an illusion of
    precision. The effects (`outlier_effect`, `calendar_effect`, `y_clean`) are returned
    in the ORIGINAL scale of the series, not the whitened one -- whitening is only a way
    to weight observations correctly while estimating and testing.

    Parameters:
    gls        -- whether to whiten; None => the module-level DEFAULT_GLS. gls=False
                  restores plain OLS behaviour (useful for the comparison in VIII.2).
    ar_order   -- AR order used for whitening; 1 covers the vast majority of macro
                  series, 2-3 makes sense with pronounced quarterly inertia.
    backward   -- backward deletion, as in X-13: candidates that lost significance in
                  the presence of later additions are removed. Without this step two
                  nearby events often produce an "AO + LS" pair where X-13 would have
                  kept a single LS.
    alpha      -- TC decay rate. None => the X-13 default for the given periodicity:
                  0.7 for monthly data and 0.7**3 for quarterly.
    calendar_cols -- indices of the columns of base_cols that are CALENDAR regressors
                  (Easter, trading days); their estimated effect is returned separately
                  as calendar_effect, so the caller can subtract it.
    skip_edge  -- width of the blind zone at the ends of the series (see VIII.1).

    Returns a dict: outliers (position, type, t), dropped_backward, beta, y_clean,
    outlier_effect, calendar_effect, n_base_cols, ar_coeffs, gls.

    [RU] Автоматический поиск выбросов: отбеливание (GLS) -> прямой проход -> обратный отсев.

    ЗАЧЕМ ОТБЕЛИВАНИЕ. Выброс отбирается по t-статистике его коэффициента. Обычный МНК
    считает её в предположении независимых остатков: Var(beta) = sigma^2 (X'X)^-1. Остаток
    макроряда почти всегда автокоррелирован (инерция делового цикла), и тогда эта формула
    ЗАНИЖАЕТ стандартную ошибку: соседние наблюдения несут меньше новой информации, чем
    предполагает МНК, а формула считает их независимыми свидетельствами. Заниженная SE --
    это завышенный |t|, и при одном и том же пороге cv=3.5 алгоритм начинает объявлять
    выбросами обычные колебания. Типичный симптом -- гроздь ложных LS там, где тренд просто
    изогнулся.

    Настоящие regARIMA (X-13) и TRAMO решают это тем, что ищут выбросы не в МНК-регрессии, а
    внутри модели с ARIMA-ошибками. Здесь реализована та же идея, проще: остаток базовой
    модели описывается AR(p), и к ряду И ко всем столбцам матрицы регрессоров применяется
    отбеливающий фильтр (преобразование Кокрена-Оркатта). После этого остаток близок к
    белому шуму, МНК-формула для SE снова верна, и t-статистики становятся сопоставимы с
    критическими значениями, на которые рассчитаны пороги.

    ЧТО МЕНЯЕТСЯ НА ВЫХОДЕ. Оценки коэффициентов были несмещёнными и при МНК -- меняются
    СТАНДАРТНЫЕ ОШИБКИ и через них состав найденных выбросов: ложных находок гораздо меньше,
    а t-статистики в отчёте падают, иногда в разы. Это не ухудшение результата, а снятие
    иллюзии точности. Эффекты (`outlier_effect`, `calendar_effect`, `y_clean`) возвращаются в
    ИСХОДНОЙ шкале ряда, а не в отбеленной: отбеливание -- лишь способ правильно взвесить
    наблюдения при оценивании и тестировании.

    Параметры:
    gls        -- отбеливать ли; None => модульный DEFAULT_GLS. gls=False возвращает чистое
                  МНК-поведение (полезно для сравнения в VIII.2).
    ar_order   -- порядок AR для отбеливания; 1 покрывает подавляющее большинство
                  макрорядов, 2-3 имеет смысл при выраженной квартальной инерции.
    backward   -- обратный отсев, как в X-13: кандидаты, потерявшие значимость в присутствии
                  добавленных позже, удаляются. Без этого шага два близких события часто дают
                  пару «AO + LS» там, где X-13 оставил бы один LS.
    alpha      -- скорость затухания TC. None => дефолт X-13 для данной периодичности:
                  0.7 для месячных данных и 0.7**3 для квартальных.
    calendar_cols -- индексы столбцов base_cols, являющихся КАЛЕНДАРНЫМИ регрессорами
                  (Пасха, торговые дни); их оценённый эффект возвращается отдельно как
                  calendar_effect, чтобы вызывающий код мог его вычесть.
    skip_edge  -- ширина «слепой зоны» у краёв ряда (см. VIII.1).

    Возвращает словарь: outliers (позиция, тип, t), dropped_backward, beta, y_clean,
    outlier_effect, calendar_effect, n_base_cols, ar_coeffs, gls.
    """
    y = np.asarray(y, dtype=float); n = len(y)
    if gls is None:
        gls = DEFAULT_GLS
    if alpha is None:
        alpha = 0.7 if period == 12 else 0.7**(12/max(period, 1))
    X_base = np.column_stack(base_cols)

    # --- step 0: fit AR(p) to the base-model residual and build the whitening filter ---
    # --- шаг 0: оцениваем AR(p) остатка базовой модели и строим отбеливающий фильтр ---
    phi = np.zeros(0)
    if gls:
        b0, *_ = np.linalg.lstsq(X_base, y, rcond=None)
        phi = ar_coefficients(y - X_base@b0, ar_order)

    def _w(v):                      # whiten a column or the series | отбеливание столбца или ряда
        return whiten(v, phi)

    yw = _w(y)
    Xw_base = np.column_stack([_w(X_base[:, k]) for k in range(X_base.shape[1])])

    def _tstat_last(Xw, yw_):
        beta, *_ = np.linalg.lstsq(Xw, yw_, rcond=None)
        resid = yw_ - Xw@beta
        dof = len(yw_) - Xw.shape[1]
        if dof <= 0:
            return None, None
        sigma2 = (resid@resid)/dof
        se = np.sqrt(max(sigma2*np.linalg.pinv(Xw.T@Xw)[-1, -1], 1e-12))
        return beta[-1]/se, beta

    # --- forward pass ---
    # --- прямой проход ---
    found, extra_cols, extra_w = [], [], []
    candidates_idx = list(range(skip_edge, n - skip_edge))
    for step in range(max_outliers):
        Xw_cur = np.column_stack([Xw_base] + extra_w) if extra_w else Xw_base
        best = None
        for idx0 in candidates_idx:
            if any(abs(idx0-f[0]) < min_gap for f in found):
                continue
            for kind in types:
                col_w = _w(_outlier_col(np.arange(n), idx0, kind, alpha))
                tstat, _ = _tstat_last(np.column_stack([Xw_cur, col_w]), yw)
                if tstat is None:
                    continue
                if best is None or abs(tstat) > abs(best[2]):
                    best = (idx0, kind, tstat)
        if best is None or abs(best[2]) < cv:
            break
        found.append(best)
        col = _outlier_col(np.arange(n), best[0], best[1], alpha)
        extra_cols.append(col); extra_w.append(_w(col))

    # --- backward deletion: candidates that lost significance in the full model ---
    # --- обратный отсев: кандидаты, потерявшие значимость в полной модели ---
    dropped = []
    if backward and found:
        while found:
            Xw_full = np.column_stack([Xw_base] + extra_w)
            beta_f, *_ = np.linalg.lstsq(Xw_full, yw, rcond=None)
            resid = yw - Xw_full@beta_f
            dof = len(yw) - Xw_full.shape[1]
            if dof <= 0:
                break
            sigma2 = (resid@resid)/dof
            cov = sigma2*np.linalg.pinv(Xw_full.T@Xw_full)
            nb = Xw_base.shape[1]
            tstats = np.array([beta_f[nb+j]/np.sqrt(max(cov[nb+j, nb+j], 1e-12))
                               for j in range(len(found))])
            worst = int(np.argmin(np.abs(tstats)))
            if abs(tstats[worst]) >= cv:
                found = [(found[j][0], found[j][1], tstats[j]) for j in range(len(found))]
                break
            dropped.append((found[worst][0], found[worst][1], tstats[worst]))
            found.pop(worst); extra_cols.pop(worst); extra_w.pop(worst)

    # --- final fit: GLS on whitened data, effects returned in the original scale ---
    # --- итоговая оценка: GLS по отбеленным данным, эффекты -- в исходной шкале ---
    Xw_final = np.column_stack([Xw_base] + extra_w) if extra_w else Xw_base
    beta_final, *_ = np.linalg.lstsq(Xw_final, yw, rcond=None)
    X_final = np.column_stack([X_base] + extra_cols) if extra_cols else X_base
    n_base = X_base.shape[1]
    outlier_effect = np.zeros(n)
    for j in range(len(found)):
        outlier_effect += X_final[:, n_base+j]*beta_final[n_base+j]
    calendar_effect = np.zeros(n)
    for j in calendar_cols:
        calendar_effect += X_final[:, j]*beta_final[j]
    return {"outliers": found, "dropped_backward": dropped, "beta": beta_final,
            "y_clean": y - outlier_effect, "outlier_effect": outlier_effect,
            "calendar_effect": calendar_effect, "n_base_cols": n_base,
            "ar_coeffs": phi, "gls": bool(gls)}

def centered_ma_weights(length):
    """Weights of a centred moving average. For an even length (typically 12 for
    monthly data) the classic 2xN scheme is used: an N-point average, then the
    average of two neighbouring ones, so that the result is centred in time.

    [RU] Веса центрированной скользящей средней. Для чётной длины (обычно 12 для месячных
    данных) используется классическая схема 2xN: сначала N-точечная средняя, затем среднее
    двух соседних, чтобы результат был центрирован во времени.
    """
    if length%2==0:
        w=np.full(length+1,1.0/length); w[0]=w[-1]=0.5/length; return w
    return np.full(length,1.0/length)

def apply_symmetric_filter(x, weights):
    """Apply a symmetric filter. At the ends of the series, where a full symmetric
    window is unavailable, only the usable part of the weights is applied, renormalised
    to sum to one -- a plain illustration of the idea of "asymmetric end weights",
    which production X-11 implements far more carefully (optimal Musgrave weights
    rather than naive truncation).

    [RU] Применение симметричного фильтра. На краях ряда, где полного симметричного окна
    нет, используется только доступная часть весов, перенормированная к единице, --
    простая иллюстрация идеи «асимметричных весов на краях», которую боевой X-11 делает
    гораздо аккуратнее (оптимальные веса Масгрейва, а не наивное усечение).
    """
    n=len(x); m=len(weights)//2; out=np.full(n,np.nan)
    for i in range(n):
        lo,hi=i-m,i+m; wlo,whi=0,len(weights)
        if lo<0: wlo=-lo; lo=0
        if hi>n-1: whi=len(weights)-(hi-(n-1)); hi=n-1
        w=weights[wlo:whi]; w=w/w.sum(); out[i]=np.dot(x[lo:hi+1],w)
    return out

def naive_decompose(y, period=12):
    """Census Method I/II: a single seasonal factor per calendar month, fixed over the
    whole sample -- no smoothing along the time axis and no adaptation to a drifting
    seasonal pattern.

    [RU] Census Method I/II: один сезонный фактор на календарный месяц, фиксированный на всю
    выборку, -- без сглаживания вдоль оси времени и без адаптации к дрейфу сезонности.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    trend = apply_symmetric_filter(y, centered_ma_weights(period))
    si = y - trend
    seasonal = np.full(n, np.nan)
    for month in range(period):
        idx = np.arange(month, n, period)
        seasonal[idx] = np.nanmean(si[idx])       # <- a single number for the whole sample | <- одно число на весь период
    seasonal -= np.nanmean(seasonal)
    return {"trend": trend, "seasonal": seasonal, "sa": y - seasonal}

def henderson_weights(n):
    """Symmetric Henderson filter weights (Henderson, 1916), n terms, n odd. The formula
    minimises the sum of squared third differences subject to reproducing polynomials
    of degree <= 3 exactly (Ladiray & Quenneville, 2001).

    [RU] Веса симметричного фильтра Хендерсона (Henderson, 1916), n членов, n нечётное.
    Формула минимизирует сумму квадратов третьих разностей при условии точного
    воспроизведения полиномов степени <= 3 (Ladiray & Quenneville, 2001).
    """
    m=(n-1)//2; m1,m2,m3=(m+1)**2,(m+2)**2,(m+3)**2
    d=8*(m+2)*(m2-1)*(4*m2-1)*(4*m2-9)*(4*m2-25); w=np.zeros(n)
    for j in range(m+1):
        j2=j*j; v=(315*(m1-j2)*(m2-j2)*(m3-j2)*(3*m2-11*j2-16))/d
        w[m+j]=v; w[m-j]=v
    return w

def seasonal_moving_average(si, period=12, ma_len=3):
    """3xN seasonal moving average: for each calendar month separately, smooth its SI
    values ALONG THE YEAR AXIS with a window of length ma_len -- locally, rather than
    with one average over the whole sample as in Census Method I/II.

    [RU] Сезонная скользящая средняя 3xN: для каждого календарного месяца отдельно его
    SI-значения сглаживаются ВДОЛЬ ОСИ ЛЕТ окном длины ma_len -- локально, а не одним
    средним на всю выборку, как в Census Method I/II.
    """
    si=np.asarray(si,dtype=float); n=len(si); out=np.full(n,np.nan); base_w=np.full(ma_len,1.0/ma_len)
    for month in range(period):
        idx=np.arange(month,n,period); out[idx]=apply_symmetric_filter(si[idx],base_w)
    return out

def x11_lite(y, period=12, iterations=2, henderson_len=13):
    """A pedagogically simplified but substantive implementation of the classic iterative
    X-11 procedure (Shiskin, Young & Musgrave, 1965).
    NOT a bit-for-bit copy of the official program: real X-11 also rejects extreme
    values inside the seasonal step, uses optimal asymmetric Musgrave weights at the
    ends and runs more refinement iterations.
    What is preserved here is the key mechanism: the trend via a moving average /
    Henderson filter, and seasonality via LOCAL (year-by-year) smoothing.

    [RU] Педагогически упрощённая, но содержательная реализация классической итеративной
    процедуры X-11 (Shiskin, Young & Musgrave, 1965).
    НЕ побитовая копия официальной программы: в настоящем X-11 есть ещё отбраковка
    экстремальных значений внутри сезонного шага, оптимальные асимметричные веса Масгрейва
    на краях и больше итераций уточнения.
    Сохранена ключевая механика: тренд через скользящую среднюю / фильтр Хендерсона,
    сезонность через ЛОКАЛЬНОЕ (по годам) сглаживание.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    trend = apply_symmetric_filter(y, centered_ma_weights(period))
    hw = henderson_weights(henderson_len)
    for it in range(iterations):
        si = y - trend
        seasonal_raw = seasonal_moving_average(si, period=period, ma_len=3 if it == 0 else 5)
        # centring: a moving average over the calendar year, so that the 12
        # factors average to zero (additive model)
        # центрирование: скользящее среднее по календарному году, чтобы 12
        # факторов в среднем давали ноль (аддитивная модель)
        correction = pd.Series(seasonal_raw).rolling(period, center=True, min_periods=1).mean().values
        seasonal = seasonal_raw - correction
        sa = y - seasonal
        trend = apply_symmetric_filter(sa, hw)
    irregular = sa - trend
    return {"trend": trend, "seasonal": seasonal, "sa": sa, "irregular": irregular}

def extend_series(y, extra=12, n_harm=2, period=12):
    """"ARIMA-lite" extension of a series: a trend plus a couple of seasonal harmonics,
    fitted by OLS and extrapolated `extra` points forward. Real X-11-ARIMA / X-13 use a
    full seasonal ARIMA model for this; see the reference code in the notebook.

    [RU] «ARIMA-lite» продление ряда: тренд плюс пара сезонных гармоник, подобранные МНК и
    экстраполированные на `extra` точек вперёд. Настоящие X-11-ARIMA / X-13 используют для
    этого полноценную сезонную ARIMA-модель; см. референсный код в блокноте.
    """
    n = len(y)
    t = np.arange(n)
    cols = [np.ones(n), t]
    for k in range(1, n_harm + 1):
        cols += [np.sin(2*np.pi*k*t/period), np.cos(2*np.pi*k*t/period)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    t_f = np.arange(n, n + extra)
    cols_f = [np.ones(extra), t_f]
    for k in range(1, n_harm + 1):
        cols_f += [np.sin(2*np.pi*k*t_f/period), np.cos(2*np.pi*k*t_f/period)]
    y_f = np.column_stack(cols_f) @ beta
    return np.concatenate([y, y_f])

def build_bsm_matrices(period, var_level, var_slope, var_seas, var_obs):
    """Basic Structural Model (Harvey, 1989): local linear trend plus dummy seasonality
    (the sum of 12 consecutive seasonal states is zero).

    [RU] Basic Structural Model (Harvey, 1989): локальный линейный тренд плюс дамми-сезонность
    (сумма 12 последовательных сезонных состояний равна нулю).
    """
    k = period - 1
    dim = 2 + k
    T = np.zeros((dim, dim))
    T[0, 0] = 1; T[0, 1] = 1; T[1, 1] = 1
    T[2, 2:2+k] = -1.0
    if k > 1:
        T[3:2+k, 2:1+k] = np.eye(k - 1)
    Z = np.zeros(dim); Z[0] = 1; Z[2] = 1
    Q = np.zeros((dim, dim)); Q[0,0]=var_level; Q[1,1]=var_slope; Q[2,2]=var_seas
    return T, Z, Q, var_obs

def kalman_filter_smoother(y, T, Z, Q, H):
    """Standard linear Kalman filter forward plus Rauch-Tung-Striebel smoothing backward.

    [RU] Стандартный линейный фильтр Калмана вперёд плюс сглаживание Рауча-Тунга-Штрибеля назад.
    """
    n = len(y); dim = T.shape[0]
    x = np.zeros(dim); P = np.eye(dim) * 1e6
    x_filt = np.zeros((n, dim)); P_filt = np.zeros((n, dim, dim))
    for t in range(n):
        yhat = Z @ x
        F = Z @ P @ Z + H
        v = y[t] - yhat
        K = P @ Z / F
        x = x + K * v
        P = P - np.outer(K, Z) @ P
        x_filt[t], P_filt[t] = x, P
        x, P = T @ x, T @ P @ T.T + Q
    x_sm, P_sm = x_filt.copy(), P_filt.copy()
    for t in range(n - 2, -1, -1):
        Pp = T @ P_filt[t] @ T.T + Q
        J = P_filt[t] @ T.T @ np.linalg.pinv(Pp)
        x_sm[t] = x_filt[t] + J @ (x_sm[t+1] - T @ x_filt[t])
        P_sm[t] = P_filt[t] + J @ (P_sm[t+1] - Pp) @ J.T
    return x_sm, P_sm

def ucm_lite(y, period=12, var_ratios=(0.02, 0.0005, 0.01)):
    """Structural model (UCM / Basic Structural Model, Harvey 1989): local linear trend
    plus dummy seasonality, estimated with a Kalman filter and smoother.

    ON THE NAME. This function used to be called seats_lite, which was inaccurate. Real
    SEATS first fits ONE ARIMA model to the whole series and then CANONICALLY factorises
    its polynomial into trend / seasonal / irregular -- and that decomposition is unique.
    Here the model of each component is SPECIFIED explicitly rather than derived from
    someone else's model: this is exactly the family (UCM/BSM) that section IX.3.5
    contrasts with SEATS. They do share a lot -- both are model-based rather than
    filter-based, and both deliver standard errors for the components -- but calling
    this "SEATS" would be a substitution of concepts.

    var_ratios -- variance ratios (level, slope, seasonal) relative to the observation
    variance; set by hand, not estimated. fit_ucm_mle() estimates them from the data by
    maximum likelihood.

    [RU] Структурная модель (UCM / Basic Structural Model, Harvey 1989): локальный линейный
    тренд плюс дамми-сезонность, оценка фильтром Калмана со сглаживанием.

    ОБ ИМЕНИ. Раньше функция называлась seats_lite, и это было неточно. Настоящий SEATS
    сначала подбирает ОДНУ ARIMA-модель для всего ряда и затем КАНОНИЧЕСКИ факторизует её
    полином на тренд / сезонность / шум -- и это разложение единственно. Здесь же модель
    каждой компоненты ЗАДАНА явно, а не выведена из чужой модели: это ровно то семейство
    (UCM/BSM), которое раздел IX.3.5 противопоставляет SEATS. Общего у них много -- оба
    относятся к модельному, а не фильтровому подходу, и оба дают стандартные ошибки
    компонент, -- но называть это «SEATS» было бы подменой понятий.

    var_ratios -- отношения дисперсий (уровень, наклон, сезонность) к дисперсии наблюдения;
    заданы вручную, а не оценены. fit_ucm_mle() оценивает их по данным максимумом
    правдоподобия.
    """
    y = np.asarray(y, dtype=float)
    var_obs = np.var(np.diff(y, 2)) / 10
    vl, vb, vs = [r * var_obs for r in var_ratios]
    T, Z, Q, H = build_bsm_matrices(period, vl, vb, vs, var_obs)
    x_sm, P_sm = kalman_filter_smoother(y, T, Z, Q, H)
    return {"trend": x_sm[:,0], "trend_se": np.sqrt(P_sm[:,0,0]),
            "seasonal": x_sm[:,2], "seasonal_se": np.sqrt(P_sm[:,2,2]),
            "sa": y - x_sm[:,2], "irregular": y - x_sm[:,0] - x_sm[:,2]}

def _wrap_data_mini(vals, per_line=10):
    """Minimal version of _wrap_data used by the standalone SEATS comparison cell.

    [RU] Минимальная версия _wrap_data для отдельной ячейки сравнения с SEATS.
    """
    return "\n        ".join(" ".join(f"{v:.5f}" for v in vals[i:i+per_line])
                               for i in range(0, len(vals), per_line))

def _read_x13_table(workdir, name, suffix):
    """Read one of the tables x13as saves to disk (e.g. <name>.s12) as a plain array.

    [RU] Читает одну из таблиц, которые x13as сохраняет на диск (например <name>.s12), как массив.
    """
    path = os.path.join(workdir, f"{name}.{suffix}")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, names=["date", "value"])
    return df["value"].values

def preprocess(y, easter_reg, base_cols, outlier_cols):
    """Single preprocessing function for the JDemetra+ 2x2 matrix (section IX.2.7): the
    only difference between TRAMO-lite and regARIMA-lite is the Easter regressor passed
    in (a 6-day window versus an 8-day one).

    base_cols    -- list of base regressors (constant, trend, harmonics);
    outlier_cols -- list of outlier dummies (AO/LS/TC).
    Returns the series with the calendar and all outlier effects subtracted.

    [RU] Единая функция препроцессинга для матрицы 2x2 JDemetra+ (раздел IX.2.7): разница
    между TRAMO-lite и regARIMA-lite -- только в подаваемом пасхальном регрессоре
    (6-дневное окно против 8-дневного).

    base_cols    -- список базовых регрессоров (константа, тренд, гармоники);
    outlier_cols -- список дамми выбросов (AO/LS/TC).
    Возвращает ряд, из которого вычтены календарный и все выбросные эффекты.
    """
    Xp = np.column_stack(list(base_cols) + [easter_reg] + list(outlier_cols))
    b, *_ = np.linalg.lstsq(Xp, y, rcond=None)
    n_det = 1 + len(outlier_cols)                      # Easter + outliers | пасха + выбросы
    deterministic_effects = Xp[:, -n_det:] @ b[-n_det:]
    return y - deterministic_effects

def trig_seasonal_fit(y, t, period, K):
    """TBATS kernel: trigonometric regression with K harmonics at period `period`
    (the period may be NON-INTEGER).

    [RU] Ядро TBATS: тригонометрическая регрессия с K гармониками на периоде `period`
    (период может быть НЕЦЕЛЫМ).
    """
    cols = [np.ones(len(t)), t]
    for k in range(1, K+1):
        cols += [np.sin(2*np.pi*k*t/period), np.cos(2*np.pi*k*t/period)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return X[:, 2:] @ beta[2:]

def prophet_lite(y, t, n_changepoints=25, changepoint_range=0.8, alpha=10.0,
                  period=365.25, K=3):
    """Prophet-lite: a piecewise-linear trend whose breakpoints are selected by Lasso
    (followed by a "post-lasso" refit with plain OLS, to remove the shrinkage bias that
    L1 regularisation puts on the coefficients), plus trigonometric seasonality.

    [RU] Prophet-lite: кусочно-линейный тренд с изломами, отобранными Lasso (с последующей
    «post-lasso» переоценкой обычным МНК, чтобы убрать смещение коэффициентов от
    L1-регуляризации), плюс тригонометрическая сезонность.
    """
    n = len(y)
    candidates = np.linspace(30, int(changepoint_range*n), n_changepoints).astype(int)
    basis = np.column_stack([np.maximum(t-s, 0) for s in candidates])
    X_raw = np.column_stack([t, basis])
    Xs = StandardScaler().fit_transform(X_raw)
    lasso = Lasso(alpha=alpha, max_iter=50000)
    lasso.fit(Xs, y - y.mean())
    active = np.where(np.abs(lasso.coef_[1:]) > 1e-2)[0]
    found_cps = candidates[active]

    X_final = np.column_stack([np.ones(n), t, basis[:, active]])
    beta, *_ = np.linalg.lstsq(X_final, y, rcond=None)
    trend_fit = X_final @ beta

    resid = y - trend_fit
    cols = [np.ones(n)]
    for k in range(1, K+1):
        cols += [np.sin(2*np.pi*k*t/period), np.cos(2*np.pi*k*t/period)]
    Xseas = np.column_stack(cols)
    bseas, *_ = np.linalg.lstsq(Xseas, resid, rcond=None)
    seasonal_fit = Xseas @ bseas
    return {"trend": trend_fit, "seasonal": seasonal_fit, "changepoints": found_cps}

def build_bsm(period, vl, vb, vs, vobs):
    """Build the BSM state-space matrices from variance ratios -- the version used by
    fit_ucm_mle(), where the ratios are estimated rather than fixed.

    [RU] Строит матрицы пространства состояний BSM из отношений дисперсий -- версия для
    fit_ucm_mle(), где отношения оцениваются, а не фиксируются.
    """
    k = period - 1; dim = 2 + k
    T = np.zeros((dim, dim)); T[0,0]=1; T[0,1]=1; T[1,1]=1
    T[2, 2:2+k] = -1.0
    if k > 1: T[3:2+k, 2:1+k] = np.eye(k - 1)
    Z = np.zeros(dim); Z[0]=1; Z[2]=1
    Q = np.zeros((dim,dim)); Q[0,0]=vl; Q[1,1]=vb; Q[2,2]=vs
    return T, Z, Q, vobs

def kf_loglik_and_smooth(y, T, Z, Q, H, want_smooth=False):
    """Kalman filter that returns both the log-likelihood (for the optimiser) and the
    smoothed states (for the final decomposition).

    [RU] Фильтр Калмана, возвращающий и логарифм правдоподобия (для оптимизатора), и
    сглаженные состояния (для итогового разложения).
    """
    n = len(y); dim = T.shape[0]
    x = np.zeros(dim); P = np.eye(dim)*1e6
    xf = np.zeros((n,dim)); Pf = np.zeros((n,dim,dim)); ll = 0.0
    for tt in range(n):
        yhat = Z@x; F = Z@P@Z+H
        if F <= 1e-10: return (-1e10, None) if not want_smooth else (-1e10, None)
        v = y[tt]-yhat; K = P@Z/F
        x = x+K*v; P = P-np.outer(K,Z)@P
        xf[tt], Pf[tt] = x, P
        ll += -0.5*(np.log(2*np.pi*F)+v**2/F)
        x, P = T@x, T@P@T.T+Q
    if not want_smooth:
        return ll, None
    xs = xf.copy()
    for tt in range(n-2, -1, -1):
        Pp = T@Pf[tt]@T.T+Q
        J = Pf[tt]@T.T@np.linalg.pinv(Pp)
        xs[tt] = xf[tt] + J@(xs[tt+1]-T@xf[tt])
    return ll, xs

def fit_ucm_mle(y, period=12):
    """Estimate the variance RATIOS of the components (level / slope / seasonal, relative
    to the observation variance) by maximum likelihood, instead of setting them by eye
    as ucm_lite() does.

    [RU] Оценивает ОТНОШЕНИЯ дисперсий компонент (уровень / наклон / сезонность к дисперсии
    наблюдения) максимизацией правдоподобия, вместо ручной установки «на глаз», как в
    ucm_lite().
    """
    y = np.asarray(y, dtype=float)
    var_obs0 = np.var(np.diff(y, 2))/10
    def negloglik(logp):
        rl, rb, rs = np.exp(logp)
        T,Z,Q,H = build_bsm(period, rl*var_obs0, rb*var_obs0, rs*var_obs0, var_obs0)
        ll, _ = kf_loglik_and_smooth(y, T, Z, Q, H)
        return -ll
    res = minimize(negloglik, np.log([0.02, 0.0005, 0.01]), method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4})
    ratios = np.exp(res.x)
    T,Z,Q,H = build_bsm(period, *(ratios*var_obs0), var_obs0)
    _, xs = kf_loglik_and_smooth(y, T, Z, Q, H, want_smooth=True)
    return ratios, xs, res

def haar_dwt(x, levels):
    """Multi-level Haar decomposition (discrete wavelet transform from scratch).

    [RU] Многоуровневое разложение Хаара (дискретное вейвлет-преобразование с нуля).
    """
    approx = np.asarray(x, dtype=float).copy()
    details = []
    for lvl in range(levels):
        n = len(approx)
        if n % 2 == 1:
            approx = approx[:-1]; n -= 1
        a = (approx[0::2] + approx[1::2]) / np.sqrt(2)
        d = (approx[0::2] - approx[1::2]) / np.sqrt(2)
        details.append(d); approx = a
    return details, approx

def three_month_ma_saar(sa_series, mode="level_ma"):
    """Three-month moving average of the SEASONALLY ADJUSTED LEVEL and its annualised
    growth rate. Three conventions are in use -- see the table in X.1:

    mode="level_ma" -- growth of the 3-month average over the previous such average, ^4
                       (double smoothing; the default in this notebook);
    mode="level_3m" -- level to level three months apart, ^4 (the classic 3-month
                       annualised rate used by BLS/the Fed for CPI and PCE);
    mode="rate_avg" -- the average of the MONTHLY seasonally adjusted growth rates over
                       three months, annualised to the 12th power. This is what the Bank
                       of Russia calls "3mma" -- use this mode when the figure has to be
                       comparable with CBR publications.

    The input is a LEVEL, not a growth rate -- see VI.2.

    [RU] Трёхмесячная скользящая средняя СЕЗОННО СКОРРЕКТИРОВАННОГО УРОВНЯ и её
    аннуализированный темп роста. В ходу три конвенции -- см. таблицу в X.1:

    mode="level_ma" -- рост 3-месячной средней к предыдущей такой же средней, ^4
                       (двойное сглаживание; вариант по умолчанию в этом блокноте);
    mode="level_3m" -- уровень к уровню через три месяца, ^4 (классический 3-month
                       annualized rate BLS/ФРС для ИПЦ и PCE);
    mode="rate_avg" -- среднее из МЕСЯЧНЫХ сезонно скорректированных приростов за три месяца,
                       аннуализированное в 12-й степени. Именно это Банк России называет
                       «3mma» -- используйте этот режим, если цифра должна быть сопоставима с
                       публикациями ЦБ РФ.

    На вход подаётся УРОВЕНЬ, а не темп -- см. VI.2.
    """
    sa = pd.Series(sa_series)
    ma3 = sa.rolling(3).mean()
    if mode == "level_ma":
        saar = ((ma3/ma3.shift(3))**4 - 1)*100
    elif mode == "level_3m":
        saar = ((sa/sa.shift(3))**4 - 1)*100
    elif mode == "rate_avg":
        g = sa/sa.shift(1) - 1
        saar = ((1 + g.rolling(3).mean())**12 - 1)*100
    else:
        raise ValueError(tr("mode должен быть 'level_ma', 'level_3m' или 'rate_avg'",
                            "mode must be 'level_ma', 'level_3m' or 'rate_avg'"))
    return ma3, saar

def forecast_deterministic(y, dates, horizon, period=12, n_harm=2, ar1_decay=True,
                            n_sim=0, seed=0, freq="ME"):
    """Forecast = extrapolation of the fitted deterministic block (trend + seasonal
    harmonics, OLS) plus the last residual carried forward with geometric decay
    (AR(1)-like persistence of a short-run shock).

    The forecast error variance has TWO parts (see X.2.1):
      Var(h) = Var(resid)*(1 - rho^(2h))        -- future noise, flattens out;
             + sigma^2 * x_h' (X'X)^(-1) x_h    -- uncertainty of the OLS coefficients
                                                   themselves, growing with the horizon
                                                   as the trend is extrapolated.
    Over a one-to-two-year horizon the second part usually dominates; without it the
    interval is understated.

    n_sim>0 additionally returns n_sim simulated paths (`paths`): the coefficients are
    drawn from N(beta, sigma^2 (X'X)^(-1)) and the noise is generated as an AR(1) process.
    Paths are needed wherever the quantity of interest is a NON-LINEAR function of the
    forecast (a growth rate such as SAAR): stretching the bounds of a level interval
    through a non-linear formula is not a confidence interval, percentiles of simulated
    paths are.

    Limitation: the model postulates a DETERMINISTIC trend and therefore has no unit root.
    A real macro level series has a stochastic trend, and its true forecast variance grows
    faster. For a published forecast beyond 3-6 months take the standard errors from
    regARIMA (section XI.6).

    [RU] Прогноз = экстраполяция подобранного детерминированного блока (тренд + сезонные
    гармоники, МНК) плюс последний остаток, переносимый вперёд с геометрическим затуханием
    (AR(1)-подобная персистентность краткосрочного шока).

    Дисперсия ошибки прогноза складывается из ДВУХ частей (см. X.2.1):
      Var(h) = Var(resid)*(1 - rho^(2h))        -- будущий шум, выходит на плато;
             + sigma^2 * x_h' (X'X)^(-1) x_h    -- неопределённость самих МНК-коэффициентов,
                                                   растёт с горизонтом по мере экстраполяции
                                                   тренда.
    На горизонте в год-два вторая часть обычно доминирует; без неё интервал занижен.

    n_sim>0 дополнительно возвращает n_sim смоделированных траекторий (`paths`):
    коэффициенты разыгрываются из N(beta, sigma^2 (X'X)^(-1)), шум генерируется как
    AR(1)-процесс. Траектории нужны, когда интересующая величина -- НЕЛИНЕЙНАЯ функция
    прогноза (темп роста вроде SAAR): протягивание границ интервала уровня через нелинейную
    формулу -- не доверительный интервал, а процентили смоделированных траекторий -- да.

    Ограничение: модель постулирует ДЕТЕРМИНИРОВАННЫЙ тренд, поэтому единичного корня у неё
    нет. У реального макроряда уровня тренд стохастический, и настоящая дисперсия прогноза
    растёт быстрее. Для публикуемого прогноза дальше 3-6 месяцев берите стандартные ошибки
    из regARIMA (раздел XI.6).
    """
    y = np.asarray(y, dtype=float); n = len(y)
    cols, t = _harmonic_trend_cols(n, period, n_harm)
    X = np.column_stack(cols); beta,*_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X@beta
    rho = np.clip(np.corrcoef(resid[1:], resid[:-1])[0,1], -0.95, 0.95) if ar1_decay else 0.0
    var_resid = np.var(resid)
    k_par = X.shape[1]
    sigma2 = float(resid@resid)/max(n - k_par, 1)
    XtX_inv = np.linalg.pinv(X.T@X)
    t_f = np.arange(n, n+horizon)
    cols_f = [np.ones(horizon), t_f]
    for k in range(1, n_harm+1): cols_f += [np.sin(2*np.pi*k*t_f/period), np.cos(2*np.pi*k*t_f/period)]
    X_f = np.column_stack(cols_f)
    forecast = X_f@beta + resid[-1]*(rho**np.arange(1, horizon+1))
    h = np.arange(1, horizon+1)
    var_noise = var_resid*(1 - rho**(2*h))
    var_param = sigma2*np.einsum("ij,jk,ik->i", X_f, XtX_inv, X_f)
    se = np.sqrt(var_noise + var_param)

    paths = None
    if n_sim:
        rng = np.random.default_rng(seed)
        cov = sigma2*XtX_inv
        cov = (cov + cov.T)/2 + 1e-12*np.eye(k_par)
        try:
            Lc = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(cov); Lc = V@np.diag(np.sqrt(np.clip(w, 0, None)))
        betas = beta + (Lc@rng.standard_normal((k_par, n_sim))).T
        sigma_e = np.sqrt(max(var_resid*(1 - rho**2), 1e-12))
        u = np.empty((n_sim, horizon)); prev = np.full(n_sim, resid[-1])
        for hh in range(horizon):
            prev = rho*prev + rng.normal(0.0, sigma_e, n_sim)
            u[:, hh] = prev
        paths = betas@X_f.T + u

    future_dates = pd.date_range(pd.Timestamp(dates[-1]), periods=horizon+1, freq=freq)[1:]
    return {"forecast": forecast, "se": se, "se_noise": np.sqrt(var_noise),
            "se_param": np.sqrt(var_param), "lo": forecast - 1.96*se, "hi": forecast + 1.96*se,
            "paths": paths, "dates": future_dates, "beta": beta, "rho": rho}

def backtest_forecast(y, dates, holdout=12, period=12, n_harm=2, inverse=None):
    """Out-of-sample check of forecast quality: fit the model WITHOUT the last `holdout`
    points, forecast exactly those, compare with the actuals -- RMSE/MAPE.

    inverse -- the inverse transformation (e.g. np.exp) when the model was fitted on a
    logarithmic scale: errors must be measured in the units the user reads.

    Caveat: if what is passed in is an ALREADY seasonally adjusted series built on the
    WHOLE sample, then information about the future leaks into the holdout (symmetric
    filters use points to the right). Such a backtest measures the quality of
    extrapolating the SA series, not of the end-to-end pipeline; a fully honest version
    would re-run the adjustment on the truncated sample.

    [RU] Out-of-sample проверка качества прогноза: модель подбирается БЕЗ последних `holdout`
    точек, прогнозируются именно они, сравнение с фактом -- RMSE/MAPE.

    inverse -- обратное преобразование (например np.exp), если модель подбиралась в
    логарифмической шкале: ошибки нужно мерить в тех единицах, которые читает пользователь.

    Оговорка: если на вход подан УЖЕ сезонно скорректированный ряд, построенный по ВСЕЙ
    выборке, в holdout просачивается информация о будущем (симметричные фильтры используют
    точки справа). Такой бэктест меряет качество экстраполяции SA-ряда, а не сквозного
    пайплайна; полностью честная версия пересчитала бы корректировку на обрезанной выборке.
    """
    y = np.asarray(y, dtype=float)
    fc = forecast_deterministic(y[:-holdout], dates[:-holdout], holdout, period, n_harm)
    y_true, y_pred = y[-holdout:], fc["forecast"]
    if inverse is not None:
        y_true, y_pred = inverse(y_true), inverse(y_pred)
    err = y_true - y_pred
    return {"rmse": np.sqrt(np.mean(err**2)),
            "mape": np.mean(np.abs(err/np.where(y_true!=0, y_true, np.nan)))*100,
            "y_true": y_true, "y_pred": y_pred}

def _wrap_data(vals, per_line=10, fmt=".5f"):
    """Data in an X-13 spec file cannot be written on a single line -- the format has a
    hard line-length limit of 133 characters. Split into lines of `per_line` values.

    [RU] Данные в spec-файле X-13 нельзя писать одной строкой: у формата жёсткое ограничение
    длины строки в 133 символа. Разбиваем на строки по `per_line` значений.
    """
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(" ".join(f"{v:{fmt}}" for v in vals[i:i+per_line]))
    return "\n        ".join(lines)

def _resolve_x13_binary(path):
    """Resolve the path to the REAL x13as executable: if a folder is given, look inside
    it for the binary under the usual names; if the file lacks the execute bit, try to
    set it. Raises a readable error instead of a bare PermissionError/OSError when
    nothing works -- in practice this is how PermissionError on '/usr/local/bin/x13as/bin'
    turned out to be resolved: that path was a FOLDER, not the file itself.

    [RU] Приводит путь к РЕАЛЬНОМУ исполняемому файлу x13as: если передана папка -- ищет внутри
    бинарник под обычными именами; если у файла нет бита на выполнение -- пытается его
    выставить. Бросает понятную ошибку вместо сырых PermissionError/OSError, если ничего не
    подошло. Именно так на практике разрешилась ошибка PermissionError на
    '/usr/local/bin/x13as/bin': этот путь оказался ПАПКОЙ, а не самим файлом.
    """
    candidates_names = ["x13as", "x13as.exe", "x13ashtml", "x13as_html"]

    def _make_executable(p):
        try:
            st = os.stat(p)
            os.chmod(p, st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        except PermissionError:
            pass

    if os.path.isdir(path):
        found = None
        for name in candidates_names:
            cand = os.path.join(path, name)
            if os.path.isfile(cand):
                found = cand
                break
        if found is None:
            raise FileNotFoundError(tr(
                f"'{path}' -- это папка, но внутри неё не нашёлся ни один из файлов "
                f"{candidates_names}. Выполните 'ls -la {path}' и укажите точный путь "
                f"к исполняемому файлу явно (X13_BIN = '...').",
                f"'{path}' is a folder, but none of {candidates_names} was found inside it. "
                f"Run 'ls -la {path}' and set the exact path to the executable "
                f"(X13_BIN = '...')."))
        path = found

    if not os.path.isfile(path):
        raise FileNotFoundError(tr(f"Файл '{path}' не найден. Проверьте путь X13_BIN.",
                                   f"File '{path}' not found. Check the X13_BIN path."))

    if not os.access(path, os.X_OK):
        _make_executable(path)
        if not os.access(path, os.X_OK):
            raise PermissionError(tr(
                f"Файл '{path}' найден, но нет прав на выполнение, и выставить их "
                f"автоматически не удалось. Выполните в терминале:\n"
                f"    chmod +x \"{path}\"\n"
                f"Если не поможет -- проверьте владельца файла:\n"
                f"    sudo chown $(whoami) \"{path}\"",
                f"File '{path}' exists but is not executable, and the permission could not "
                f"be set automatically. Run in a terminal:\n"
                f"    chmod +x \"{path}\"\n"
                f"If that does not help, check the file owner:\n"
                f"    sudo chown $(whoami) \"{path}\""))
    return path

def run_x13(spec_text, workdir, name="run"):
    """Write spec_text to <name>.spc and run x13as (X13_BIN may point either at the file
    itself or at the folder holding it -- this is resolved automatically).
    Returns the contents of the resulting .out (the human-readable report) and .err.

    [RU] Записывает spec_text в <name>.spc и запускает x13as (X13_BIN может указывать на сам
    файл или на папку с ним -- разбирается автоматически).
    Возвращает содержимое полученных .out (человекочитаемый отчёт) и .err.
    """
    binary = _resolve_x13_binary(X13_BIN)
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, name + ".spc"), "w") as f:
        f.write(spec_text)
    try:
        proc = subprocess.run([binary, name], cwd=workdir, capture_output=True, text=True)
    except OSError as e:
        raise OSError(tr(
            f"Не удалось запустить '{binary}' ({e}). Частые причины:\n"
            f"  -- бинарник собран под другую ОС/архитектуру (Linux vs macOS, x86_64 vs arm64) "
            f"-- см. https://github.com/x13org/x13prebuilt или R-пакет x13binary;\n"
            f"  -- файл повреждён при скачивании (проверьте размер -- должен быть несколько МБ);\n"
            f"  -- нет прав на выполнение (chmod +x).",
            f"Could not run '{binary}' ({e}). Common causes:\n"
            f"  -- the binary was built for another OS/architecture (Linux vs macOS, x86_64 vs arm64) "
            f"-- see https://github.com/x13org/x13prebuilt or the R package x13binary;\n"
            f"  -- the file was corrupted during download (check the size -- it should be several MB);\n"
            f"  -- no execute permission (chmod +x).")) from e
    out_p, err_p = os.path.join(workdir, name+".out"), os.path.join(workdir, name+".err")
    out = open(out_p, encoding="utf-8", errors="ignore").read() if os.path.exists(out_p) else ""
    err = open(err_p, encoding="utf-8", errors="ignore").read() if os.path.exists(err_p) else ""
    if not out and proc.returncode != 0:
        err += f"\n[subprocess stderr]\n{proc.stderr}\n[subprocess stdout]\n{proc.stdout}"
    return out, err

def parse_outliers(out_text):
    """Extract the 'Automatically Identified Outliers' table -> the list of detected
    outliers with type, date, estimated effect and t-statistic.

    [RU] Достаёт таблицу 'Automatically Identified Outliers' -> список найденных выбросов с
    типом, датой, оценённым эффектом и t-статистикой.
    """
    m = re.search(r"Automatically Identified Outliers\s*\n(.*?)\n\s*-{5,}", out_text, re.S)
    if not m: return []
    rows = []
    for line in m.group(1).strip().splitlines():
        mm = re.match(r"\s*(AO|LS|TC)(\d{4})\.(\w{3})\s+([\-\d.]+)\s+([\-\d.]+)\s+([\-\d.]+)", line)
        if mm:
            kind, year, mon, coef, se, t = mm.groups()
            rows.append({"type": kind, "date": f"{year}-{mon}", "coef": float(coef), "t": float(t)})
    return rows

def parse_arima_model(out_text):
    """Extract the order (p d q)(P D Q) of the final ARIMA model.

    [RU] Достаёт порядок (p d q)(P D Q) итоговой ARIMA-модели.
    """
    m = re.search(r"ARIMA Model:\s*\(([\d\s]+)\)\(([\d\s]+)\)", out_text)
    if not m: return None
    return tuple(int(x) for x in m.group(1).split()), tuple(int(x) for x in m.group(2).split())

def parse_likelihood(out_text):
    """Extract AIC/AICC/BIC from the last 'Likelihood Statistics' block. AICC is what
    different ARIMA specifications are compared on (smaller is better) -- but only
    within the same order of differencing, see XI.5.1.

    [RU] Достаёт AIC/AICC/BIC из последнего блока 'Likelihood Statistics'. По AICC сравнивают
    разные спецификации ARIMA (меньше -- лучше), но только при одинаковом порядке
    дифференцирования, см. XI.5.1.
    """
    blocks = re.findall(r"Likelihood Statistics\s*-+\s*(.*?)-{5,}", out_text, re.S)
    if not blocks: return {}
    block = blocks[-1]; out = {}
    for key, pat in [("aic", r"^ AIC\s+([\-\d.]+)"), ("aicc", r"AICC \(F-corrected-AIC\)\s+([\-\d.]+)"),
                      ("bic", r"BIC\s+([\-\d.]+)"), ("loglik", r"Log likelihood \(L\)\s+([\-\d.]+)")]:
        mm = re.search(pat, block, re.M)
        if mm: out[key] = float(mm.group(1))
    return out

def parse_forecasts(out_text):
    """Extract the forecast table from an x13as report.

    X-13 prints it in two different shapes:

    * `Forecasts and Standard Errors` -- columns "date, forecast, standard error".
      This is what the report looks like under `transform{ function=none }`.
    * `Forecasts on the Original Scale` / the confidence-interval table -- columns
      "date and three numbers". This is what the report looks like once a logarithmic
      transformation has been applied: on the original scale the interval is asymmetric,
      so the bounds are printed instead of a standard error. That case became the norm
      once `transform{ function=auto }` was made the default.

    Under a logarithmic transformation the report contains BOTH tables: one on the
    original scale and one on the transformed (i.e. logarithmic) scale. Picking the
    wrong one yields a forecast an order of magnitude below the series, so the tables
    are searched by header priority and headers containing "transformed" are rejected.

    In the three-column case the column order is not guessed but inferred from the
    magnitudes: the smallest is the lower bound, the middle one the point forecast, the
    largest the upper bound; the standard error is reconstructed as (hi - lo) / (2*1.96)
    and flagged with se_approx=True.

    Returns a dict with keys dates, forecast, se, lo, hi, se_approx, header -- or None
    if the report has no such table (usually meaning the run did not converge; inspect
    result["x13_out"] and result["x13_err"]).

    [RU] Достаёт таблицу прогноза из отчёта x13as.

    X-13 печатает её в двух видах:

    * `Forecasts and Standard Errors` -- колонки «дата, прогноз, стандартная ошибка».
      Так выглядит отчёт при `transform{ function=none }`.
    * `Forecasts on the Original Scale` / таблица доверительных интервалов -- колонки
      «дата и три числа». Так выглядит отчёт при логарифмическом преобразовании: на
      исходной шкале интервал несимметричен, поэтому печатаются границы, а не стандартная
      ошибка. Этот случай стал обычным после того, как по умолчанию включился
      `transform{ function=auto }`.

    При логарифмическом преобразовании в отчёте ОБЕ таблицы: на исходной шкале и на
    преобразованной (то есть в логарифмах). Взять не ту -- значит получить прогноз на порядок
    ниже ряда, поэтому таблицы ищутся по приоритету заголовка, а заголовки со словом
    «transformed» отбрасываются.

    В трёхколоночном случае порядок колонок не угадывается, а определяется по величине:
    меньшее -- нижняя граница, среднее -- точечный прогноз, большее -- верхняя; стандартная
    ошибка восстанавливается как (hi - lo) / (2*1.96) и помечается флагом se_approx=True.

    Возвращает словарь с ключами dates, forecast, se, lo, hi, se_approx, header -- или None,
    если такой таблицы в отчёте нет (обычно это значит, что прогон не сошёлся; смотрите
    result["x13_out"] и result["x13_err"]).
    """
    row_re = re.compile(r"^\s*(\d{4})[.](\w{3})\s+([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?\s*$")
    lines = out_text.splitlines()

    def _rows_after(k):
        rows, gap = [], 0
        for ln in lines[k+1: k+500]:
            m = row_re.match(ln)
            if m:
                rows.append(m.groups()); gap = 0
            elif rows:
                gap += 1
                if gap > 3:
                    break
        return rows

    # Order matters: under transform{function=log|auto} X-13 prints TWO forecast tables --
    # one on the original scale and one on the transformed (logarithmic) scale. Take the
    # first, or the forecast lands an order of magnitude below the series. Hence tables are
    # searched by header priority, not by length, and "transformed" headers are rejected.
    # Порядок важен: при transform{function=log|auto} X-13 печатает ДВЕ таблицы прогноза --
    # на исходной шкале и на преобразованной (в логарифмах). Брать нужно первую, иначе
    # прогноз окажется на порядок ниже ряда. Поэтому таблицы перебираются по приоритету,
    # а не по длине, и заголовки с "transformed" отбрасываются в первых двух проходах.
    tiers = (
        lambda low: "original scale" in low,
        lambda low: "confidence interval" in low and "transformed" not in low,
        lambda low: "standard error" in low and "transformed" not in low,
        lambda low: "transformed" not in low,
        lambda low: True,                      # last resort: anything mentioning "forecast" | последняя попытка: что угодно с "forecast"
    )
    best, best_header = [], ""
    for accept in tiers:
        for k, ln in enumerate(lines):
            low = ln.lower()
            if "forecast" not in low or not accept(low):
                continue
            rows = _rows_after(k)
            if rows:
                best, best_header = rows, ln.strip()
                break
        if best:
            break
    if not best:
        return None

    dates, c = [], []
    for year, mon, a, b, third in best:
        dates.append(pd.Period(f"{year}-{mon}").to_timestamp(how="end").normalize())
        c.append([float(a), float(b), np.nan if third is None else float(third)])
    c = np.asarray(c, dtype=float)
    if np.isnan(c[:, 2]).all():
        fc, se = c[:, 0], np.abs(c[:, 1])
        lo, hi = fc - 1.96*se, fc + 1.96*se
        approx = False
    else:
        srt = np.sort(c, axis=1)
        lo, fc, hi = srt[:, 0], srt[:, 1], srt[:, 2]
        se = (hi - lo)/(2*1.96)
        approx = True
    return {"dates": pd.DatetimeIndex(dates), "forecast": fc, "se": se,
            "lo": lo, "hi": hi, "se_approx": approx, "header": best_header}

def read_d11(workdir, name):
    """Read the <name>.d11 file -- the finished seasonally adjusted series that
    x11{save=(d11)} writes to disk.

    [RU] Читает файл <name>.d11 -- готовый сезонно скорректированный ряд, который
    x11{save=(d11)} записывает на диск.
    """
    path = os.path.join(workdir, f"{name}.d11")
    if not os.path.exists(path): return None
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, names=["date","value"])
    return df["value"].values

def trading_day_regressor_x13demo(dates):
    """Deviation of the number of weekdays (Mon-Fri) in the month from its expected
    average (5/7 of the month length) -- our OWN calendar regressor, the one fed to the
    real binary as a user regressor in section XI.4.

    [RU] Отклонение числа будних дней (Пн-Пт) в месяце от ожидаемого среднего (5/7 длины
    месяца) -- наш СОБСТВЕННЫЙ календарный регрессор, тот самый, что подаётся в настоящий
    бинарник как user-регрессор в разделе XI.4.
    """
    reg = np.zeros(len(dates))
    for i, d in enumerate(dates):
        days_in_month = pd.Period(d, freq="M").days_in_month
        month_days = pd.date_range(d.replace(day=1), periods=days_in_month, freq="D")
        reg[i] = (month_days.weekday < 5).sum() - (5/7)*days_in_month
    return reg

def fmt_order(o):
    """Format an ARIMA order tuple for printing inside an X-13 spec file.

    [RU] Форматирует кортеж порядка ARIMA для записи в spec-файл X-13.
    """
    return " ".join(map(str, o))

def make_spec(y, arima_order=None):
    """Assemble an X-13 spec file from a series, an ARIMA block and the usual options --
    used by the ARIMA grid search in section XI.5.

    [RU] Собирает spec-файл X-13 из ряда, блока ARIMA и обычных опций -- используется перебором
    моделей ARIMA в разделе XI.5.
    """
    arima_block = (f"arima{{ model=({fmt_order(arima_order[0])})({fmt_order(arima_order[1])}) }}"
                   if arima_order else "automdl{ }")
    return f"""series{{
    title="grid"
    start=2005.1
    period=12
    data=({_wrap_data(y)})
}}
transform{{ function=none }}
regression{{ aictest=(td easter) }}
outlier{{ types=(AO LS TC) }}
{arima_block}
estimate{{ }}
"""

def parse_spectral_peaks(out_text):
    """Extract the spectral peak table (S1..S6 seasonal frequencies and TD) from the
    `spectrum{}` output, together with the significance markers X-13 prints.

    [RU] Достаёт таблицу спектральных пиков (сезонные частоты S1..S6 и TD) из вывода
    `spectrum{}` вместе с отметками значимости, которые печатает X-13.
    """
    m = re.search(r"Peak probabilities for Tukey spectrum estimator\s*\n\s*Spectrum estimated from (\S+) to (\S+)\.\s*\n(.*?)\n\s{2,}-{8,}", out_text, re.S)
    if not m: return None
    lines = [l for l in m.group(3).splitlines() if l.strip()]
    header = re.split(r"\s{2,}", lines[0].strip())
    rows = {}
    for line in lines[2:]:
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) >= 2:
            rows[parts[0]] = parts[1:]
    return {"span": (m.group(1), m.group(2)), "header": header, "rows": rows}

def parse_residual_seasonality_ftest(out_text):
    """Extract the F-test for residual seasonality that X-13 prints for the seasonally
    adjusted series.

    [RU] Достаёт F-тест на остаточную сезонность, который X-13 печатает для сезонно
    скорректированного ряда.
    """
    results = []
    for m in re.finditer(r"(No evidence|Evidence) of residual seasonality in the (entire series|last 3 years) at the\s*\n\s*(\d+) per cent level\.(?:\s*F\s*=\s*([\d.]+))?", out_text):
        verdict, scope, pct, fval = m.groups()
        results.append({"scope": scope, "pct": pct, "verdict": verdict, "F": fval})
    return results

def parse_np_statistic(out_text):
    """Extract the non-parametric seasonality statistic from the X-13 report.

    [RU] Достаёт непараметрическую статистику сезонности из отчёта X-13.
    """
    m = re.search(r"NP statistic for residual seasonality \(Full series\)\s*\n\s*Residual Seasonality\?\s*\n(.*?)\n(.*?)\n", out_text)
    res = {}
    if m:
        for line in [m.group(1), m.group(2)]:
            mm = re.match(r"\s*(.+?)\s{2,}(yes|no)\s*$", line, re.I)
            if mm: res[mm.group(1).strip()] = mm.group(2)
    return res

def periodogram_own(x):
    """Our own periodogram, built the same way as inside evaluate_seasonality, so the
    home-made and the binary-produced diagnostics can be placed side by side.

    [RU] Наша собственная периодограмма, построенная так же, как внутри evaluate_seasonality,
    чтобы самодельную и бинарную диагностику можно было поставить рядом.
    """
    x = np.asarray(x, dtype=float); x = x - np.mean(x)
    fft = np.fft.rfft(x)
    power = (np.abs(fft)**2) / len(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0) * 12   # cycles per year for monthly data | циклы/год для месячных данных
    return freqs, 10*np.log10(power + 1e-12)

# Historical alias: section XII.1 used to define a second, condensed copy of the same
# procedure. It is now one and the same function.
# Исторический псевдоним: раньше в разделе XII.1 определялась вторая, сжатая копия
# той же процедуры. Теперь это одна и та же функция.
x11_lite_wrap = x11_lite
def seasonal_adjustment_pipeline(y, dates, method="x11", period=12, transform=None,
        calendar_adjust=True, easter_window=8, easter_tradition=None, trading_day=True,
        working_days_calendar=None,
        auto_outliers=True, outliers=None, outlier_cv=3.5, outlier_types=("AO","LS","TC"),
        outlier_skip_edge=6, outlier_min_gap=3, gls=None, ar_order=1,
        n_harm=2, forecast_horizon=12, backtest_holdout=12, saar_mode="level_ma",
        show_chart=True, dark=False, method_name=None):
    """The single seasonal adjustment pipeline -- every stage of the lecture in one function.

    transform          -- "auto" | "log" | "none" (defaults to DEFAULT_TRANSFORM). All
                          internal work happens in the working scale; sa / trend /
                          forecast / 3MMA SAAR come back in the ORIGINAL units. See VI.4.
    easter_tradition   -- "orthodox" | "catholic" (defaults to DEFAULT_EASTER_TRADITION).
    calendar_adjust    -- estimate AND SUBTRACT the calendar effects (Easter, trading days).
    outlier_skip_edge  -- how many points at each end to exclude from outlier search.
                          Set 0-2 for current monitoring, see VIII.1.
    gls / ar_order     -- whether to whiten the residual before searching for outliers and
                          with which AR order (None => DEFAULT_GLS). Without whitening the
                          t-statistics are inflated on an autocorrelated residual -- VIII.2.
    saar_mode          -- 3MMA SAAR convention: "level_ma" | "level_3m" | "rate_avg" (CBR).
    outliers=[(idx,'AO'), ...] switches outlier handling to MANUAL mode.

    Returns one dict holding every intermediate and final result.

    [RU] Единый пайплайн сезонной корректировки -- все стадии лекции в одной функции.

    transform          -- "auto" | "log" | "none" (по умолчанию DEFAULT_TRANSFORM). Вся
                          внутренняя работа идёт в рабочей шкале; sa / trend / прогноз /
                          3MMA SAAR возвращаются в ИСХОДНЫХ единицах. См. VI.4.
    easter_tradition   -- "orthodox" | "catholic" (по умолчанию DEFAULT_EASTER_TRADITION).
    calendar_adjust    -- оценить И ВЫЧЕСТЬ календарные эффекты (Пасха, торговые дни).
    outlier_skip_edge  -- сколько точек у каждого края исключить из поиска выбросов.
                          Для оперативного мониторинга ставьте 0-2, см. VIII.1.
    gls / ar_order     -- отбеливать ли остаток перед поиском выбросов и каким порядком AR
                          (None => DEFAULT_GLS). Без отбеливания t-статистики на
                          автокоррелированном остатке завышены -- VIII.2.
    saar_mode          -- конвенция 3MMA SAAR: "level_ma" | "level_3m" | "rate_avg" (ЦБ РФ).
    outliers=[(idx,'AO'), ...] переводит обработку выбросов в РУЧНОЙ режим.

    Возвращает единый словарь со всеми промежуточными и итоговыми результатами.
    """
    y = np.asarray(y, dtype=float); n = len(y); dates = pd.DatetimeIndex(dates)
    method_name = method_name or _resolve_method(method).upper()
    transform = DEFAULT_TRANSFORM if transform is None else transform
    easter_tradition = DEFAULT_EASTER_TRADITION if easter_tradition is None else easter_tradition
    if transform == "auto":
        transform = choose_transform(y, period)
    if transform == "log" and (not np.isfinite(y).all() or np.nanmin(y) <= 0):
        warnings.warn(tr("transform='log' невозможен: в ряде есть неположительные значения -- "
                         "используется аддитивная модель.",
                         "transform='log' is impossible: the series has non-positive values -- "
                         "the additive model is used."))
        transform = "none"
    is_log = (transform == "log")
    yw = np.log(y) if is_log else y                       # working scale | рабочая шкала
    to_level = (lambda v: np.exp(np.asarray(v, dtype=float))) if is_log else (lambda v: np.asarray(v, dtype=float))

    base_cols, t = _harmonic_trend_cols(n, period, n_harm)
    calendar_idx = []
    if calendar_adjust and period == 12:
        base_cols.append(easter_regressor(dates, L=easter_window,
                                          tradition=easter_tradition, center=True))
        calendar_idx.append(len(base_cols)-1)
        if trading_day:
            base_cols.append(trading_day_regressor(dates, working_days_calendar))
            calendar_idx.append(len(base_cols)-1)

    if outliers is not None:
        extra = [_outlier_col(t, idx0, kind) for idx0, kind in outliers]
        Xf = np.column_stack(base_cols+extra) if extra else np.column_stack(base_cols)
        beta,*_ = np.linalg.lstsq(Xf, yw, rcond=None)
        nb = len(base_cols)
        oeff = sum(Xf[:,nb+j]*beta[nb+j] for j in range(len(extra))) if extra else np.zeros(n)
        cal_eff = sum(Xf[:, j]*beta[j] for j in calendar_idx) if calendar_idx else np.zeros(n)
        found_outliers = [(i,k,np.nan) for i,k in outliers]; y_clean = yw - oeff
        outlier_source = "manual"
    else:
        search_ = auto_outlier_search(yw, base_cols, period=period, cv=outlier_cv,
                                      types=outlier_types, skip_edge=outlier_skip_edge,
                                      min_gap=outlier_min_gap, calendar_cols=tuple(calendar_idx),
                                      gls=gls, ar_order=ar_order)
        found_outliers = search_["outliers"]; y_clean = search_["y_clean"]; beta = search_["beta"]
        cal_eff = search_["calendar_effect"]
        outlier_source = "auto"

    # What goes into the decomposition is cleaned of BOTH outliers AND calendar effects:
    # estimating the calendar beta without subtracting it is not a calendar adjustment.
    # В декомпозицию идёт ряд, очищенный И от выбросов, И от календарных эффектов:
    # оценить beta календаря и не вычесть его -- значит не сделать календарную корректировку.
    y_ready = y_clean - cal_eff
    decomp = DECOMPOSERS[_resolve_method(method)](y_ready, period)
    metrics = evaluate_seasonality(yw, decomp["sa"], dates, period=period, seasonal=decomp.get("seasonal"),
                                    irregular=decomp.get("irregular"), outliers=found_outliers,
                                    method_name=method_name, show_chart=False)
    sa_level = to_level(decomp["sa"])
    decomp_level = {"trend": to_level(decomp["trend"]) if decomp.get("trend") is not None else None,
                    "seasonal": (np.exp(decomp["seasonal"]) if is_log else decomp.get("seasonal")),
                    "sa": sa_level, "irregular": decomp.get("irregular")}
    ma3_, saar_ = three_month_ma_saar(sa_level, mode=saar_mode)
    fc_w = forecast_deterministic(decomp["sa"], dates, forecast_horizon, period, n_harm)
    forecast_ = {"forecast": to_level(fc_w["forecast"]), "lo": to_level(fc_w["lo"]),
                 "hi": to_level(fc_w["hi"]), "se": fc_w["se"], "dates": fc_w["dates"],
                 "rho": fc_w["rho"], "scale": "log" if is_log else "level"}
    backtest_ = backtest_forecast(decomp["sa"], dates, backtest_holdout, period, n_harm,
                                  inverse=np.exp if is_log else None)

    result = {"y": y, "dates": dates, "transform": transform,
              "y_clean": to_level(y_clean), "y_ready": to_level(y_ready),
              "calendar_effect": cal_eff, "outliers": found_outliers,
              "outlier_source": outlier_source,
              "decomposition": decomp_level, "decomposition_work": decomp,
              "metrics": metrics, "ma3": ma3_, "saar_3mma": saar_,
              "forecast": forecast_, "backtest": backtest_, "method": method_name}
    if show_chart:
        _plot_dashboard(result, dark=dark)
    return result

def _plot_dashboard(result, dark=False):
    """Draw the summary dashboard from the result of seasonal_adjustment_pipeline():
    original vs SA series, seasonal component, 3MMA and 3MMA SAAR, forecast with
    interval, backtest.

    [RU] Сводный дашборд по результату seasonal_adjustment_pipeline(): исходный и SA-ряд,
    сезонная компонента, 3MMA и 3MMA SAAR, прогноз с интервалом, бэктест.
    """
    pal_ = econ_style("dark" if dark else "light")
    dates, y, decomp = result["dates"], result["y"], result["decomposition"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes[0,0].plot(dates, y, color=pal_["korall"], label=tr("Исходный", "Original"), linewidth=1.2)
    axes[0,0].plot(dates, decomp["sa"], color=pal_["lazur"], label="SA")
    outlier_style = dict(color=pal_["zoloto"], linestyle="--", linewidth=1.1) if result.get("outlier_source")=="manual" \
                    else dict(color=pal_["muted"], linestyle=":", linewidth=0.8)
    for idx0,kind,_ in result["outliers"]:
        axes[0,0].axvline(dates[idx0], **outlier_style)
        axes[0,0].annotate(kind, (dates[idx0], y.max()), fontsize=7, color=outlier_style["color"], ha="center")
    src_label = (tr("заданы аналитиком вручную", "set by hand") if result.get("outlier_source")=="manual"
                 else tr("найдены автоматически", "detected automatically"))
    chart_frame(axes[0,0], tr(f"{result['method']}: исходный vs SA ({len(result['outliers'])} выбросов -- {src_label})",
                             f"{result['method']}: original vs SA ({len(result['outliers'])} outliers -- {src_label})"), pal=pal)
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(dates, decomp["seasonal"], color=pal_["volna"]); chart_frame(axes[0,1], tr("Сезонная компонента", "Seasonal component"), pal=pal)
    axes[1,0].plot(dates, result["ma3"], color=pal_["lazur"]); chart_frame(axes[1,0], "3MMA", pal=pal)
    axes[1,1].plot(dates, result["saar_3mma"], color=pal_["zoloto"]); axes[1,1].axhline(0,color=pal_["muted"],linewidth=0.7)
    chart_frame(axes[1,1], "3MMA SAAR, %", pal=pal)
    n_show = min(36, len(y))
    axes[2,0].plot(dates[-n_show:], decomp["sa"][-n_show:], color=pal_["lazur"], label="SA")
    axes[2,0].plot(result["forecast"]["dates"], result["forecast"]["forecast"], "--", color=pal_["korall"], label=tr("Прогноз", "Forecast"))
    if result["forecast"].get("lo") is not None:
        axes[2,0].fill_between(result["forecast"]["dates"],
                                result["forecast"]["lo"], result["forecast"]["hi"],
                                color=pal_["korall"], alpha=0.15, label=tr("95% ДИ", "95% CI"))
    axes[2,0].legend(fontsize=8); chart_frame(axes[2,0], tr("Прогноз (с 95% ДИ)", "Forecast (with 95% CI)"), pal=pal)
    bt = result["backtest"]
    axes[2,1].plot(bt["y_true"], color=pal_["korall"], label=tr("Факт", "Actual"))
    axes[2,1].plot(bt["y_pred"], "--", color=pal_["lazur"], label="Backtest")
    axes[2,1].legend(fontsize=8); chart_frame(axes[2,1], tr(f"Бэктест: RMSE={bt['rmse']:.2f}, MAPE={bt['mape']:.1f}%", f"Backtest: RMSE={bt['rmse']:.2f}, MAPE={bt['mape']:.1f}%"), pal=pal)
    fig.suptitle(tr(f"Единый дашборд -- {result['method']}", f"Summary dashboard -- {result['method']}"), fontsize=14, y=1.01)
    annotate_source(axes[2,1], pal=pal_); plt.tight_layout(); plt.show()

def convert_to_level(values, input_format, dates=None):
    """Bring the input to a LEVEL -- the single currency of everything downstream.

    'level'          -- already a level, nothing to do.
    'mom' / 'qoq'    -- month-on-month or quarter-on-quarter growth in PERCENT; the
                        level is rebuilt with cumprod (see "levels or rates?").
    'ytd_cumulative' -- cumulated since the start of the year; de-cumulated with
                        decumulate_ytd().

    [RU] Приводит вход к УРОВНЮ -- единой валюте всего дальнейшего анализа.

    'level'          -- уже уровень, ничего не делаем.
    'mom' / 'qoq'    -- темп м/м или к/к в ПРОЦЕНТАХ; уровень восстанавливается через cumprod
                        (см. «уровни или темпы?»).
    'ytd_cumulative' -- накопленный итог с начала года; де-накапливается через decumulate_ytd().
    """
    values = np.asarray(values, dtype=float)
    if input_format == "level":
        return values
    elif input_format in ("mom", "qoq"):
        # Rosstat publishes the m/m CPI in INDEX form (100.6), not as growth (0.6);
        # the mix-up is easy and silent, hence the explicit check.
        # Росстат публикует ИПЦ м/м в ИНДЕКСНОЙ форме (100.6), а не как прирост (0.6);
        # перепутать легко, а ошибка тихая -- поэтому явная проверка.
        if np.nanmedian(np.abs(values)) > 50:
            raise ValueError(tr(
                "input_format='mom'/'qoq' ожидает ПРИРОСТ в процентах (например 0.6), а получены "
                "значения около 100 -- похоже, это индекс к предыдущему периоду (100.6). "
                "Передайте values - 100, либо input_format='level', если это уровень.",
                "input_format='mom'/'qoq' expects GROWTH in percent (e.g. 0.6), but the values are "
                "around 100 -- this looks like an index to the previous period (100.6). "
                "Pass values - 100, or input_format='level' if this is a level."))
        return 100 * np.cumprod(1 + values/100)
    elif input_format == "ytd_cumulative":
        s = pd.Series(values, index=pd.DatetimeIndex(dates))
        return decumulate_ytd(s).values
    else:
        raise ValueError(tr(f"Неизвестный input_format: {input_format} (level/mom/qoq/ytd_cumulative)",
                           f"Unknown input_format: {input_format} (level/mom/qoq/ytd_cumulative)"))

def _interpret_quality(metrics):
    """Human-readable interpretation of the quality metrics -- not only numbers but a
    verbal conclusion.

    [RU] Человекочитаемая интерпретация метрик качества -- не только числа, но и вывод словами.
    """
    lines = []
    if metrics["F_pvalue_after"] > 0.05:
        lines.append(tr(f"Остаточной сезонности не обнаружено (F-тест: p={metrics['F_pvalue_after']:.3f} > 0.05).",
                        f"No residual seasonality detected (F-test: p={metrics['F_pvalue_after']:.3f} > 0.05)."))
    else:
        lines.append(tr(f"ВНИМАНИЕ: остаточная сезонность возможна (F-тест: p={metrics['F_pvalue_after']:.3f} <= 0.05).",
                        f"WARNING: residual seasonality is possible (F-test: p={metrics['F_pvalue_after']:.3f} <= 0.05)."))
    reduction = metrics["seasonal_power_reduction_x"]
    lines.append(tr(
        f"Доля спектральной мощности на сезонных частотах снизилась в {reduction:.0f} раз "
        f"({metrics['seasonal_power_share_before']*100:.1f}% -> {metrics['seasonal_power_share_after']*100:.2f}%).",
        f"The share of spectral power at seasonal frequencies fell {reduction:.0f}-fold "
        f"({metrics['seasonal_power_share_before']*100:.1f}% -> {metrics['seasonal_power_share_after']*100:.2f}%)."))
    return " ".join(lines)

def analyze_series(data, input_format="level", period=12, method="x11",
                    transform=None, arima_order=None, calendar_adjust=True, easter_window=8,
                    easter_tradition=None, trading_day=True, working_days_calendar=None,
                    auto_outliers=True, outliers=None, outlier_cv=3.5,
                    outlier_types=("AO", "LS", "TC"), outlier_skip_edge=6, outlier_min_gap=3,
                    gls=None, ar_order=1,
                    n_harm=2, forecast_horizon=12, backtest_holdout=12, saar_mode="level_ma",
                    saar_n_sim=2000, plot_mode="full", dark=False):
    """THE main entry point -- takes raw data in any of the usual shapes and returns a
    full set of diagnostics and forecasts.

    data            -- pd.Series, or DataFrame with a 'Value' column and a DatetimeIndex
                       (see load_indicator() for loading from Excel or CSV).
    input_format    -- 'level' (already a level), 'mom'/'qoq' (month- or quarter-on-quarter
                       growth in %, converted to a level with cumprod), 'ytd_cumulative'
                       (cumulated since the start of the year, de-cumulated automatically).
    method          -- 'x11'/'ucm'/'stl' (our implementations) or 'x13_real' (the real
                       x13as binary -- requires X13_BIN). 'seats' is a legacy synonym of
                       'ucm', see IX.1.7.
    transform       -- 'auto'/'log'/'none': additive or multiplicative model (defaults to
                       DEFAULT_TRANSFORM). For method='x13_real' it is passed into the spec
                       as transform{ function=... }; 'auto' leaves the choice to X-13 and
                       its own AICC test. See section VI.4.
    arima_order     -- ((p,d,q),(P,D,Q)) for method='x13_real' when a specific ARIMA is
                       wanted instead of automdl (default None -> automdl).
    easter_tradition -- 'orthodox' (default, for Russian series) or 'catholic'. See VII.1.
    outliers        -- pd.DataFrame with columns 'date','type' (AO/LS/TC) to specify
                       outliers by hand instead of searching for them.
    working_days_calendar -- dict {(year, month): working_days} -- your own official
                       working calendar instead of the simplified Mon-Fri count.
    outlier_skip_edge -- width of the blind zone at the ends during outlier search; set
                       0-2 when monitoring a fresh shock (see VIII.1).
    gls / ar_order  -- whether to whiten the residual before searching for outliers
                       (None => DEFAULT_GLS) and with which AR order. See VIII.2.
    saar_mode       -- 3MMA SAAR convention: 'level_ma' | 'level_3m' | 'rate_avg' (CBR),
                       see the table in X.1.
    saar_n_sim      -- number of simulated forecast paths used to build the SAAR fan.
    plot_mode       -- 'full' (the complete dashboard with the quality criterion),
                       'single' (one chart: original + SA + forecast + interval),
                       'saar' (3MMA SAAR only, with forecast and interval), or None.

    Returns a dict with EVERYTHING computed along the way -- outliers, the model used,
    quality criteria (with a verbal interpretation), the forecast, 3MMA SAAR with its
    forecast and interval, the backtest, and for method='x13_real' the raw x13as report
    in x13_out / x13_err.

    [RU] ГЛАВНАЯ точка входа -- принимает сырые данные в любом из привычных видов и
    возвращает полный набор диагностик и прогнозов.

    data            -- pd.Series или DataFrame с колонкой 'Value' и DatetimeIndex
                       (см. load_indicator() для загрузки из Excel или CSV).
    input_format    -- 'level' (уже уровень), 'mom'/'qoq' (темп м/м или к/к в %, переводится
                       в уровень через cumprod), 'ytd_cumulative' (накопленный с начала года,
                       де-накапливается автоматически).
    method          -- 'x11'/'ucm'/'stl' (наши реализации) или 'x13_real' (настоящий бинарник
                       x13as -- нужен X13_BIN). 'seats' -- устаревший синоним 'ucm', см. IX.1.7.
    transform       -- 'auto'/'log'/'none': аддитивная или мультипликативная модель
                       (по умолчанию DEFAULT_TRANSFORM). Для method='x13_real' передаётся в
                       spec как transform{ function=... }; 'auto' оставляет выбор X-13 и его
                       собственному AICC-тесту. См. раздел VI.4.
    arima_order     -- ((p,d,q),(P,D,Q)) для method='x13_real', если нужна конкретная ARIMA
                       вместо automdl (по умолчанию None -> automdl).
    easter_tradition -- 'orthodox' (по умолчанию, для рядов РФ) или 'catholic'. См. VII.1.
    outliers        -- pd.DataFrame с колонками 'date','type' (AO/LS/TC), чтобы задать
                       выбросы вручную вместо поиска.
    working_days_calendar -- словарь {(год, месяц): рабочих_дней} -- свой производственный
                       календарь вместо упрощённого подсчёта Пн-Пт.
    outlier_skip_edge -- ширина «слепой зоны» у краёв при поиске выбросов; для мониторинга
                       свежего шока ставьте 0-2 (см. VIII.1).
    gls / ar_order  -- отбеливать ли остаток перед поиском выбросов (None => DEFAULT_GLS) и
                       каким порядком AR. См. VIII.2.
    saar_mode       -- конвенция 3MMA SAAR: 'level_ma' | 'level_3m' | 'rate_avg' (ЦБ РФ),
                       см. таблицу в X.1.
    saar_n_sim      -- число смоделированных траекторий прогноза для веера SAAR.
    plot_mode       -- 'full' (полный дашборд с критерием качества), 'single' (один график:
                       исходный + SA + прогноз + интервал), 'saar' (только 3MMA SAAR с
                       прогнозом и интервалом) или None.

    Возвращает словарь со ВСЕМ, что посчитано по дороге: выбросы, использованная модель,
    критерии качества (со словесной интерпретацией), прогноз, 3MMA SAAR с прогнозом и
    интервалом, бэктест, а для method='x13_real' -- сырой отчёт x13as в x13_out / x13_err.
    """
    if isinstance(data, pd.Series):
        dates_in, values_in = data.index, data.values
    else:
        dates_in, values_in = data.index, data["Value"].values
    dates_in = pd.DatetimeIndex(dates_in)
    n = len(values_in)

    # --- 1. input format -> level ---
    # --- 1. формат входных данных -> уровень ---
    y_level = convert_to_level(values_in, input_format, dates=dates_in)

    # --- 1b. additive or multiplicative model (see VI.4) ---
    # --- 1b. аддитивная или мультипликативная модель (см. VI.4) ---
    transform = DEFAULT_TRANSFORM if transform is None else transform
    easter_tradition = DEFAULT_EASTER_TRADITION if easter_tradition is None else easter_tradition
    if transform == "auto" and method != "x13_real":
        transform = choose_transform(y_level, period)   # for x13_real the engine decides | для x13_real решает сам движок
    if transform == "log" and (not np.isfinite(y_level).all() or np.nanmin(y_level) <= 0):
        warnings.warn(tr("transform='log' невозможен: в ряде есть неположительные значения -- "
                         "используется аддитивная модель.",
                         "transform='log' is impossible: the series has non-positive values -- "
                         "the additive model is used."))
        transform = "none"
    is_log = (transform == "log") and method != "x13_real"
    yw = np.log(y_level) if is_log else y_level
    to_level = (lambda v: np.exp(np.asarray(v, dtype=float))) if is_log else (lambda v: np.asarray(v, dtype=float))

    # --- 2. calendar regressors + base (trend and harmonics) ---
    # --- 2. календарные регрессоры + база (тренд+гармоники) ---
    base_cols, t = _harmonic_trend_cols(n, period, n_harm)
    calendar_names, calendar_idx = [], []
    if calendar_adjust and period == 12:
        base_cols.append(easter_regressor(dates_in, L=easter_window,
                                          tradition=easter_tradition, center=True))
        calendar_names.append(f"easter_{easter_tradition}"); calendar_idx.append(len(base_cols)-1)
        if trading_day:
            base_cols.append(trading_day_regressor(dates_in, working_days_calendar))
            calendar_names.append("trading_day"); calendar_idx.append(len(base_cols)-1)

    method_name = _resolve_method(method).upper()
    arima_used = None

    if method == "x13_real":
        # --- the real-binary branch: a user-specified OR an automatic ARIMA ---
        # --- ветка настоящего бинарника: своя ИЛИ автоматическая ARIMA ---
        if arima_order is not None:
            (p, d, q), (P, D, Q) = arima_order
            arima_block = f"arima{{ model=({p} {d} {q})({P} {D} {Q}) }}"
        else:
            arima_block = "automdl{ }"
        outlier_block = f"outlier{{ types=({' '.join(outlier_types)}) }}" if (auto_outliers and outliers is None) else ""
        start_str = f"{dates_in[0].year}.{dates_in[0].month}"
        tr_fun = {"auto": "auto", "log": "log", "none": "none"}[transform]
        # A custom working calendar is passed in as a user regressor of type td.
        # X-13 does NOT extend it itself, so the regressor is built over the forecast horizon.
        # Пользовательский производственный календарь подаётся как user-регрессор типа td.
        # X-13 сам его НЕ продлевает, поэтому регрессор строится на горизонт прогноза вперёд.
        if working_days_calendar is not None and period == 12:
            fut = pd.date_range(dates_in[-1], periods=forecast_horizon+1, freq="ME")[1:]
            td_ext = trading_day_regressor(dates_in.append(fut), working_days_calendar)
            reg_block = (f"regression{{\n    user=(mytd)\n    usertype=td\n    start={start_str}\n"
                         f"    data=({_wrap_data(td_ext)})\n    aictest=(easter)\n}}")
        else:
            reg_block = "regression{ aictest=(td easter) }"
        spec = f"""series{{
    title="analyze_series"
    start={start_str}
    period={period}
    data=({_wrap_data(y_level)})
}}
transform{{ function={tr_fun} }}
{reg_block}
{outlier_block}
{arima_block}
x11{{ save=(d11) }}
forecast{{ maxlead={forecast_horizon} }}
"""
        out_x13, err_x13 = run_x13(spec, ".", "analyze_series_tmp")
        found_outliers = parse_outliers(out_x13)
        arima_used = parse_arima_model(out_x13)
        sa_series = read_d11(".", "analyze_series_tmp")
        y_clean = y_level  # x13 cleans internally; y_clean is kept here only for interface consistency | x13 сам чистит внутри; y_clean здесь -- для единообразия интерфейса
        decomp = {"trend": None, "seasonal": y_level - sa_series if sa_series is not None else None,
                  "sa": sa_series, "irregular": None}
        if sa_series is None:
            head = (err_x13 or out_x13 or "").strip().splitlines()[:8]
            raise RuntimeError(tr(
                "x13as не вернул сезонно скорректированный ряд (таблица d11). Частые причины: "
                "модель не сошлась (особенно при вручную заданной ARIMA вместе с авто-поиском "
                "выбросов -- см. XI.5), ошибка в spec-файле, нет прав на запись в рабочую папку.\n"
                "Первые строки отчёта:\n  ",
                "x13as returned no seasonally adjusted series (table d11). Common causes: the "
                "model did not converge (especially with a hand-set ARIMA combined with automatic "
                "outlier search -- see XI.5), an error in the spec file, no write permission in the "
                "working folder.\nFirst lines of the report:\n  ") + "\n  ".join(head))
        fc_real = parse_forecasts(out_x13)
        if fc_real is not None:
            # Safety net in case a particular x13as build names its tables differently and the
            # parsed forecast still turns out to be in logarithms:
            # compare its level with the last observations of the series.
            # Подстраховка на случай, если в конкретной сборке x13as заголовки таблиц
            # отличаются и распознанный прогноз всё-таки оказался в логарифмах:
            # сравниваем его уровень с последними наблюдениями ряда.
            ref = float(np.nanmedian(y_level[-min(12, len(y_level)):]))
            med = float(np.nanmedian(fc_real["forecast"]))
            if ref > 0 and np.isfinite(med):
                off_raw = abs(np.log(max(med, 1e-12)/ref))
                off_exp = abs(np.log(max(np.exp(min(med, 700)), 1e-12)/ref))
                if off_raw > np.log(3) and off_exp < np.log(1.5):
                    for key in ("forecast", "lo", "hi"):
                        fc_real[key] = np.exp(fc_real[key])
                    fc_real["se"] = (fc_real["hi"] - fc_real["lo"])/(2*1.96)
                    fc_real["se_approx"] = True
                    warnings.warn(tr(
                        "Прогноз из x13as был напечатан в логарифмах (таблица преобразованных "
                        "данных) -- возвращён на исходную шкалу экспонентой. Проверьте заголовок "
                        f'таблицы: "{fc_real.get("header", "")}".',
                        "The x13as forecast was printed in logarithms (the transformed-data table) "
                        "-- it was brought back to the original scale with exp. Check the table "
                        f'header: "{fc_real.get("header", "")}".'))
        if fc_real is None:
            warnings.warn(tr(
                "x13as не напечатал таблицу прогноза: analyze_series вернёт forecast=None, "
                "графики прогноза и веер SAAR построены не будут. Сам отчёт доступен в "
                'result["x13_out"] / result["x13_err"].',
                "x13as printed no forecast table: analyze_series will return forecast=None, and "
                "no forecast charts or SAAR fan will be drawn. The report itself is in "
                'result["x13_out"] / result["x13_err"].'))
        # fc_real may be None (the binary printed no forecast table) --
        # further down this used to raise TypeError; the branch is now guarded explicitly.
        # fc_real может быть None (бинарник не напечатал таблицу прогнозов) --
        # дальше по коду это раньше приводило к TypeError; теперь ветка явно защищена.
        forecast = ({"forecast": fc_real["forecast"], "se": fc_real["se"],
                     "lo": fc_real["lo"], "hi": fc_real["hi"],
                     "se_approx": fc_real["se_approx"], "dates": fc_real["dates"]}
                    if fc_real else None)
    else:
        # --- the branch of our -lite implementations ---
        # --- ветка наших -lite реализаций ---
        if outliers is not None:
            idx_map = {d: i for i, d in enumerate(dates_in)}
            outliers_list = [(idx_map[pd.Timestamp(d)], k) for d, k in
                              zip(outliers["date"], outliers["type"]) if pd.Timestamp(d) in idx_map]
            extra_cols = [_outlier_col(t, idx0, kind) for idx0, kind in outliers_list]
            Xf = np.column_stack(base_cols + extra_cols) if extra_cols else np.column_stack(base_cols)
            beta, *_ = np.linalg.lstsq(Xf, yw, rcond=None)
            nb = len(base_cols)
            oeff = sum(Xf[:, nb+j]*beta[nb+j] for j in range(len(extra_cols))) if extra_cols else np.zeros(n)
            cal_eff = sum(Xf[:, j]*beta[j] for j in calendar_idx) if calendar_idx else np.zeros(n)
            found_outliers = [(i, k, np.nan) for i, k in outliers_list]
            y_clean = yw - oeff
        elif auto_outliers:
            search_ = auto_outlier_search(yw, base_cols, period=period, cv=outlier_cv,
                                          types=outlier_types, skip_edge=outlier_skip_edge,
                                          min_gap=outlier_min_gap, calendar_cols=tuple(calendar_idx),
                                          gls=gls, ar_order=ar_order)
            found_outliers = search_["outliers"]; y_clean = search_["y_clean"]
            cal_eff = search_["calendar_effect"]
        else:
            Xf = np.column_stack(base_cols)
            beta, *_ = np.linalg.lstsq(Xf, yw, rcond=None)
            found_outliers = []; y_clean = yw
            cal_eff = sum(Xf[:, j]*beta[j] for j in calendar_idx) if calendar_idx else np.zeros(n)

        # The calendar effect is SUBTRACTED, not merely estimated -- otherwise the
        # calendar_adjust argument would not affect the result.
        # Календарный эффект ВЫЧИТАЕТСЯ, а не только оценивается -- иначе аргумент
        # calendar_adjust не влиял бы на результат.
        # Эффект выбросов в рабочей шкале: он понадобится, чтобы ВЕРНУТЬ его в итоговый ряд.
        # The outlier effect in the work scale: needed to RESTORE it in the final series.
        out_eff = yw - y_clean
        y_ready = y_clean - cal_eff
        decomp = DECOMPOSERS[_resolve_method(method)](y_ready, period)

        # ВОЗВРАТ ЭФФЕКТА ВЫБРОСОВ В ИТОГОВЫЙ РЯД.
        # Сезонная корректировка убирает СЕЗОННОСТЬ И КАЛЕНДАРЬ, а не выбросы. Выбросы нужны
        # только для того, чтобы устойчиво оценить сезонные факторы: пока они в ряде, они
        # протекают в оценку сезонности. После оценки их эффект возвращается -- именно поэтому
        # в опубликованных SA-рядах ковидный провал виден, он реальный, а не артефакт.
        # Настоящий X-13 так и делает (его d11 содержит выбросы), а наши -lite реализации
        # раньше отдавали ряд, очищенный и от сезонности, и от выбросов. На M2 это давало
        # сдвиг уровня +2,85% и расхождение с оценкой Банка России 2,89% вместо 0,43%.
        #
        # RESTORING THE OUTLIER EFFECT IN THE FINAL SERIES. Seasonal adjustment removes
        # SEASONALITY AND THE CALENDAR, not outliers: outliers are removed only so that the
        # seasonal factors can be estimated robustly, and their effect is then put back. The
        # real X-13 does exactly this; our -lite implementations used to return a series
        # cleaned of both, which shifted the level of M2 by +2.85%.
        if np.any(out_eff):
            decomp["sa"] = np.asarray(decomp["sa"], dtype=float) + out_eff
            if decomp.get("irregular") is not None:
                # Тождество тренд + сезонность + нерегулярная = ряд должно сохраниться.
                decomp["irregular"] = np.asarray(decomp["irregular"], dtype=float) + out_eff
        forecast = forecast_deterministic(decomp["sa"], dates_in, forecast_horizon, period, n_harm,
                                          n_sim=saar_n_sim)

    # --- 3. quality (identical for ANY method; computed in the working scale) ---
    # --- 3. качество (одинаково для ЛЮБОГО метода; считается в рабочей шкале) ---
    metrics = evaluate_seasonality(yw if method != "x13_real" else y_level, decomp["sa"], dates_in,
                                    period=period, seasonal=decomp.get("seasonal"),
                                    irregular=decomp.get("irregular"),
                                    outliers=found_outliers, method_name=method_name, show_chart=False)
    metrics["interpretation"] = _interpret_quality(metrics)

    # --- 3b. back to the original units ---
    # --- 3b. возврат в исходные единицы ---
    sa_level = to_level(decomp["sa"]) if method != "x13_real" else decomp["sa"]
    sa_work_for_backtest = decomp["sa"]          # SA до перевода в уровни -- нужен бэктесту
    decomp = {"trend": (to_level(decomp["trend"]) if decomp.get("trend") is not None else None),
              "seasonal": (np.exp(decomp["seasonal"]) if (is_log and decomp.get("seasonal") is not None)
                           else decomp.get("seasonal")),
              "sa": sa_level, "irregular": decomp.get("irregular")}
    if forecast is not None and method != "x13_real":
        forecast = {"forecast": to_level(forecast["forecast"]),
                    "lo": to_level(forecast["lo"]), "hi": to_level(forecast["hi"]),
                    "se": forecast["se"], "dates": forecast["dates"], "rho": forecast["rho"],
                    "paths": (np.exp(forecast["paths"]) if (is_log and forecast["paths"] is not None)
                              else forecast["paths"]),
                    "scale": "log" if is_log else "level"}
    elif forecast is not None:
        forecast = {**forecast, "paths": None, "scale": "level"}

    # --- 4. 3MMA SAAR + forecast + interval ---
    # Two changes against the original version:
    #  (1) for method="x13_real" the "Forecasts and Standard Errors" table forecasts the
    #      ORIGINAL (seasonal) series, not d11. Splicing it onto SA history is wrong, yet that
    #      is how the SAAR fan used to be built. Now x13_real gives saar_forecast = None (XI.6).
    #  (2) a SAAR fan obtained by pushing the bounds of a LEVEL interval through the same
    #      formula is not a confidence interval for SAAR: from horizon 3 on, numerator and
    #      denominator shift by the same amount and the band collapses. A crude illustration,
    #      as the chart subtitle states.
    # --- 4. 3MMA SAAR + прогноз + ДИ ---
    # Две правки против исходной версии:
    #  (1) для method="x13_real" таблица "Forecasts and Standard Errors" -- это прогноз ИСХОДНОГО
    #      (сезонного) ряда, а не d11. Клеить его к SA-истории нельзя: раньше веер SAAR строился
    #      именно так. Теперь для x13_real saar_forecast = None (см. оговорку в XI.6).
    #  (2) веер SAAR, полученный протягиванием границ ДИ УРОВНЯ через ту же формулу, -- не
    #      доверительный интервал для SAAR: начиная с горизонта 3 и числитель, и знаменатель
    #      сдвигаются на одну и ту же величину, и интервал схлопывается. Это грубая иллюстрация,
    #      что и помечено в подписи к графику.
    ma3, saar_hist = three_month_ma_saar(decomp["sa"], mode=saar_mode)
    saar_forecast = None
    if forecast is not None and method != "x13_real" and forecast.get("paths") is not None:
        # SAAR is a NON-LINEAR function of the level, so pushing the bounds of a level interval
        # through its formula is invalid: wherever numerator and denominator shift equally,
        # the band collapses. Instead compute SAAR on every simulated path and take
        # percentiles -- that is the correct interval for a growth rate.
        # SAAR -- НЕЛИНЕЙНАЯ функция уровня, поэтому протягивать границы интервала уровня
        # через её формулу нельзя: там, где числитель и знаменатель сдвинуты одинаково,
        # полоса схлопывается. Считаем SAAR по каждой смоделированной траектории и берём
        # процентили -- это и есть корректный интервал для темпа.
        paths = forecast["paths"]
        full = np.concatenate([np.tile(decomp["sa"], (paths.shape[0], 1)), paths], axis=1)
        dfp = pd.DataFrame(full.T)
        ma3p = dfp.rolling(3).mean()
        if saar_mode == "level_ma":
            saar_p = ((ma3p/ma3p.shift(3))**4 - 1)*100
        elif saar_mode == "level_3m":
            saar_p = ((dfp/dfp.shift(3))**4 - 1)*100
        else:
            gp = dfp/dfp.shift(1) - 1
            saar_p = ((1 + gp.rolling(3).mean())**12 - 1)*100
        arr = saar_p.to_numpy()
        saar_point = np.concatenate([saar_hist.to_numpy()[:len(decomp["sa"])],
                                     np.nanmedian(arr[len(decomp["sa"]):], axis=1)])
        saar_lo = np.full_like(saar_point, np.nan); saar_hi = np.full_like(saar_point, np.nan)
        saar_lo[len(decomp["sa"]):] = np.nanpercentile(arr[len(decomp["sa"]):], 2.5, axis=1)
        saar_hi[len(decomp["sa"]):] = np.nanpercentile(arr[len(decomp["sa"]):], 97.5, axis=1)
        saar_forecast = {"point": saar_point, "hi": saar_hi, "lo": saar_lo}

    # --- 5. backtest (only for -lite methods; x13_real would need repeated binary runs) ---
    # The backtest runs on the same object as the forecast -- the SA series (this used to be
    # the raw level, so the RMSE referred to a different series than the plotted forecast).
    # --- 5. бэктест (только для -lite методов -- для x13_real потребовал бы повторных вызовов бинарника) ---
    # Бэктест считается на том же объекте, что и прогноз, -- на SA-ряде (раньше здесь был
    # сырой уровень, и RMSE относился к другому ряду, чем нарисованный прогноз).
    # ВАЖНО: бэктест считается в РАБОЧЕЙ шкале (в логарифме для мультипликативной модели),
    # как и сам прогноз, а результат переводится обратно. Раньше сюда попадал SA-ряд уже в
    # уровнях и без обратного преобразования: линейный тренд подгонялся к экспоненте, прогноз
    # систематически отставал, и MAPE выходил завышенным (3.4% вместо 1.5% на том же ряде).
    # IMPORTANT: the backtest runs in the WORK scale (in logs for a multiplicative model), like
    # the forecast itself, and the result is transformed back. It used to receive the SA series
    # already in levels with no inverse: a linear trend was fitted to exponential data, the
    # forecast lagged systematically and MAPE came out overstated (3.4% instead of 1.5%).
    backtest = (backtest_forecast(sa_work_for_backtest, dates_in, backtest_holdout, period,
                                  n_harm, inverse=np.exp if is_log else None)
                if method != "x13_real" else None)

    result = {
        "input_format": input_format, "transform": transform, "y_level": y_level,
        "y_clean": to_level(y_clean) if method != "x13_real" else y_clean,
        "y_ready": to_level(y_ready) if method != "x13_real" else y_level,
        "calendar_names": calendar_names, "dates": dates_in,
        "x13_out": out_x13 if method == "x13_real" else None,
        "x13_err": err_x13 if method == "x13_real" else None,
        "outliers": found_outliers, "arima_used": arima_used, "decomposition": decomp,
        "quality": metrics, "ma3": ma3, "saar_hist": saar_hist, "forecast": forecast,
        "saar_forecast": saar_forecast,
        "backtest": backtest, "method": method_name,
    }

    if plot_mode == "full":
        _plot_full(result, period=period, dark=dark)
    elif plot_mode == "single":
        _plot_single(result, dark=dark)
    elif plot_mode == "saar":
        _plot_saar(result, dark=dark)

    return result

def _plot_full(result, period=12, dark=False):
    """The full dashboard -- includes an explicitly visible quality criterion (the share
    of seasonal power before and after), not only pretty lines.

    [RU] Полный дашборд -- включает явно видимый критерий качества (доля сезонной мощности до
    и после), а не только красивые линии.
    """
    pal_ = econ_style("dark" if dark else "light")
    dates, y_level, decomp = result["dates"], result["y_level"], result["decomposition"]
    fig, axes = plt.subplots(3, 2, figsize=(14, 11))

    axes[0,0].plot(dates, y_level, color=pal_["korall"], alpha=0.7, label=tr("Исходный (уровень)", "Original (level)"))
    axes[0,0].plot(dates, decomp["sa"], color=pal_["lazur"], label=tr("Сезонно скорректированный", "Seasonally adjusted"))
    for idx0, kind, _ in result["outliers"]:
        if isinstance(idx0, int):
            axes[0,0].axvline(dates[idx0], color=pal_["zoloto"], linestyle=":", linewidth=0.8)
    chart_frame(axes[0,0], tr(f"{result['method']}: исходный vs SA", f"{result['method']}: original vs SA"),
                tr(f"выбросов: {len(result['outliers'])}", f"outliers: {len(result['outliers'])}"), pal=pal_)
    axes[0,0].legend(fontsize=8)

    if decomp.get("seasonal") is not None:
        axes[0,1].plot(dates, decomp["seasonal"], color=pal_["volna"], label=tr("Сезонная компонента", "Seasonal component"))
        axes[0,1].axhline(0, color=pal_["muted"], linewidth=0.6)
        axes[0,1].legend(fontsize=8)
    chart_frame(axes[0,1], tr("Сезонная компонента", "Seasonal component"), pal=pal_)

    # The EXPLICIT quality criterion -- share of seasonal power before/after
    # ЯВНЫЙ критерий качества -- доля сезонной мощности до/после
    q = result["quality"]
    bars = axes[1,0].bar([tr("До", "Before"), tr("После", "After")],
                          [q["seasonal_power_share_before"]*100, q["seasonal_power_share_after"]*100],
                          color=[pal_["korall"], pal_["lazur"]])
    for b in bars:
        axes[1,0].text(b.get_x()+b.get_width()/2, b.get_height(), f"{b.get_height():.2f}%", ha="center", va="bottom", fontsize=9)
    chart_frame(axes[1,0], tr("Критерий: доля мощности на сезонных частотах", "Criterion: share of power at seasonal frequencies"), q["interpretation"][:60]+"...", pal=pal_)

    axes[1,1].plot(dates, result["saar_hist"], color=pal_["zoloto"], label=tr("3MMA SAAR (факт)", "3MMA SAAR (actual)"))
    axes[1,1].axhline(0, color=pal_["muted"], linewidth=0.6)
    chart_frame(axes[1,1], "3MMA SAAR", pal=pal_)
    axes[1,1].legend(fontsize=8)

    n_show = min(36, len(y_level))
    axes[2,0].plot(dates[-n_show:], decomp["sa"][-n_show:], color=pal_["lazur"], label=tr("SA (факт)", "SA (actual)"))
    if result["forecast"] is not None:   # there may be no forecast at all (x13_real without a forecast table) | прогноза может не быть (x13_real без таблицы forecast)
        axes[2,0].plot(result["forecast"]["dates"], result["forecast"]["forecast"], "--", color=pal_["korall"], label=tr("Прогноз", "Forecast"))
    if result["forecast"] is not None and result["forecast"].get("lo") is not None:
        axes[2,0].fill_between(result["forecast"]["dates"],
                                result["forecast"]["lo"], result["forecast"]["hi"],
                                color=pal_["korall"], alpha=0.15, label=tr("95% ДИ", "95% CI"))
    chart_frame(axes[2,0], tr("Прогноз SA-ряда", "Forecast of the SA series"), pal=pal_)
    axes[2,0].legend(fontsize=8)

    if result["backtest"] is not None:
        bt = result["backtest"]
        axes[2,1].plot(range(len(bt["y_true"])), bt["y_true"], color=pal_["korall"], label=tr("Факт (holdout)", "Actual (holdout)"))
        axes[2,1].plot(range(len(bt["y_pred"])), bt["y_pred"], "--", color=pal_["lazur"], label=tr("Прогноз (backtest)", "Forecast (backtest)"))
        chart_frame(axes[2,1], tr(f"Бэктест: RMSE={bt['rmse']:.2f}, MAPE={bt['mape']:.1f}%", f"Backtest: RMSE={bt['rmse']:.2f}, MAPE={bt['mape']:.1f}%"), pal=pal_)
        axes[2,1].legend(fontsize=8)
    else:
        axes[2,1].axis("off")
        axes[2,1].text(0.5, 0.5, tr("Бэктест недоступен для method='x13_real'\n(потребовал бы повторных вызовов бинарника)",
                          "Backtest unavailable for method='x13_real'\n(it would require repeated runs of the binary)"),
                        ha="center", va="center", fontsize=9, color=pal_["muted"], transform=axes[2,1].transAxes)

    fig.suptitle(tr(f"Полный дашборд -- {result['method']}", f"Full dashboard -- {result['method']}")
                 + (tr(", лог-модель", ", log model") if result.get('transform') == 'log' else ""),
                 fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()
    print(q["interpretation"])

def _plot_single(result, dark=False):
    """One chart: original + adjusted + forecast with interval -- for a quick look.

    [RU] Один график: исходный + скорректированный + прогноз с интервалом -- для быстрого обзора.
    """
    pal_ = econ_style("dark" if dark else "light")
    dates, y_level, decomp = result["dates"], result["y_level"], result["decomposition"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(dates, y_level, color=pal_["muted"], alpha=0.4, label=tr("Исходный", "Original"))
    ax.plot(dates, decomp["sa"], color=pal_["lazur"], label=tr("Сезонно скорректированный", "Seasonally adjusted"))
    fc = result.get("forecast")          # for x13_real there may be no forecast at all | для x13_real прогноза может не быть вовсе
    if fc is not None:
        ax.plot(fc["dates"], fc["forecast"], "--", color=pal_["korall"], label=tr("Прогноз", "Forecast"))
        if fc.get("lo") is not None:
            ax.fill_between(fc["dates"], fc["lo"], fc["hi"],
                             color=pal_["korall"], alpha=0.15, label=tr("95% ДИ", "95% CI"))
    sub = None if fc is not None else tr("прогноз недоступен: x13as не напечатал таблицу прогноза",
                                          "no forecast: x13as printed no forecast table")
    chart_frame(ax, tr(f"{result['method']}: обзор одним графиком", f"{result['method']}: overview in one chart"), sub, pal=pal_)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()

def _plot_saar(result, dark=False):
    """One chart: 3MMA SAAR (actual plus forecast with interval) -- for monitoring the
    pace of growth.

    [RU] Один график: 3MMA SAAR (факт плюс прогноз с интервалом) -- для мониторинга темпа роста.
    """
    pal_ = econ_style("dark" if dark else "light")
    dates = result["dates"]
    fc = result.get("forecast")
    saar_f = result.get("saar_forecast")
    all_dates = dates.append(fc["dates"]) if fc is not None else dates
    n_hist = len(dates)
    if saar_f is None or fc is None:   # for x13_real no SAAR fan is built -- plot the actuals only | для x13_real веер SAAR не строится -- рисуем только факт
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.plot(dates, result["saar_hist"], color=pal_["lazur"], label=tr("3MMA SAAR (факт)", "3MMA SAAR (actual)"))
        ax.axhline(0, color=pal_["muted"], linewidth=0.6)
        chart_frame(ax, f"{result['method']}: 3MMA SAAR",
                    tr("прогноз SAAR недоступен: x13as прогнозирует исходный ряд, а не SA",
                       "no SAAR forecast: x13as forecasts the original series, not the SA one"), pal=pal_)
        ax.legend(fontsize=8); plt.tight_layout(); plt.show()
        return

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(all_dates[:n_hist], saar_f["point"][:n_hist], color=pal_["lazur"], label=tr("3MMA SAAR (факт)", "3MMA SAAR (actual)"))
    ax.plot(all_dates[n_hist:], saar_f["point"][n_hist:], "--", color=pal_["korall"], label=tr("3MMA SAAR (прогноз)", "3MMA SAAR (forecast)"))
    ax.fill_between(all_dates[n_hist:], saar_f["lo"][n_hist:], saar_f["hi"][n_hist:],
                     color=pal_["korall"], alpha=0.15, label=tr("95% ДИ (по траекториям)", "95% CI (from paths)"))
    ax.axhline(0, color=pal_["muted"], linewidth=0.6)
    chart_frame(ax, tr(f"{result['method']}: 3MMA SAAR с прогнозом", f"{result['method']}: 3MMA SAAR with forecast"),
                tr("веер -- 2.5/97.5 процентили SAAR по смоделированным траекториям прогноза",
                   "fan -- 2.5/97.5 percentiles of SAAR across simulated forecast paths"), pal=pal_)
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()


# ======================================================================
#  REGISTRY OF DECOMPOSITION METHODS
#  A new method is one line here; it immediately becomes available in both
#  seasonal_adjustment_pipeline() and analyze_series().
#  MSTL / the TBATS kernel / Prophet-lite are deliberately excluded: their input
#  signature differs (a list of periods, a non-integer period, a separate regression),
#  so honest integration needs an adapter, not a line in a dict.
#  РЕЕСТР МЕТОДОВ ДЕКОМПОЗИЦИИ
#  Новый метод -- одна строка здесь; он сразу станет доступен и в
#  seasonal_adjustment_pipeline(), и в analyze_series().
#  MSTL / TBATS-ядро / Prophet-lite сюда не включены осознанно: у них другая
#  сигнатура входа (список периодов, нецелый период, отдельная регрессия),
#  честная интеграция требует адаптера, а не строчки в словаре.
# ======================================================================
DECOMPOSERS = {
    "x11": lambda yc, p, **kw: x11_lite(yc, p, **kw),
    "ucm": lambda yc, p, **kw: ucm_lite(yc, p, **kw),
    "stl": lambda yc, p, **kw: stl_lite(yc, p, **kw),
}
# The old name "seats" still works as a synonym of "ucm" (_resolve_method maps it inside
# the pipelines), but it is absent from the dict itself -- otherwise a loop over
# DECOMPOSERS would run the same method twice.
# Старое имя "seats" продолжает работать как синоним "ucm" (его подменяет _resolve_method
# внутри пайплайнов), но в самом словаре его нет -- иначе цикл по DECOMPOSERS прогонял бы
# один и тот же метод дважды.
_METHOD_ALIASES = {"seats": "ucm", "bsm": "ucm"}


def _resolve_method(method):
    """Map a method name to its canonical form: "seats" -> "ucm" (the rename in IX.1.7).

    [RU] Приводит имя метода к каноническому виду: "seats" -> "ucm" (переименование в IX.1.7).
    """
    return _METHOD_ALIASES.get(method, method)





# ======================================================================
#  PART 2 (v1.1): PASSPORTS, RUNS, DATABASE, APPLYING A SPECIFICATION
#  Everything above this line is version 1.0 and is NOT modified: the same
#  functions, the same signatures, the same results. This part only adds.
#  ЧАСТЬ 2 (v1.1): ПАСПОРТА, ПЕРЕСМОТРЫ, БАЗА, ПРИМЕНЕНИЕ СПЕЦИФИКАЦИИ
#  Всё, что выше этой строки, -- версия 1.0 и НЕ изменено: те же функции,
#  те же сигнатуры, те же результаты. Эта часть только добавляет.
# ======================================================================
import copy
import glob
import hashlib
import json


__version__ = "1.1"

# ----------------------------------------------------------------------
#  Settings of part 2. Override them from the notebook, as in part 1.
#  Настройки части 2. Переопределяются из блокнота, как и в части 1.
# ----------------------------------------------------------------------
PASSPORT_SCHEMA_VERSION = "1.0"      # version of the passport FORMAT | версия ФОРМАТА паспорта
X13_BUILD = ""                       # x13as build, written into run metadata | сборка x13as, пишется в метаданные


# ======================================================================
#  INDICATOR TYPES AND ENGINES
#  ТИПЫ ПОКАЗАТЕЛЕЙ И ДВИЖКИ
# ======================================================================
INDICATOR_TYPES = {
    # type -> sensible starting values for the passport; the analyst confirms or changes them
    # тип -> разумные стартовые значения паспорта; аналитик подтверждает или меняет
    "flow_value":  {"transform": "log",  "trading_day": True,  "easter": True,  "saar_mode": "level_ma",
                    "ru": "поток в текущих ценах (оборот, выручка, экспорт)",
                    "en": "flow in current prices (turnover, revenue, exports)"},
    "flow_volume": {"transform": "auto", "trading_day": True,  "easter": True,  "saar_mode": "level_ma",
                    "ru": "поток в натуральном выражении (выпуск, перевозки)",
                    "en": "flow in physical terms (output, freight)"},
    "price_index": {"transform": "log",  "trading_day": False, "easter": True,  "saar_mode": "rate_avg",
                    "ru": "ценовой индекс (ИПЦ, ИЦП)",
                    "en": "price index (CPI, PPI)"},
    "stock":       {"transform": "log",  "trading_day": False, "easter": False, "saar_mode": "level_ma",
                    "ru": "запас на дату (денежная масса, остатки, резервы)",
                    "en": "stock at a date (money supply, balances, reserves)"},
    "balance":     {"transform": "none", "trading_day": False, "easter": False, "saar_mode": None,
                    "ru": "сальдо или чистый поток (может быть отрицательным)",
                    "en": "balance or net flow (may be negative)"},
    "rate":        {"transform": "none", "trading_day": False, "easter": False, "saar_mode": None,
                    "ru": "ставка, доля, показатель в процентах",
                    "en": "rate, share, indicator in percent"},
}

ENGINE_REGISTRY = {
    # pipeline=True  -- the engine is wired into analyze_series() and can be applied from a passport
    # pipeline=False -- implemented in the module, but not yet wired into the pipeline
    # pipeline=True  -- движок подключён к analyze_series() и применяется из паспорта
    # pipeline=False -- в модуле реализован, но к пайплайну пока не подключён
    "x11":      {"period": "int",      "optional": ("henderson_len", "iterations"),
                 "preprocess": "module", "outliers_default": "auto", "pipeline": True},
    "ucm":      {"period": "int",      "optional": ("var_ratios", "mle"),
                 "preprocess": "module", "outliers_default": "auto", "pipeline": True},
    "stl":      {"period": "int",      "optional": ("seasonal_frac", "n_inner", "n_outer"),
                 "preprocess": "module", "outliers_default": "manual", "pipeline": True},
    "x13_real": {"period": "int_12_4", "optional": ("decomposition", "arima"),
                 "preprocess": "engine", "outliers_default": "auto", "pipeline": True},
    "mstl":     {"period": "list_int", "optional": ("iterations",),
                 "preprocess": "module", "outliers_default": "manual", "pipeline": False},
    "tbats":    {"period": "float",    "optional": ("K",),
                 "preprocess": "module", "outliers_default": "manual", "pipeline": False},
    "prophet":  {"period": "float",    "optional": ("K", "n_changepoints", "alpha"),
                 "preprocess": "module", "outliers_default": "none", "pipeline": False},
    "none":     {"period": None,       "optional": (),
                 "preprocess": "module", "outliers_default": "none", "pipeline": True},
}

# A starter list of shocks in the Russian economy. Candidate outliers that coincide with these
# dates are highlighted for the analyst; the list is meant to be extended.
# Стартовый список шоков российской экономики. Кандидаты в выбросы, совпавшие с этими датами,
# подсвечиваются аналитику; список предполагается пополнять.
EVENTS_RU = [
    {"date": "1998-08-31", "ru": "Дефолт и девальвация", "en": "Default and devaluation", "scope": "all"},
    {"date": "2008-09-30", "ru": "Мировой финансовый кризис", "en": "Global financial crisis", "scope": "all"},
    {"date": "2014-12-31", "ru": "Резкая девальвация рубля, ключевая ставка 17%",
     "en": "Sharp rouble devaluation, key rate at 17%", "scope": "all"},
    {"date": "2015-01-31", "ru": "Смена методологии ряда показателей Росстата",
     "en": "Methodology change in a number of Rosstat indicators", "scope": "real"},
    {"date": "2019-01-31", "ru": "Повышение НДС с 18% до 20%", "en": "VAT raised from 18% to 20%", "scope": "prices"},
    {"date": "2020-04-30", "ru": "Локдаун, первая волна пандемии", "en": "Lockdown, first wave of the pandemic", "scope": "all"},
    {"date": "2020-06-30", "ru": "Снятие ограничений, отскок", "en": "Restrictions lifted, rebound", "scope": "all"},
    {"date": "2022-03-31", "ru": "Санкционный шок, ажиотажный спрос, ставка 20%",
     "en": "Sanctions shock, panic demand, key rate at 20%", "scope": "all"},
    {"date": "2022-04-30", "ru": "Коррекция после мартовского шока", "en": "Correction after the March shock", "scope": "all"},
    {"date": "2022-09-30", "ru": "Частичная мобилизация, отток наличных",
     "en": "Partial mobilisation, cash outflow", "scope": "money"},
    {"date": "2023-08-31", "ru": "Ослабление рубля, внеочередное повышение ставки",
     "en": "Rouble weakening, emergency rate hike", "scope": "money"},
]


# ======================================================================
#  PASSPORT
#  ПАСПОРТ
# ======================================================================
def new_passport(indicator_id, name_ru, name_en="", source="", units="", frequency="M",
                 indicator_type="flow_value", input_format="level", run_id=None,
                 table=None, period=12, forecast_horizon=12, engine="x11"):
    """Build a passport skeleton with starting values taken from the indicator type.

    The passport has three parts that must not be mixed up:
      spec          -- what is executed (used by apply_passport);
      justification -- why it was decided so (read by humans);
      results       -- what came out (reference only, never used when applying).

    [RU] Собирает заготовку паспорта со стартовыми значениями по типу показателя.

    У паспорта три части, которые нельзя смешивать:
      spec          -- то, что исполняется (используется apply_passport);
      justification -- почему так решили (читают люди);
      results       -- что получилось (справочно, при применении не используется).
    """
    d = INDICATOR_TYPES.get(indicator_type)
    if d is None:
        raise ValueError(tr(f"Неизвестный тип показателя: {indicator_type}. Доступно: {list(INDICATOR_TYPES)}",
                            f"Unknown indicator type: {indicator_type}. Available: {list(INDICATOR_TYPES)}"))
    today = datetime.date.today().isoformat()
    run_id = run_id or f"{today.replace('-', '')}_{indicator_id}_sa"
    return {
        "schema_version": PASSPORT_SCHEMA_VERSION,
        "indicator": {"id": indicator_id, "table": table or indicator_id,
                      "name": {"ru": name_ru, "en": name_en or name_ru},
                      "source": source, "units": units, "frequency": frequency,
                      "type": indicator_type, "input_format": input_format},
        "run": {"run_id": run_id, "calc_date": today, "data_cutoff": None, "status": "draft",
                "parent_run": None, "valid_from": None, "valid_to": None,
                "module_version": __version__, "author": os.getenv("USER", "")},
        "spec": {
            "transform": d["transform"],
            "preprocess": {
                "calendar": {
                    "trading_day": {"use": d["trading_day"], "calendar_table": None, "fallback_simple": True,
                                # "flow" -- число рабочих дней за месяц; "stock" -- день недели
                                # отчётной даты (tdstock[w]); см. раздел про календарь
                                "kind": "stock" if indicator_type in ("stock", "balance") else "flow",
                                "stock_day": 31},
                    "easter": {"use": d["easter"], "tradition": DEFAULT_EASTER_TRADITION, "window": DEFAULT_EASTER_WINDOW},
                },
                "outliers": {"mode": ENGINE_REGISTRY[engine]["outliers_default"], "list": [],
                             "cv": DEFAULT_OUTLIER_CV, "types": list(DEFAULT_OUTLIER_TYPES),
                             "skip_edge": 6, "gls": DEFAULT_GLS, "ar_order": 1},
            },
            "decompose": {"engine": engine, "period": period, "params": {}},
            "output": {"saar_mode": d["saar_mode"], "forecast": {"horizon": forecast_horizon}},
            "sample": {"start": None, "end": None,
                       # estimate the model on this interval, adjust the whole series
                       # модель оценивается на этом интервале, корректируется весь ряд
                       "model_span": {"start": None, "end": None, "origin": None, "reason": ""}},
            "update_policy": "flag_tail",
        },
        "justification": {"transform": "", "calendar": "", "outliers": [], "method": "", "sample": ""},
        "results": {},
    }


def validate_passport(passport):
    """Check a passport against the engine registry.

    Returns (errors, warnings). Errors block execution; warnings are things worth knowing:
    the most important one is a setting that IS specified but will be IGNORED by the chosen
    engine -- silently ignoring it would be the worst outcome, because the analyst would be
    sure the setting had been taken into account.

    [RU] Проверяет паспорт по реестру движков.

    Возвращает (ошибки, предупреждения). Ошибки блокируют расчёт; предупреждения -- то, о чём
    стоит знать, и главное из них -- настройка, которая ЗАДАНА, но будет ПРОИГНОРИРОВАНА
    выбранным движком: молча игнорировать её хуже всего, потому что аналитик будет уверен,
    что учёл её.
    """
    e, w = [], []
    p = passport
    if p.get("schema_version") != PASSPORT_SCHEMA_VERSION:
        w.append(tr(f"Паспорт схемы {p.get('schema_version')}, текущая {PASSPORT_SCHEMA_VERSION} -- будет мигрирован.",
                    f"Passport of schema {p.get('schema_version')}, current is {PASSPORT_SCHEMA_VERSION} -- it will be migrated."))
    spec = p.get("spec", {})
    dec = spec.get("decompose", {})
    eng = dec.get("engine")
    reg = ENGINE_REGISTRY.get(eng)
    if reg is None:
        e.append(tr(f"Неизвестный движок: {eng}. Доступно: {list(ENGINE_REGISTRY)}",
                    f"Unknown engine: {eng}. Available: {list(ENGINE_REGISTRY)}"))
        return e, w
    if not reg["pipeline"]:
        e.append(tr(f"Движок '{eng}' реализован в модуле, но в версии {__version__} не подключён к "
                    f"analyze_series() -- применить паспорт с ним нельзя.",
                    f"Engine '{eng}' is implemented in the module but in version {__version__} is not wired "
                    f"into analyze_series() -- a passport with it cannot be applied."))
    per, kind = dec.get("period"), reg["period"]
    if kind == "int" and not (isinstance(per, int) and per > 1):
        e.append(tr(f"Движок '{eng}' требует целый период, получено: {per!r}",
                    f"Engine '{eng}' requires an integer period, got: {per!r}"))
    if kind == "int_12_4" and per not in (12, 4):
        e.append(tr(f"Движок '{eng}' работает только с period=12 или 4, получено: {per!r}",
                    f"Engine '{eng}' works only with period=12 or 4, got: {per!r}"))
    if kind == "list_int" and not (isinstance(per, (list, tuple)) and len(per) >= 2):
        e.append(tr(f"Движок '{eng}' требует список периодов, например [7, 365].",
                    f"Engine '{eng}' requires a list of periods, e.g. [7, 365]."))
    if spec.get("transform") not in ("auto", "log", "none"):
        e.append(tr(f"transform должен быть 'auto', 'log' или 'none', получено: {spec.get('transform')!r}",
                    f"transform must be 'auto', 'log' or 'none', got: {spec.get('transform')!r}"))
    outl = spec.get("preprocess", {}).get("outliers", {})
    if outl.get("mode") not in ("auto", "manual", "none"):
        e.append(tr(f"outliers.mode должен быть 'auto', 'manual' или 'none', получено: {outl.get('mode')!r}",
                    f"outliers.mode must be 'auto', 'manual' or 'none', got: {outl.get('mode')!r}"))
    for o in outl.get("list", []):
        if o.get("type") not in ("AO", "LS", "TC", "SO"):
            e.append(tr(f"Неизвестный тип выброса {o.get('type')!r} на {o.get('date')}",
                        f"Unknown outlier type {o.get('type')!r} at {o.get('date')}"))
        try:
            pd.Timestamp(o.get("date"))
        except Exception:
            e.append(tr(f"Не разбирается дата выброса: {o.get('date')!r}",
                        f"Cannot parse the outlier date: {o.get('date')!r}"))
    if outl.get("mode") == "none" and outl.get("list"):
        w.append(tr("outliers.mode='none', но список выбросов не пуст -- он будет проигнорирован.",
                    "outliers.mode='none', but the outlier list is not empty -- it will be ignored."))
    if outl.get("mode") == "manual" and not outl.get("list"):
        w.append(tr("outliers.mode='manual' при пустом списке -- выбросы не будут учтены вовсе.",
                    "outliers.mode='manual' with an empty list -- no outliers will be taken into account at all."))
    cal = spec.get("preprocess", {}).get("calendar", {})
    if (reg["preprocess"] == "engine" and cal.get("trading_day", {}).get("use")
            and cal.get("trading_day", {}).get("calendar_table")):
        w.append(tr("Свой производственный календарь передаётся в x13as как user-регрессор типа td; "
                    "он должен покрывать и горизонт прогноза.",
                    "A custom working calendar is passed to x13as as a user regressor of type td; "
                    "it must also cover the forecast horizon."))
    if eng in ("prophet", "tbats") and cal.get("trading_day", {}).get("use"):
        w.append(tr(f"Календарный регрессор будет вычтен на шаге препроцессинга, но внутрь '{eng}' не попадёт.",
                    f"The calendar regressor will be subtracted at the pre-processing step, but will not go inside '{eng}'."))
    if spec.get("output", {}).get("saar_mode") not in (None, "level_ma", "level_3m", "rate_avg"):
        e.append(tr(f"saar_mode должен быть 'level_ma', 'level_3m', 'rate_avg' или null.",
                    f"saar_mode must be 'level_ma', 'level_3m', 'rate_avg' or null."))
    if any(o.get("type") == "SO" for o in outl.get("list", [])) and eng != "x13_real":
        w.append(tr("Сезонные выбросы (SO) поддерживает только движок x13_real -- в -lite методах "
                    "они будут проигнорированы.",
                    "Seasonal outliers (SO) are supported only by the x13_real engine -- the -lite "
                    "methods will ignore them."))
    ms = spec.get("sample", {}).get("model_span", {})
    if ms.get("start") or ms.get("end"):
        if eng != "x13_real":
            w.append(tr("Обучающая выборка (model_span) применяется только движком x13_real; "
                        "-lite методы оценивают модель по всей выборке.",
                        "The model span is applied only by the x13_real engine; the -lite methods "
                        "estimate the model on the whole sample."))
        if not ms.get("origin"):
            w.append(tr("Задана обучающая выборка без origin: укажите 'expert' или 'computed' и "
                        "причину -- иначе через год будет непонятно, откуда взялся интервал.",
                        "A model span is set without origin: state 'expert' or 'computed' and the "
                        "reason -- otherwise in a year it will be unclear where the interval came from."))
    if spec.get("update_policy") not in ("frozen", "flag_tail", "refit"):
        e.append(tr(f"update_policy должен быть 'frozen', 'flag_tail' или 'refit'.",
                    f"update_policy must be 'frozen', 'flag_tail' or 'refit'."))
    if p.get("indicator", {}).get("input_format") not in ("level", "mom", "qoq", "ytd_cumulative"):
        e.append(tr("input_format должен быть 'level', 'mom', 'qoq' или 'ytd_cumulative'.",
                    "input_format must be 'level', 'mom', 'qoq' or 'ytd_cumulative'."))
    return e, w


# Chain of migrations: each function raises the passport by exactly one step.
# A minor version (1.0 -> 1.1) only adds optional fields and needs no function here.
# Цепочка миграций: каждая функция поднимает паспорт ровно на одну ступень.
# Минорная версия (1.0 -> 1.1) только добавляет необязательные поля и функции здесь не требует.
_PASSPORT_MIGRATIONS = {}


def migrate_passport(passport):
    """Bring a passport to the current schema IN MEMORY. The file on disk and the row in the
    database are never rewritten: a run made in 2026 stays a run of the schema it was made with,
    otherwise a historical snapshot would stop being historical.

    [RU] Приводит паспорт к текущей схеме В ПАМЯТИ. Файл на диске и строка в базе не
    переписываются: пересмотр, сделанный в 2026 году, остаётся пересмотром своей схемы, иначе
    исторический срез перестанет быть историческим.
    """
    p = copy.deepcopy(passport)
    seen = set()
    while p.get("schema_version") != PASSPORT_SCHEMA_VERSION:
        v = p.get("schema_version")
        if v in seen or v not in _PASSPORT_MIGRATIONS:
            raise ValueError(tr(
                f"Не могу мигрировать паспорт схемы {v!r} до {PASSPORT_SCHEMA_VERSION}: нет функции миграции. "
                f"Нужна более новая версия seasonal_toolbox.",
                f"Cannot migrate a passport of schema {v!r} to {PASSPORT_SCHEMA_VERSION}: no migration function. "
                f"A newer version of seasonal_toolbox is required."))
        seen.add(v)
        p = _PASSPORT_MIGRATIONS[v](p)
    return p


def passport_summary(passport):
    """One-screen summary of a passport: what will be done to the series and why.

    [RU] Паспорт на одном экране: что будет сделано с рядом и почему.
    """
    p = migrate_passport(passport)
    s, ind, run = p["spec"], p["indicator"], p["run"]
    cal = s["preprocess"]["calendar"]
    outl = s["preprocess"]["outliers"]
    L = []
    L.append(tr(f"Настройки показателя ({run['run_id']})", f"Indicator settings ({run['run_id']})"))
    L.append(tr(f"  показатель : {ind['id']} -- {tx(ind['name'])} ({ind['type']}, {ind['frequency']}, вход: {ind['input_format']})",
                f"  indicator  : {ind['id']} -- {tx(ind['name'])} ({ind['type']}, {ind['frequency']}, input: {ind['input_format']})"))
    L.append(tr(f"  данные до  : {run['data_cutoff']}   расчёт: {run['calc_date']}",
                f"  data up to : {run['data_cutoff']}   calculated: {run['calc_date']}"))
    L.append(tr(f"  модель     : transform={s['transform']}, движок={s['decompose']['engine']}, период={s['decompose']['period']}",
                f"  model      : transform={s['transform']}, engine={s['decompose']['engine']}, period={s['decompose']['period']}"))
    td_kind = cal["trading_day"].get("kind", "flow")
    td_txt = (f"tdstock[{cal['trading_day'].get('stock_day', 31)}]" if td_kind == "stock" else "td")
    L.append(tr(f"  календарь  : рабочие дни={cal['trading_day']['use']} ({td_txt}), "
                f"Пасха={cal['easter']['use']} ({cal['easter']['tradition']})",
                f"  calendar   : trading days={cal['trading_day']['use']} ({td_txt}), "
                f"Easter={cal['easter']['use']} ({cal['easter']['tradition']})"))
    ms = s["sample"].get("model_span", {})
    if ms.get("start") or ms.get("end"):
        L.append(tr(f"  обучение   : {ms.get('start') or '...'} .. {ms.get('end') or '...'} "
                    f"[{ms.get('origin') or '?'}] {ms.get('reason', '')}",
                    f"  model span : {ms.get('start') or '...'} .. {ms.get('end') or '...'} "
                    f"[{ms.get('origin') or '?'}] {ms.get('reason', '')}"))
    L.append(tr(f"  выбросы    : режим={outl['mode']}, задано вручную={len(outl['list'])}",
                f"  outliers   : mode={outl['mode']}, set by hand={len(outl['list'])}"))
    for o in outl["list"]:
        L.append(f"      {o['date']}  {o['type']:3s}  {o.get('status', '')}  {o.get('reason', '')}")
    L.append(tr(f"  на выходе  : SAAR={s['output']['saar_mode']}, прогноз={s['output']['forecast']['horizon']} мес., "
                f"политика обновления={s['update_policy']}",
                f"  output     : SAAR={s['output']['saar_mode']}, forecast={s['output']['forecast']['horizon']} months, "
                f"update policy={s['update_policy']}"))
    return "\n".join(L)



# ======================================================================
#  RUNS: FOLDER, SNAPSHOTS, FILES
#  ПЕРЕСМОТРЫ: ПАПКА, СНИМКИ, ФАЙЛЫ
# ======================================================================












"""Что делать с записью в утверждённый пересмотр: "raise" или "skip".

Утверждённый пересмотр неизменяем -- иначе теряется смысл слова «утверждённый»: непонятно,
что именно было опубликовано в тот день. Но запретить нужно ПЕРЕЗАПИСЬ, а не запуск: открыть
утверждённый пересмотр, пересчитать его и посмотреть графики -- законное и нужное действие,
ради него всё и хранится.

"raise" -- попытка записи бросает исключение (по умолчанию: защищает от случайной порчи);
"skip"  -- запись молча пропускается, расчёт идёт целиком. Это режим просмотра, который
           блокнот включает сам, увидев утверждённый пересмотр.

[EN] What to do about writing into an approved run: "raise" or "skip". An approved run is
immutable, but re-running it to look at the charts must stay possible -- that is what it is
kept for. "skip" is the read-only replay mode the notebook switches on by itself.
"""








def save_passport(passport, path=None, note=""):
    """Сохранить настройки показателя в JSON рядом с блокнотом.

    Паспорт -- это словарь решений: логарифм, календарь, выбросы, метод, горизонт прогноза.
    Смысл файла в воспроизводимости: через полгода будет видно не только какие цифры
    получились, но и почему они такие. Имя по умолчанию -- <id показателя>_passport.json.

    [EN] Save the indicator settings as JSON next to the notebook. The passport is a dictionary
    of decisions -- transform, calendar, outliers, method, forecast horizon -- kept so that in
    six months it is clear not only what the numbers were but why.
    """
    p = migrate_passport(passport)
    path = path or f"{p['indicator']['id']}_passport.json"
    p["run"]["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    if note:
        p["run"]["note"] = note
    json.dump(p, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return path



def load_passport_file(path):
    """Read a passport from a file and migrate it to the current schema in memory.

    [RU] Читает паспорт из файла и мигрирует его к текущей схеме в памяти.
    """
    return migrate_passport(json.load(open(path, encoding="utf-8")))




# ======================================================================
#  CHARTS OF A RUN
#  ГРАФИКИ ПЕРЕСМОТРА
# ======================================================================














# ======================================================================
#  DATABASE: ONE TABLE WITH SPECIFICATIONS
#  БАЗА: ОДНА ТАБЛИЦА СО СПЕЦИФИКАЦИЯМИ
# ======================================================================




from contextlib import contextmanager as _contextmanager






















# ======================================================================
#  LOADING AN INDICATOR AND THE WORKING CALENDAR
#  ЗАГРУЗКА ПОКАЗАТЕЛЯ И ПРОИЗВОДСТВЕННОГО КАЛЕНДАРЯ
# ======================================================================
def load_indicator(table, path=None, date_col="Date", value_col="Value"):
    """Load one indicator: one indicator -- one table (or one Excel file of the same name).

    Returns a pd.Series with a DatetimeIndex sorted ascending. The data are read from the Excel
    or CSV file <path>, or from <table>.xlsx next to the notebook.

    [RU] Загружает один показатель: один индикатор -- один Excel- или CSV-файл.

    Возвращает pd.Series с возрастающим DatetimeIndex. Данные читаются из файла <path> либо из
    <table>.xlsx рядом с блокнотом.
    """
    path = path or f"{table}.xlsx"
    if str(path).lower().endswith((".csv", ".txt")):
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_excel(path)
        except ImportError as ex:
            # A frequent cause: openpyxl was upgraded in a different interpreter than the
            # kernel runs in. Either install it with the %pip magic and restart the kernel,
            # or save the file as CSV -- this function reads CSV too.
            # Частая причина: openpyxl обновлён не в том интерпретаторе, в котором работает
            # ядро. Либо ставьте его магией %pip и перезапустите ядро, либо сохраните файл
            # как CSV -- эта функция читает и CSV.
            raise ImportError(tr(
                f"Не удалось прочитать '{path}': {ex}\n"
                f"Проверьте в блокноте: import sys, openpyxl; print(sys.executable, openpyxl.__version__).\n"
                f"Если версия старая -- выполните %pip install --upgrade openpyxl и ПЕРЕЗАПУСТИТЕ ядро.\n"
                f"Обходной путь: сохраните файл как {os.path.splitext(str(path))[0]}.csv -- "
                f"load_indicator() читает CSV без openpyxl.",
                f"Could not read '{path}': {ex}\n"
                f"Check in the notebook: import sys, openpyxl; print(sys.executable, openpyxl.__version__).\n"
                f"If the version is old -- run %pip install --upgrade openpyxl and RESTART the kernel.\n"
                f"Workaround: save the file as {os.path.splitext(str(path))[0]}.csv -- "
                f"load_indicator() reads CSV without openpyxl.")) from ex
    if date_col not in df.columns or value_col not in df.columns:
        raise ValueError(tr(f"Ожидались колонки '{date_col}' и '{value_col}', получено: {list(df.columns)}",
                            f"Expected columns '{date_col}' and '{value_col}', got: {list(df.columns)}"))
    s = pd.Series(pd.to_numeric(df[value_col]).values,
                  index=pd.DatetimeIndex(pd.to_datetime(df[date_col])), name=table).sort_index()
    return s.dropna()


def load_working_days(table="RU_TDAYSWKPC_M_GV", path=None):
    """Working calendar as {(year, month): working_days} -- exactly the format expected by
    trading_day_regressor() and analyze_series(working_days_calendar=...).

    The calendar must also cover the forecast horizon: X-13 does not extend a user regressor itself.

    [RU] Производственный календарь в виде {(год, месяц): рабочих_дней} -- ровно тот формат,
    который ждут trading_day_regressor() и analyze_series(working_days_calendar=...).

    Календарь должен покрывать и горизонт прогноза: пользовательский регрессор X-13 сам не продлевает.
    """
    s = load_indicator(table, path=path)
    return {(d.year, d.month): int(v) for d, v in s.items()}


def calendar_coverage(series, calendar):
    """Check that the calendar covers the series and the forecast horizon.

    A frequent situation: the indicator starts in the 1990s while the calendar starts in 2008.
    Then either the sample is trimmed to the calendar, or the simplified Mon-Fri count is used on
    the early part -- this is a decision for the passport, not something to be resolved silently.

    [RU] Проверяет, покрывает ли календарь ряд и горизонт прогноза.

    Частая ситуация: показатель начинается в 1990-х, а календарь -- с 2008 года. Тогда либо режем
    выборку по календарю, либо на раннем отрезке используем упрощённый подсчёт Пн-Пт -- это решение
    для паспорта, а не то, что стоит решать молча.
    """
    idx = pd.DatetimeIndex(pd.Series(series).index)
    have = set(calendar)
    missing = [d for d in idx if (d.year, d.month) not in have]
    cal_idx = sorted(have)
    return {"n_missing": len(missing),
            "missing_first": str(missing[0].date()) if missing else None,
            "missing_last": str(missing[-1].date()) if missing else None,
            "calendar_from": f"{cal_idx[0][0]}-{cal_idx[0][1]:02d}" if cal_idx else None,
            "calendar_to": f"{cal_idx[-1][0]}-{cal_idx[-1][1]:02d}" if cal_idx else None,
            "series_from": str(idx[0].date()), "series_to": str(idx[-1].date())}


def x13_available(binary=None):
    """Is the real x13as binary available? Used to decide automatically whether to include
    "x13_real" in a comparison of methods: on a machine without the binary the comparison must
    not fall over, and on a machine with it the official engine must not be silently skipped.

    [RU] Доступен ли настоящий бинарник x13as? Используется, чтобы автоматически решать, включать
    ли "x13_real" в сравнение методов: на машине без бинарника сравнение не должно падать, а на
    машине с ним официальный движок не должен молча выпадать.
    """
    try:
        path = _resolve_x13_binary(binary or X13_BIN)
        return bool(path) and os.path.exists(path)
    except Exception:
        return False



# ======================================================================
#  APPLYING A SPECIFICATION
#  ПРИМЕНЕНИЕ СПЕЦИФИКАЦИИ
# ======================================================================
def _outliers_frame(spec_list, index):
    """Outliers are stored as DATES, not positions: a position shifts as soon as the series starts
    a month earlier, while a date stays correct on any span.

    [RU] Выбросы хранятся ДАТАМИ, а не позициями: позиция сдвигается, стоит ряду начаться на месяц
    раньше, а дата остаётся верной на любом отрезке.
    """
    rows, missed = [], []
    idx = pd.DatetimeIndex(index)
    for o in spec_list or []:
        ts = pd.Timestamp(o["date"])
        if ts in idx:
            rows.append({"date": ts, "type": o["type"]})
        else:
            near = idx[(idx.year == ts.year) & (idx.month == ts.month)]
            if len(near):
                rows.append({"date": near[0], "type": o["type"]})
            else:
                missed.append(o["date"])
    return pd.DataFrame(rows, columns=["date", "type"]), missed


def passport_to_kwargs(passport, working_days_calendar=None):
    """Translate a specification into the arguments of analyze_series(). One place where the
    passport meets the version 1.0 pipeline, so that the mapping is not scattered over notebooks.

    [RU] Переводит спецификацию в аргументы analyze_series(). Единственное место, где паспорт
    встречается с пайплайном версии 1.0, чтобы это соответствие не расползлось по блокнотам.
    """
    p = migrate_passport(passport)
    s = p["spec"]
    cal = s["preprocess"]["calendar"]
    outl = s["preprocess"]["outliers"]
    dec = s["decompose"]
    kw = {
        "input_format": p["indicator"]["input_format"],
        "period": dec["period"] if isinstance(dec["period"], int) else 12,
        "method": dec["engine"],
        "transform": s["transform"],
        "calendar_adjust": bool(cal["trading_day"]["use"] or cal["easter"]["use"]),
        "trading_day": bool(cal["trading_day"]["use"]),
        "easter_window": cal["easter"].get("window", DEFAULT_EASTER_WINDOW),
        "easter_tradition": cal["easter"].get("tradition", DEFAULT_EASTER_TRADITION),
        # The calendar goes to the engine ONLY if the passport switched it on. Otherwise
        # analyze_series builds the user td regressor for x13_real just because a calendar was
        # passed -- and the specification would be quietly violated.
        # Календарь уходит в движок ТОЛЬКО если он включён в паспорте. Иначе analyze_series
        # построит для x13_real user-регрессор td просто потому, что календарь передан, --
        # и спецификация будет тихо нарушена.
        "working_days_calendar": working_days_calendar if cal["trading_day"]["use"] else None,
        "outlier_cv": outl.get("cv", DEFAULT_OUTLIER_CV),
        "outlier_types": tuple(outl.get("types", DEFAULT_OUTLIER_TYPES)),
        "outlier_skip_edge": outl.get("skip_edge", 6),
        "gls": outl.get("gls", DEFAULT_GLS),
        "ar_order": outl.get("ar_order", 1),
        "forecast_horizon": s["output"]["forecast"].get("horizon", 12),
        "saar_mode": s["output"].get("saar_mode") or "level_ma",
    }
    if dec["engine"] == "x13_real":
        arima = dec["params"].get("arima")
        kw["arima_order"] = None if arima in (None, "automdl") else arima
    # Easter alone without trading days is still a calendar adjustment; the pipeline flag
    # calendar_adjust switches the block as a whole, trading_day switches only the working days.
    # Пасха без торговых дней -- тоже календарная корректировка; флаг calendar_adjust включает
    # блок целиком, trading_day -- только рабочие дни.
    return kw


def apply_passport(series, passport, working_days_calendar=None, plot_mode=None,
                   update_policy=None, saar_n_sim=2000):
    """Apply a specification to data. This is the interface external models use: the notebook of
    an indicator produces the passport, everything else only executes it.

    update_policy decides what happens to observations that appeared after data_cutoff:
      "frozen"    -- the specification and the outliers are fixed, coefficients are re-estimated,
                     no new outliers are searched for;
      "flag_tail" -- the same, but the new tail is searched and findings are returned as warnings
                     rather than applied (the default: the model must not silently change the
                     analyst's decisions, nor miss a fresh shock);
      "refit"     -- full re-estimation, including the transformation and the outlier search.

    Returns the usual analyze_series() dictionary plus "warnings" and "applied_policy".

    [RU] Применяет спецификацию к данным. Это и есть интерфейс для внешних моделей: блокнот
    показателя рождает паспорт, всё остальное его только исполняет.

    update_policy решает, что происходит с наблюдениями после data_cutoff:
      "frozen"    -- спецификация и выбросы зафиксированы, коэффициенты переоцениваются, новых
                     выбросов не ищем;
      "flag_tail" -- то же, но на новом хвосте запускается поиск, и найденное возвращается
                     предупреждением, а не применяется (по умолчанию: модель не должна молча
                     менять решения аналитика, но и пропускать свежий шок нельзя);
      "refit"     -- полная переоценка, включая трансформацию и поиск выбросов.
    """
    p = migrate_passport(passport)
    err, warn = validate_passport(p)
    if err:
        raise ValueError(tr("Паспорт не прошёл проверку:\n  ", "The passport failed validation:\n  ")
                         + "\n  ".join(err))
    warnings_out = list(warn)

    s = pd.Series(series).sort_index()
    sample = p["spec"].get("sample", {})
    if sample.get("start"):
        s = s[s.index >= pd.Timestamp(sample["start"])]
    if sample.get("end"):
        s = s[s.index <= pd.Timestamp(sample["end"])]

    policy = update_policy or p["spec"].get("update_policy", "flag_tail")
    cutoff = p["run"].get("data_cutoff")
    n_new = int((s.index > pd.Timestamp(cutoff)).sum()) if cutoff else 0
    if n_new:
        warnings_out.append(tr(
            f"В данных {n_new} наблюдений после data_cutoff ({cutoff}); политика обновления: {policy}.",
            f"The data contain {n_new} observations after data_cutoff ({cutoff}); update policy: {policy}."))

    kw = passport_to_kwargs(p, working_days_calendar)
    mode = p["spec"]["preprocess"]["outliers"]["mode"]
    if policy == "refit":
        kw["transform"] = "auto"
        kw["auto_outliers"] = True
        kw["outliers"] = None
    elif mode == "manual":
        df, missed = _outliers_frame(p["spec"]["preprocess"]["outliers"]["list"], s.index)
        kw["auto_outliers"] = False
        kw["outliers"] = df if len(df) else None
        if missed:
            warnings_out.append(tr(f"Выбросы вне выборки и потому не применены: {missed}",
                                   f"Outliers outside the sample and therefore not applied: {missed}"))
    elif mode == "none":
        kw["auto_outliers"] = False
        kw["outliers"] = None
    else:
        kw["auto_outliers"] = True
        kw["outliers"] = None

    if p["spec"]["decompose"]["engine"] == "x13_real":
        # The production path: our own spec builder instead of the version 1.0 branch. It wraps
        # the data by line width, passes the analyst's manual outliers to the engine and extends
        # the seasonal factors over the horizon, which gives a forecast of the SA series.
        # Производственный путь: свой сборщик spec вместо ветки версии 1.0. Он переносит данные по
        # ширине строки, передаёт движку выбросы, заданные аналитиком, и продлевает сезонные
        # факторы на горизонт -- это даёт прогноз SA-ряда.
        res = x13_run_passport(s, p, working_days_calendar=working_days_calendar,
                               saar_n_sim=saar_n_sim)
        res["warnings"] = warnings_out
        res["applied_policy"] = policy
        res["passport"] = p
        return res

    if False:
        # Remove the temporary files of a previous x13as run. Without this, a failed run leaves
        # the OLD .d11 on disk, read_d11() reads it and the length silently stops matching the
        # series -- which is exactly how a failure of the engine turns into an incomprehensible
        # "operands could not be broadcast" instead of the report from x13as.
        # Удаляем временные файлы прошлого прогона x13as. Без этого неудачный прогон оставляет
        # на диске СТАРЫЙ .d11, read_d11() читает его, и длина молча перестаёт совпадать с рядом --
        # именно так сбой движка превращается в невнятное "operands could not be broadcast"
        # вместо отчёта x13as.
        for f in glob.glob(os.path.join(X13_WORKDIR, "analyze_series_tmp.*")) + \
                 glob.glob(os.path.join(".", "analyze_series_tmp.*")):
            try:
                os.remove(f)
            except OSError:
                pass

    res = analyze_series(s, plot_mode=plot_mode, saar_n_sim=saar_n_sim, **kw)

    if policy == "flag_tail" and n_new and mode != "auto":
        tail = outlier_candidates(s, skip_end=0, cv=p["spec"]["preprocess"]["outliers"].get("cv", DEFAULT_OUTLIER_CV))
        known = {pd.Timestamp(o["date"]) for o in p["spec"]["preprocess"]["outliers"]["list"]}
        fresh = [r for _, r in tail.iterrows()
                 if pd.Timestamp(r["date"]) > pd.Timestamp(cutoff) and pd.Timestamp(r["date"]) not in known]
        for r in fresh:
            warnings_out.append(tr(
                f"На новом хвосте найден кандидат в выбросы: {r['date']} {r['type']} (t={r['t_stat']:.1f}, "
                f"эффект {r['effect_pct']:+.1f}%) -- не применён, требуется решение аналитика.",
                f"A candidate outlier was found on the new tail: {r['date']} {r['type']} (t={r['t_stat']:.1f}, "
                f"effect {r['effect_pct']:+.1f}%) -- not applied, the analyst has to decide."))

    mv = p["run"].get("module_version")
    if mv and mv.split(".")[0] != __version__.split(".")[0]:
        warnings_out.append(tr(
            f"Паспорт посчитан модулем {mv}, сейчас {__version__}: спецификация применена, но побитовое "
            f"совпадение чисел не гарантируется.",
            f"The passport was computed by module {mv}, now {__version__}: the specification has been applied, "
            f"but a bit-for-bit match of the numbers is not guaranteed."))

    res["warnings"] = warnings_out
    res["applied_policy"] = policy
    res["passport"] = p
    return res




# ======================================================================
#  NEW DIAGNOSTICS
#  НОВЫЕ ДИАГНОСТИКИ
# ======================================================================
def _detrend_si(y, period):
    """SI ratios: the series minus a centred moving average. This is what the official F-test for
    stable seasonality is computed on -- unlike a linear detrending, a moving average follows any
    trend, including an exponential one.

    [RU] SI-отношения: ряд минус центрированное скользящее среднее. Именно по ним считается
    официальный F-тест на устойчивую сезонность -- в отличие от линейного детрендирования,
    скользящее среднее следует за любым трендом, в том числе экспоненциальным.
    """
    y = np.asarray(y, dtype=float)
    trend = apply_symmetric_filter(y, centered_ma_weights(period))
    return y - trend


def has_seasonality(y, dates, period=12, alpha=0.05, transform="auto"):
    """Is there seasonality at all? If there is none, the series must not be adjusted: the
    procedure would only add noise. Happens with some financial stocks.

    The test runs on SI ratios and, for a series that grows several-fold, on the logarithm --
    otherwise the curvature of the trend eats the whole signal and the test reports "no
    seasonality" where it plainly exists.

    [RU] Есть ли сезонность вообще? Если её нет, ряд корректировать не нужно: процедура только
    добавит шума. Бывает с некоторыми финансовыми запасами.

    Тест считается по SI-отношениям и, для ряда, выросшего в разы, по логарифму -- иначе кривизна
    тренда съедает весь сигнал и тест рапортует «сезонности нет» там, где она очевидна.
    """
    y = np.asarray(y, dtype=float)
    used = transform
    if transform == "auto":
        used = choose_transform(y, period)
    z = np.log(y) if (used == "log" and np.nanmin(y) > 0) else y
    si = _detrend_si(z, period)
    ok = np.isfinite(si)
    f = seasonal_dummy_ftest(si[ok], pd.DatetimeIndex(dates)[ok], period)
    share = seasonal_power_share(si[ok], period)
    a = acf(si[ok], 2 * period)
    verdict = bool(f["p_value"] < alpha)
    return {"transform_used": used, "F": f["F"], "p_value": f["p_value"],
            "acf_seasonal_lag": float(a[period]), "power_share": float(share),
            "has_seasonality": verdict,
            "comment": tr("Сезонность значима -- корректировка имеет смысл." if verdict else
                          "Сезонность не обнаружена -- корректировать ряд не нужно.",
                          "Seasonality is significant -- adjustment makes sense." if verdict else
                          "No seasonality detected -- the series should not be adjusted.")}


def calendar_significance(y, dates, period=12, n_harm=2, working_days_calendar=None,
                          easter_tradition=None, easter_window=None):
    """Check the calendar regressors SEPARATELY: working days and Easter are different effects and
    are decided on separately.

    Easter does not belong to the working calendar: it always falls on a Sunday and therefore does
    not change the number of working days; its effect is a shift of demand between months.

    For each regressor the table gives the coefficient, the t-statistic and the change in AICC when
    it is added -- plus the verdict. The final decision is also an economic one: for flows (output,
    retail, freight) working days are usually significant, for stocks at a date they usually are not.

    [RU] Проверяет календарные регрессоры ПО ОТДЕЛЬНОСТИ: рабочие дни и Пасха -- разные эффекты,
    и решение по ним принимается раздельно.

    Пасха не входит в производственный календарь: она всегда приходится на воскресенье и потому не
    меняет число рабочих дней; её эффект -- сдвиг спроса между месяцами.

    По каждому регрессору таблица даёт коэффициент, t-статистику и изменение AICC при его
    добавлении, плюс вердикт. Окончательное решение ещё и экономическое: для потоков (выпуск,
    розница, перевозки) рабочие дни обычно значимы, для запасов на дату -- обычно нет.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    dates = pd.DatetimeIndex(dates)
    base, _ = _harmonic_trend_cols(n, period, n_harm)
    tradition = easter_tradition or DEFAULT_EASTER_TRADITION
    window = easter_window or DEFAULT_EASTER_WINDOW
    regs = {
        "trading_day": trading_day_regressor(dates, working_days_calendar),
        "easter": easter_regressor(dates, L=window, tradition=tradition, center=True),
    }

    def _fit(cols):
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        k = X.shape[1]
        s2 = float(r @ r) / n
        ll = -0.5 * n * (np.log(2 * np.pi * max(s2, 1e-300)) + 1)
        aicc = -2 * ll + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1)
        return beta, r, aicc, X

    _, _, aicc0, _ = _fit(base)
    rows = []
    for name, col in regs.items():
        beta, r, aicc1, X = _fit(base + [col])
        dof = n - X.shape[1]
        s2 = float(r @ r) / max(dof, 1)
        se = np.sqrt(max(s2 * np.linalg.pinv(X.T @ X)[-1, -1], 1e-12))
        t = beta[-1] / se
        rows.append({"regressor": name, "coef": beta[-1], "t_stat": t,
                     "p_value": float(2 * stats.t.sf(abs(t), max(dof, 1))),
                     "aicc_delta": aicc1 - aicc0,
                     "verdict": tr("включать", "include") if (abs(t) >= 2 and aicc1 < aicc0)
                     else tr("не включать", "do not include")})
    return pd.DataFrame(rows)


def _search_outliers(y, base_cols, period=12, cv=None, types=None, alpha=None, min_gap=3,
                     max_outliers=8, skip_start=None, skip_end=0, ls_guard=2, gls=None, ar_order=1):
    """Outlier search with asymmetric edges -- what monitoring a fresh shock needs.

    Two differences from auto_outlier_search() of version 1.0, which stays untouched:

    1. The start and the end of the series are guarded separately. For current monitoring the end
       must stay open (that is where the freshest shock is), while the start does not need to be.
    2. LS and TC are forbidden in the first `ls_guard` observations. An LS dummy at the very first
       point is a column of ones, i.e. an exact copy of the constant: the regression degenerates
       and the t-statistic becomes meaningless (in practice it came out at 120 and dragged the
       whole greedy search after it). Real X-13 does not allow an LS in the first period either.

    [RU] Поиск выбросов с асимметричными краями -- то, что нужно для мониторинга свежего шока.

    Два отличия от auto_outlier_search() версии 1.0, которая остаётся нетронутой:

    1. Начало и конец ряда защищаются по отдельности. Для оперативного мониторинга конец должен
       оставаться открытым (там самый свежий шок), а начало -- нет.
    2. LS и TC запрещены в первых `ls_guard` наблюдениях. Дамми LS в самой первой точке -- это
       столбец из единиц, то есть точная копия константы: регрессия вырождается, а t-статистика
       теряет смысл (на практике она вышла равной 120 и утянула за собой весь жадный поиск).
       Настоящий X-13 тоже не допускает LS в первом периоде.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    cv = DEFAULT_OUTLIER_CV if cv is None else cv
    types = tuple(types or DEFAULT_OUTLIER_TYPES)
    gls = DEFAULT_GLS if gls is None else gls
    if alpha is None:
        alpha = 0.7 if period == 12 else 0.7 ** (12 / max(period, 1))
    skip_start = period // 2 if skip_start is None else skip_start
    X_base = np.column_stack(base_cols)

    phi = np.zeros(0)
    if gls:
        b0, *_ = np.linalg.lstsq(X_base, y, rcond=None)
        phi = ar_coefficients(y - X_base @ b0, ar_order)
    W = lambda v: whiten(v, phi)
    yw = W(y)
    Xw_base = np.column_stack([W(X_base[:, k]) for k in range(X_base.shape[1])])

    found, cols, cols_w = [], [], []
    cand = range(skip_start, n - skip_end)
    for _ in range(max_outliers):
        Xw_cur = np.column_stack([Xw_base] + cols_w) if cols_w else Xw_base
        best = None
        for i in cand:
            if any(abs(i - f[0]) < min_gap for f in found):
                continue
            for kind in types:
                if kind in ("LS", "TC") and i < ls_guard:
                    continue
                cw = W(_outlier_col(np.arange(n), i, kind, alpha))
                Xw = np.column_stack([Xw_cur, cw])
                beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
                r = yw - Xw @ beta
                dof = len(yw) - Xw.shape[1]
                if dof <= 0:
                    continue
                s2 = float(r @ r) / dof
                se = np.sqrt(max(s2 * np.linalg.pinv(Xw.T @ Xw)[-1, -1], 1e-12))
                t = beta[-1] / se
                if best is None or abs(t) > abs(best[2]):
                    best = (i, kind, t)
        if best is None or abs(best[2]) < cv:
            break
        found.append(best)
        c = _outlier_col(np.arange(n), best[0], best[1], alpha)
        cols.append(c); cols_w.append(W(c))

    dropped = []
    while found:
        Xw_full = np.column_stack([Xw_base] + cols_w)
        beta, *_ = np.linalg.lstsq(Xw_full, yw, rcond=None)
        r = yw - Xw_full @ beta
        dof = len(yw) - Xw_full.shape[1]
        if dof <= 0:
            break
        cov = float(r @ r) / dof * np.linalg.pinv(Xw_full.T @ Xw_full)
        nb = Xw_base.shape[1]
        ts = np.array([beta[nb + j] / np.sqrt(max(cov[nb + j, nb + j], 1e-12)) for j in range(len(found))])
        worst = int(np.argmin(np.abs(ts)))
        if abs(ts[worst]) >= cv:
            found = [(found[j][0], found[j][1], ts[j], beta[nb + j]) for j in range(len(found))]
            break
        dropped.append((found[worst][0], found[worst][1], ts[worst]))
        found.pop(worst); cols.pop(worst); cols_w.pop(worst)
    if found and len(found[0]) == 3:
        found = [(i, k, t, np.nan) for i, k, t in found]
    return {"outliers": found, "dropped_backward": dropped, "ar_coeffs": phi}


def outlier_candidates(series, period=12, n_harm=None, cv=None, types=None, skip_start=None,
                       skip_end=0, gls=None, ar_order=1, calendar_cols=None, events=None,
                       scale="auto"):
    """A table of candidate outliers for the analyst: date, type, t-statistic, size of the effect
    in percent, and a matching event from the shock reference book.

    This is exactly the place where the analyst adds their own knowledge: a news item, a Rosstat
    methodological note, a regulator's decision. Candidates in the last year are marked in the
    `edge` column -- a fresh shock cannot be reliably told from the start of a new trend, so such a
    decision is provisional and worth re-checking next month.

    scale="log" turns the coefficient into a percentage as exp(b)-1 (the usual case: the search runs
    on the logarithm); "level" takes it as a share of the median level; "auto" decides by the data.

    [RU] Таблица кандидатов в выбросы для аналитика: дата, тип, t-статистика, величина эффекта в
    процентах и совпавшее событие из справочника шоков.

    Это ровно то место, где аналитик добавляет своё знание: новость, методическое пояснение
    Росстата, решение регулятора. Кандидаты последнего года помечаются в колонке `edge`: свежий шок
    нельзя надёжно отличить от начала нового тренда, поэтому такое решение предварительно и его
    стоит перепроверить через месяц.

    scale="log" переводит коэффициент в проценты как exp(b)-1 (обычный случай: поиск идёт по
    логарифму); "level" считает долей от медианного уровня; "auto" решает по данным.
    """
    s = pd.Series(series).sort_index()
    y = np.asarray(s.values, dtype=float)
    idx = pd.DatetimeIndex(s.index)
    n = len(y)
    if scale == "auto":
        scale = "log" if (np.nanmax(np.abs(y)) < 50 and np.nanmin(y) > -50) else "level"
    # A full set of harmonics (period//2) instead of the two of the pipeline: with a base that is
    # too smooth, a sharp December peak does not fit the model and the algorithm declares EVERY
    # December an outlier. With period//2 harmonics any monthly shape is reproduced exactly.
    # Полный набор гармоник (period//2) вместо двух у пайплайна: при слишком гладкой базе острый
    # декабрьский пик не помещается в модель, и алгоритм объявляет выбросом КАЖДЫЙ декабрь.
    # При period//2 гармониках любая месячная форма воспроизводится точно.
    base, _ = _harmonic_trend_cols(n, period, period // 2 if n_harm is None else n_harm)
    if calendar_cols:
        base = base + list(calendar_cols)
    res = _search_outliers(y, base, period=period, cv=cv, types=types, skip_start=skip_start,
                           skip_end=skip_end, gls=gls, ar_order=ar_order)
    # Tolerance of one month: a shock in September shows up in the October figure.
    # Допуск в один месяц: шок сентября отзывается в октябрьской цифре.
    ev = {}
    for e in (events if events is not None else EVENTS_RU):
        base_per = pd.Timestamp(e["date"]).to_period("M")
        for shift in (0, 1, -1):
            ev.setdefault(base_per + shift, e)
    lvl = float(np.nanmedian(np.abs(y))) or 1.0
    rows = []
    for i, kind, t, beta in sorted(res["outliers"]):
        pct = 100 * (np.exp(beta) - 1) if scale == "log" else 100 * beta / lvl
        e = ev.get(idx[i].to_period("M"))
        rows.append({"date": str(idx[i].date()), "type": kind, "t_stat": float(t),
                     "effect": float(beta), "effect_pct": float(pct),
                     "edge": bool(i >= n - period),
                     "event": tx({"ru": e["ru"], "en": e["en"]}) if e else "",
                     "origin": "auto", "decision": "", "reason": ""})
    df = pd.DataFrame(rows, columns=["date", "type", "t_stat", "effect", "effect_pct",
                                     "edge", "event", "origin", "decision", "reason"])
    df.attrs["dropped_backward"] = [(str(idx[i].date()), k, float(t)) for i, k, t in res["dropped_backward"]]
    df.attrs["ar_coeffs"] = res["ar_coeffs"]
    df.attrs["scale"] = scale
    return df


def events_without_candidates(series, candidates, events=None, period=12):
    """The other side of the same check: events from the reference book that the algorithm did NOT
    react to. Sometimes that is correct (the shock did not touch this indicator), and sometimes it
    is a hint that the threshold is too strict.

    [RU] Обратная сторона той же проверки: события из справочника, на которые алгоритм НЕ
    отреагировал. Иногда это правильно (шок не затронул показатель), а иногда подсказка, что порог
    слишком строгий.
    """
    idx = pd.DatetimeIndex(pd.Series(series).index)
    have = set()
    for d in (candidates["date"] if len(candidates) else []):
        per = pd.Timestamp(d).to_period("M")
        have.update({per, per + 1, per - 1})
    span = (idx[0].to_period("M"), idx[-1].to_period("M"))
    rows = []
    for e in (events if events is not None else EVENTS_RU):
        per = pd.Timestamp(e["date"]).to_period("M")
        if span[0] <= per <= span[1] and per not in have:
            rows.append({"date": e["date"], "event": tx({"ru": e["ru"], "en": e["en"]}), "scope": e.get("scope", "")})
    return pd.DataFrame(rows, columns=["date", "event", "scope"])


def plot_month_subseries(series, period=12, dark=False, title=None, subtitle=None, transform=None):
    """Sub-series by month: all Januaries, all Februaries and so on, each as a line across years.

    The most informative single chart before any adjustment: it shows at once whether there is
    seasonality, whether it changes, and whether it broke at some point.

    [RU] Подряды по месяцам: все январи, все феврали и так далее, каждый линией по годам.

    Самый информативный один график до всякой корректировки: сразу видно, есть ли сезонность,
    меняется ли она и не сломалась ли в какой-то момент.
    """
    pal_ = econ_style("dark" if dark else "light")
    s = pd.Series(series).sort_index()
    idx = pd.DatetimeIndex(s.index)
    y = np.asarray(s.values, dtype=float)
    # Detrending by SI ratios (the series minus a centred moving average) and, for a fast-growing
    # series, on the logarithm. A straight line does not follow exponential growth: on a money
    # aggregate that turned the chart into the curvature of the trend, where all twelve lines run
    # as one thick band and no seasonality is visible at all.
    # Детрендирование по SI-отношениям (ряд минус центрированное скользящее среднее) и, для быстро
    # растущего ряда, по логарифму. Прямая за экспоненциальным ростом не поспевает: на денежном
    # агрегате это превращало график в кривизну тренда, где все двенадцать линий идут одной лентой
    # и сезонности не видно вовсе.
    used = transform or ("log" if (np.nanmin(y) > 0 and choose_transform(y, period) == "log") else "none")
    z = np.log(y) if used == "log" else y
    si = pd.Series(_detrend_si(z, period), index=idx).dropna()
    pct = used == "log"
    vals = (np.exp(si.values) - 1) * 100 if pct else si.values
    df = pd.DataFrame({"v": vals, "m": si.index.month, "y": si.index.year})
    fig, ax = plt.subplots(figsize=(12, 4.6))
    cols = cycle_colors(period)
    for k, m in enumerate(sorted(df["m"].unique())):
        g = df[df["m"] == m]
        ax.plot(g["y"], g["v"], color=cols[k % len(cols)], linewidth=1.3)
        ax.annotate(str(m), (g["y"].iloc[-1], g["v"].iloc[-1]), fontsize=7,
                    color=cols[k % len(cols)], xytext=(3, 0), textcoords="offset points")
    ax.axhline(0, color=pal_["muted"], linewidth=0.7)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel(tr("год", "year"), fontsize=8, color=pal_["muted"])
    ax.set_ylabel(tr("отклонение месяца от тренда, %", "the month's deviation from trend, %") if pct
                  else tr("отклонение месяца от тренда", "the month's deviation from trend"),
                  fontsize=8, color=pal_["muted"])
    chart_frame(ax, title or tr("Подряды по месяцам: каждая линия -- один календарный месяц по годам",
                                "Month sub-series: each line is one calendar month across years"),
                subtitle or tr(f"по горизонтали -- годы; линий ровно {period} -- по одной на календарный "
                               f"месяц, цифра справа = номер месяца",
                               f"horizontal -- years; exactly {period} lines, one per calendar month, "
                               f"the number on the right is the month"),
                pal=pal_)
    plt.tight_layout()
    return fig, ax


def revision_metric(series, passport, working_days_calendar=None, spans=(12, 24, 36), last=6):
    """Size of revisions: how much the estimate of the last months changes when new data arrive.

    The series is truncated by `spans` months, the specification is applied to the truncated sample,
    and the SA values of the same dates are compared with the estimate on the full sample. For a
    publication this is the most important criterion: a figure that will be different in two months
    is worse than a slightly noisier but stable one.

    [RU] Величина пересмотров: насколько меняется оценка последних месяцев, когда приходят новые
    данные.

    Ряд обрезается на `spans` месяцев, спецификация применяется к обрезанной выборке, и SA-значения
    тех же дат сравниваются с оценкой по полной выборке. Для публикации это главный критерий:
    цифра, которая через два месяца станет другой, хуже чуть более шумной, но стабильной.
    """
    s = pd.Series(series).sort_index()
    full = apply_passport(s, passport, working_days_calendar=working_days_calendar, saar_n_sim=0)
    sa_full = pd.Series(full["decomposition"]["sa"], index=pd.DatetimeIndex(full["dates"]))
    rows = []
    for k in spans:
        if len(s) - k < 4 * 12:
            continue
        cut = s.iloc[:-k]
        try:
            rt = apply_passport(cut, passport, working_days_calendar=working_days_calendar,
                                saar_n_sim=0, update_policy="frozen")
        except Exception as ex:
            rows.append({"span": k, "mean_abs_rev_pct": np.nan, "max_abs_rev_pct": np.nan,
                         "note": f"{type(ex).__name__}"})
            continue
        sa_rt = pd.Series(rt["decomposition"]["sa"], index=pd.DatetimeIndex(rt["dates"]))
        tail = sa_rt.index[-last:]
        rev = 100 * (sa_rt.loc[tail] - sa_full.loc[tail]) / sa_full.loc[tail].abs()
        rows.append({"span": k, "mean_abs_rev_pct": float(np.mean(np.abs(rev))),
                     "max_abs_rev_pct": float(np.max(np.abs(rev))), "note": ""})
    return pd.DataFrame(rows)


def compare_methods(series, passport, engines=None, working_days_calendar=None,
                    spans=(12, 24), period=12):
    """Comparison of methods WITHOUT a known truth -- the main difference from the lecture.

    On real data the true seasonality does not exist, so RMSE against it cannot be computed. The
    table therefore has two tiers:

      tier 1, "fit for use": no residual seasonality (F-test), the share of seasonal power after
              adjustment is low, no residual autocorrelation at the seasonal lag;
      tier 2, ranking among those fit: the size of revisions (most important for publication),
              the noisiness of the SA series month on month, the backtest error.

    engines=None (the default) takes x11, ucm, stl and adds x13_real when the binary is available.
    Mind the time: for each engine the revision metric re-runs the whole specification on truncated
    samples, so x13_real means several calls to the binary rather than one.

    Returns the table and the recommended engine; the analyst may override the choice, and then
    the reason goes into the passport.

    [RU] Сравнение методов БЕЗ известной истины -- главное отличие от лекции.

    На реальных данных истинной сезонности не существует, поэтому RMSE относительно неё посчитать
    нельзя. Поэтому таблица двухъярусная:

      ярус 1, «годен»: нет остаточной сезонности (F-тест), доля сезонной мощности после
              корректировки мала, нет остаточной автокорреляции на сезонном лаге;
      ярус 2, ранжирование среди годных: величина пересмотров (важнее всего для публикации),
              шумность SA-ряда м/м, ошибка бэктеста.

    engines=None (по умолчанию) берёт x11, ucm, stl и добавляет x13_real, если бинарник доступен.
    Учитывайте время: для каждого движка метрика пересмотров заново прогоняет всю спецификацию на
    обрезанных выборках, поэтому x13_real -- это несколько запусков бинарника, а не один.

    Возвращает таблицу и рекомендованный движок; аналитик может переопределить выбор, и тогда
    причина уходит в паспорт.
    """
    s = pd.Series(series).sort_index()
    if engines is None:
        # The official engine belongs in the comparison whenever it is available: for a
        # publication it is the one to aim at. Without the binary it is dropped, so that the
        # comparison does not fall over on a machine where X-13 is not installed.
        # Официальный движок должен участвовать в сравнении всегда, когда он доступен: именно на
        # него нужно ориентироваться для публикации. Без бинарника он выпадает, чтобы сравнение не
        # падало на машине, где X-13 не установлен.
        engines = ("x11", "ucm", "stl") + (("x13_real",) if x13_available() else ())
    rows, results = [], {}
    for eng in engines:
        p = copy.deepcopy(migrate_passport(passport))
        p["spec"]["decompose"]["engine"] = eng
        if not ENGINE_REGISTRY.get(eng, {}).get("pipeline"):
            rows.append({"engine": eng, "ok": False, "note": tr("не подключён к пайплайну", "not wired into the pipeline")})
            continue
        try:
            r = apply_passport(s, p, working_days_calendar=working_days_calendar, saar_n_sim=0)
        except Exception as ex:
            rows.append({"engine": eng, "ok": False, "note": f"{type(ex).__name__}: {ex}"[:60]})
            continue
        results[eng] = r
        sa = pd.Series(r["decomposition"]["sa"], index=pd.DatetimeIndex(r["dates"]))
        rev = revision_metric(s, p, working_days_calendar=working_days_calendar, spans=spans)
        mom = 100 * (sa / sa.shift(1) - 1)
        bt = r.get("backtest") or {}
        # Quality is measured on DETRENDED series. On a strongly trending series (a money
        # aggregate grows ten-fold) the spectrum is dominated by the trend at frequency zero, the
        # seasonal share is small both before and after, and the criterion stops distinguishing
        # anything -- exactly what happened on the first live run.
        # Качество меряется по ДЕТРЕНДИРОВАННЫМ рядам. У сильно трендового ряда (денежный агрегат
        # растёт в десять раз) спектр забит трендом на нулевой частоте, доля сезонной мощности мала
        # и до, и после, и критерий перестаёт что-либо различать -- ровно это и вышло на первом
        # боевом прогоне.
        z0 = np.log(s.values) if r["transform"] == "log" else s.values
        z1 = np.log(sa.values) if r["transform"] == "log" else sa.values
        si0, si1 = _detrend_si(z0, period), _detrend_si(z1, period)
        ok_m = np.isfinite(si1)
        f_after = seasonal_dummy_ftest(si1[ok_m], sa.index[ok_m], period)
        sh0 = seasonal_power_share(si0[np.isfinite(si0)], period)
        sh1 = seasonal_power_share(si1[ok_m], period)
        # Tier 1 is relative, not absolute: on SI ratios the residual share depends on how noisy
        # the particular series is, so an absolute threshold does not carry over between
        # indicators. "Fit for use" = no residual seasonality by the F-test AND the seasonal power
        # at least halved.
        # Первый ярус -- относительный, а не абсолютный: по SI-отношениям остаточная доля зависит
        # от шумности конкретного ряда, поэтому абсолютный порог не переносится между
        # показателями. «Годен» = нет остаточной сезонности по F-тесту И сезонная мощность упала
        # минимум вдвое.
        ok = (f_after["p_value"] > 0.05) and (sh0 / max(sh1, 1e-12) >= 2.0)
        rows.append({"engine": eng, "ok": bool(ok),
                     "F_p_after": f_after["p_value"],
                     "power_after": sh1,
                     "power_drop": sh0 / max(sh1, 1e-12),
                     "revision_pct": float(rev["mean_abs_rev_pct"].mean()) if len(rev) else np.nan,
                     "sa_noise_pct": float(np.nanstd(mom)),
                     "backtest_rmse": bt.get("rmse", np.nan),
                     "note": ""})
    df = pd.DataFrame(rows)
    good = df[df.get("ok", False) == True].copy() if "ok" in df.columns else df.iloc[0:0]
    best = None
    if len(good):
        good["rank"] = good["revision_pct"].rank() + 0.5 * good["sa_noise_pct"].rank()
        best = good.sort_values("rank").iloc[0]["engine"]
        df = df.merge(good[["engine", "rank"]], on="engine", how="left")
    return df, best, results


# ======================================================================
#  COMPARING TWO RUNS
#  СРАВНЕНИЕ ДВУХ ПЕРЕСМОТРОВ
# ======================================================================








def x13_check(series, passport, working_days_calendar=None, workdir=None, show=20):
    """Run x13as with exactly the specification of the passport and show what it answered.

    A diagnostic tool for the case when method="x13_real" fails. It builds the same spec file as
    analyze_series(), removes the temporary files of previous runs (a stale .d11 is the main
    reason a failure of the engine looks like a length mismatch), runs the binary and returns the
    spec, the .out report and the .err file.

    Returns a dict: spec, out, err, d11_len, d11_first, d11_last, errors -- the lines of the
    report containing ERROR.

    [RU] Запускает x13as ровно со спецификацией паспорта и показывает, что он ответил.

    Диагностический инструмент для случая, когда method="x13_real" падает. Строит тот же
    spec-файл, что и analyze_series(), удаляет временные файлы прошлых прогонов (устаревший .d11 --
    главная причина, по которой сбой движка выглядит как несовпадение длин), запускает бинарник и
    возвращает spec, отчёт .out и файл .err.

    Возвращает словарь: spec, out, err, d11_len, d11_first, d11_last, errors -- строки отчёта,
    содержащие ERROR.
    """
    p = migrate_passport(passport)
    s = pd.Series(series).sort_index()
    sample = p["spec"].get("sample", {})
    if sample.get("start"):
        s = s[s.index >= pd.Timestamp(sample["start"])]
    if sample.get("end"):
        s = s[s.index <= pd.Timestamp(sample["end"])]
    y = np.asarray(convert_to_level(s.values, p["indicator"]["input_format"], dates=s.index), dtype=float)
    dates_in = pd.DatetimeIndex(s.index)
    spec_ = p["spec"]
    period = spec_["decompose"]["period"] if isinstance(spec_["decompose"]["period"], int) else 12
    horizon = spec_["output"]["forecast"].get("horizon", 12)
    arima = spec_["decompose"]["params"].get("arima")
    if arima in (None, "automdl"):
        arima_block = "automdl{ }"
    else:
        (pp, dd, qq), (PP, DD, QQ) = arima
        arima_block = f"arima{{ model=({pp} {dd} {qq})({PP} {DD} {QQ}) }}"
    outl = spec_["preprocess"]["outliers"]
    outlier_block = f"outlier{{ types=({' '.join(outl['types'])}) }}" if outl["mode"] == "auto" else ""
    start_str = f"{dates_in[0].year}.{dates_in[0].month}"
    tr_fun = {"auto": "auto", "log": "log", "none": "none"}[spec_["transform"]]
    use_td = spec_["preprocess"]["calendar"]["trading_day"]["use"] and working_days_calendar is not None
    if use_td and period == 12:
        fut = pd.date_range(dates_in[-1], periods=horizon + 1, freq="ME")[1:]
        td_ext = trading_day_regressor(dates_in.append(fut), working_days_calendar)
        reg_block = (f"regression{{\n    user=(mytd)\n    usertype=td\n    start={start_str}\n"
                     f"    data=({_wrap_data(td_ext)})\n    aictest=(easter)\n}}")
    else:
        reg_block = "regression{ aictest=(td easter) }"
    spec_text = f"""series{{
    title="x13_check"
    start={start_str}
    period={period}
    data=({_wrap_data(y)})
}}
transform{{ function={tr_fun} }}
{reg_block}
{outlier_block}
{arima_block}
x11{{ save=(d11) }}
forecast{{ maxlead={horizon} }}
"""
    workdir = workdir or X13_WORKDIR
    os.makedirs(workdir, exist_ok=True)
    for f in glob.glob(os.path.join(workdir, "x13_check_tmp.*")):
        try:
            os.remove(f)
        except OSError:
            pass
    out, err = run_x13(spec_text, workdir, "x13_check_tmp")
    path = os.path.join(workdir, "x13_check_tmp.d11")
    d11 = None
    if os.path.exists(path):
        d11 = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, names=["date", "value"])
    # Ловим именно сообщения об ошибках, а не любые строки со словом Errors: у x13as так
    # называются и колонки таблиц ("Standard Errors"), и имя файла в шапке отчёта.
    # Сообщения об ошибках могут быть и в .out, и в .err -- смотрим оба. Ловим именно их,
    # а не любые строки со словом Errors: у x13as так называются и колонки таблиц
    # ("Standard Errors"), и имя файла в шапке отчёта.
    _is_err = lambda l: bool(re.match(r"\s*(ERROR|\*\*ERROR)", l)) or "ERROR:" in l
    errors = [l.strip() for l in ((out or "") + "\n" + (err or "")).splitlines() if _is_err(l)]
    res = {"spec": spec_text, "out": out, "err": err,
           "n_series": len(y), "d11_len": (len(d11) if d11 is not None else 0),
           "d11_first": (str(d11["date"].iloc[0]) if d11 is not None and len(d11) else None),
           "d11_last": (str(d11["date"].iloc[-1]) if d11 is not None and len(d11) else None),
           "errors": errors}
    print(tr(f"наблюдений подано: {res['n_series']} | строк в d11: {res['d11_len']}"
             + (f" ({res['d11_first']} .. {res['d11_last']})" if res["d11_len"] else ""),
             f"observations sent: {res['n_series']} | rows in d11: {res['d11_len']}"
             + (f" ({res['d11_first']} .. {res['d11_last']})" if res["d11_len"] else "")))
    if errors:
        print(tr("\nОШИБКИ x13as:", "\nx13as ERRORS:"))
        for l in errors[:show]:
            print("  ", l)
    if err and err.strip():
        print(tr("\nфайл .err:", "\n.err file:"))
        print("  " + "\n  ".join(err.strip().splitlines()[:show]))
    if not errors and res["d11_len"] and res["d11_len"] != res["n_series"]:
        print(tr("\nd11 длиннее ряда: вероятно, в таблицу дописан прогноз -- сравните даты выше.",
                 "\nd11 is longer than the series: the forecast was probably appended -- compare the dates above."))
    if not errors and not res["d11_len"]:
        print(tr("\nd11 не создан, но и явных ошибок в отчёте нет -- смотрите res['out'] целиком.",
                 "\nd11 was not created, yet there are no explicit errors in the report -- inspect res['out'] in full."))
    return res



# ======================================================================
#  PRODUCTION PATH THROUGH THE REAL x13as
#  ПРОИЗВОДСТВЕННЫЙ ПУТЬ ЧЕРЕЗ НАСТОЯЩИЙ x13as
# ======================================================================
def _wrap_data_safe(vals, width=120, indent=8, sig=10):
    """Write data into a spec file, watching the LINE WIDTH rather than the number of values.

    The spec file format has a hard limit of 133 characters per line. The version 1.0 function
    splits by a fixed count of 10 values per line, which is safe for two-digit numbers and breaks
    on six-digit ones: a money aggregate of 119088.70000 gives a line of 138 characters, x13as
    answers "Input record longer than limit: 133" and cuts the number in half. Hence the width is
    what is controlled here, and the values are written with significant digits rather than a
    fixed number of decimals.

    [RU] Пишет данные в spec-файл, следя за ДЛИНОЙ СТРОКИ, а не за числом значений.

    У формата spec-файла жёсткий лимит 133 символа на строку. Функция версии 1.0 делит по
    фиксированным 10 значениям в строке: это безопасно для двузначных чисел и ломается на
    шестизначных -- денежный агрегат 119088.70000 даёт строку в 138 символов, x13as отвечает
    "Input record longer than limit: 133" и режет число пополам. Поэтому здесь контролируется
    именно ширина, а значения пишутся значащими цифрами, а не фиксированным числом знаков после
    запятой.
    """
    out, line, pad = [], " " * indent, " " * indent
    for v in np.asarray(vals, dtype=float):
        t = f"{v:.{sig}g}"
        if len(line) + len(t) + 1 > width and line.strip():
            out.append(line.rstrip())
            line = pad
        line += t + " "
    if line.strip():
        out.append(line.rstrip())
    return "\n".join(out)[indent:] if out else ""


# The version 1.0 code is left byte-identical, but the name is rebound to the safe version: the
# old one produces an unusable spec file on any series with large values, and analyze_series looks
# the function up as a module global at call time.
# Код версии 1.0 оставлен дословно, но имя переопределено на безопасную версию: старая делает
# непригодный spec-файл на любом ряде с крупными значениями, а analyze_series ищет функцию как
# глобальное имя модуля в момент вызова.
_wrap_data_v10 = _wrap_data
_wrap_data = _wrap_data_safe


def _read_x13_tab(path, cols):
    """Read a table saved by x13as: a tab-separated file with a YYYYMM date and values in
    Fortran exponential notation.

    [RU] Читает таблицу, сохранённую x13as: файл с табуляцией, датой вида ГГГГММ и значениями в
    фортрановской экспоненциальной записи.
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep="\t", skiprows=2, header=None)
    df = df.iloc[:, :len(cols) + 1]
    df.columns = ["date"] + list(cols)
    df["date"] = pd.to_datetime(df["date"].astype(int).astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    for c in cols:
        df[c] = df[c].astype(str).str.replace("D", "E", regex=False).astype(float)
    return df.set_index("date")


def _x13_outlier_name(date, kind, period=12):
    """Name of an outlier in the X-13 language: LS2008.10, AO2022.3, TC2020.4.

    [RU] Имя выброса на языке X-13: LS2008.10, AO2022.3, TC2020.4.
    """
    d = pd.Timestamp(date)
    return f"{kind}{d.year}.{d.month if period == 12 else ((d.month - 1) // 3 + 1)}"


def x13_spec_from_passport(y, dates, passport, working_days_calendar=None, name="sa_run"):
    """Build a spec file from a passport. Three differences from the version 1.0 branch, all of
    which showed up on live data:

    1. The data are wrapped by line width (see _wrap_data_safe).
    2. Outliers set BY HAND are passed to the engine as regression variables. In version 1.0 they
       were silently dropped for x13_real: the block was written only for an automatic search, so
       the analyst's decisions did not reach the engine at all.
    3. The seasonal factors are extended over the forecast horizon (appendfcst), which is what
       gives a forecast of the SEASONALLY ADJUSTED series and not only of the original one.

    [RU] Собирает spec-файл из паспорта. Три отличия от ветки версии 1.0, и все три вылезли на
    живых данных:

    1. Данные переносятся по ширине строки (см. _wrap_data_safe).
    2. Выбросы, заданные ВРУЧНУЮ, передаются движку как переменные регрессии. В версии 1.0 они для
       x13_real молча терялись: блок писался только для автоматического поиска, и решения аналитика
       до движка не доходили.
    3. Сезонные факторы продлеваются на горизонт прогноза (appendfcst) -- именно это даёт прогноз
       СЕЗОННО СКОРРЕКТИРОВАННОГО ряда, а не только исходного.
    """
    p = migrate_passport(passport)
    spec_ = p["spec"]
    period = spec_["decompose"]["period"] if isinstance(spec_["decompose"]["period"], int) else 12
    horizon = spec_["output"]["forecast"].get("horizon", 12)
    dates = pd.DatetimeIndex(dates)
    _per = lambda d: (d.month if period == 12 else ((d.month - 1) // 3 + 1))
    start_str = f"{dates[0].year}.{_per(dates[0])}"
    ms = spec_["sample"].get("model_span", {})
    span_line = ""
    if ms.get("start") or ms.get("end"):
        a = pd.Timestamp(ms["start"]) if ms.get("start") else dates[0]
        b = pd.Timestamp(ms["end"]) if ms.get("end") else dates[-1]
        span_line = f"    modelspan = ({a.year}.{_per(a)}, {b.year}.{_per(b)})\n"

    tr_fun = {"auto": "auto", "log": "log", "none": "none"}[spec_["transform"]]
    outl = spec_["preprocess"]["outliers"]
    cal = spec_["preprocess"]["calendar"]

    reg_vars, reg_lines = [], []
    if outl["mode"] == "manual":
        reg_vars = [_x13_outlier_name(o["date"], o["type"], period) for o in outl["list"]]
    aictest = []
    if cal["easter"]["use"]:
        aictest.append("easter")
    td_kind = cal["trading_day"].get("kind", "flow")
    if cal["trading_day"]["use"] and td_kind == "stock":
        # A stock is measured on one date: what matters is the day of the week it fell on,
        # not the number of working days in the month.
        # Запас измеряется на одну дату: важен день недели, на который она пришлась, а не
        # число рабочих дней в месяце.
        reg_vars.append(f"tdstock[{cal['trading_day'].get('stock_day', 31)}]")
    use_td = (cal["trading_day"]["use"] and td_kind == "flow" and working_days_calendar is not None)
    if cal["trading_day"]["use"] and td_kind == "flow" and not use_td:
        aictest.append("td")
    if use_td and period == 12 and td_kind == "flow":
        fut = pd.date_range(dates[-1], periods=horizon + 1, freq="ME")[1:]
        td_ext = trading_day_regressor(dates.append(fut), working_days_calendar)
        reg_lines += ["    user=(mytd)", "    usertype=td", f"    start={start_str}",
                      f"    data=({_wrap_data_safe(td_ext)})"]
    if reg_vars:
        reg_lines.insert(0, f"    variables=({' '.join(reg_vars)})")
    if aictest:
        reg_lines.append(f"    aictest=({' '.join(aictest)})")
    reg_block = "regression{\n" + "\n".join(reg_lines) + "\n}" if reg_lines else ""

    arima = spec_["decompose"]["params"].get("arima")
    if arima in (None, "automdl"):
        arima_block = "automdl{ }"
    else:
        (pp, dd, qq), (PP, DD, QQ) = arima
        arima_block = f"arima{{ model=({pp} {dd} {qq})({PP} {DD} {QQ}) }}"
    outlier_block = (f"outlier{{ types=({' '.join(outl['types'])}) critical={outl.get('cv', 3.5)} }}"
                     if outl["mode"] == "auto" else "")

    # Coverage of the forecast interval: X-13 prints exactly the band asked for. Keeping it in
    # the passport means the chart legend and the numbers can never disagree.
    # Охват прогнозного интервала: X-13 печатает ровно ту полосу, которую попросили. Хранение его
    # в паспорте означает, что подпись на графике и числа не могут разойтись.
    prob = float(spec_["output"].get("forecast_ci", 0.95))
    decomposition = spec_["decompose"]["params"].get("decomposition", "x11")
    if decomposition == "seats":
        dec_block = "seats{ save=(s10 s11 s12) }"
    else:
        dec_block = "x11{ appendfcst=yes save=(d10 d11 d12 d16) }"

    return f"""series{{
    title="{name}"
    start={start_str}
    period={period}
{span_line}    data=({_wrap_data_safe(y)})
}}
transform{{ function={tr_fun} }}
{reg_block}
{outlier_block}
{arima_block}
{dec_block}
forecast{{ maxlead={horizon} probability={prob} save=(fct) }}
"""


def x13_run_passport(series, passport, working_days_calendar=None, workdir=None, name="sa_x13",
                     saar_n_sim=2000, seed=0):
    """Run the real x13as with the specification of the passport and return a result in the same
    shape as analyze_series(), so that the notebook and the model work with it identically.

    The forecast of the seasonally adjusted series is obtained the way the engine itself does it:
    the forecast of the original series (table fct, with a confidence interval) divided by the
    seasonal factors extended over the horizon (table d10). The SAAR fan is built by simulating
    paths whose spread matches the printed interval and which accumulate along the horizon --
    stretching the bounds of a level interval through the SAAR formula would not be an interval
    for a growth rate.

    [RU] Запускает настоящий x13as со спецификацией паспорта и возвращает результат в том же виде,
    что и analyze_series(), чтобы блокнот и модель работали с ним одинаково.

    Прогноз сезонно скорректированного ряда получается так же, как это делает сам движок: прогноз
    исходного ряда (таблица fct, с доверительным интервалом), делённый на сезонные факторы,
    продлённые на горизонт (таблица d10). Веер SAAR строится симуляцией траекторий, разброс которых
    соответствует напечатанному интервалу и накапливается по горизонту: протягивание границ
    интервала уровня через формулу SAAR интервалом для темпа не является.
    """
    p = migrate_passport(passport)
    s = pd.Series(series).sort_index()
    sample = p["spec"].get("sample", {})
    if sample.get("start"):
        s = s[s.index >= pd.Timestamp(sample["start"])]
    if sample.get("end"):
        s = s[s.index <= pd.Timestamp(sample["end"])]
    dates = pd.DatetimeIndex(s.index)
    y = np.asarray(convert_to_level(s.values, p["indicator"]["input_format"], dates=dates), dtype=float)

    workdir = workdir or X13_WORKDIR
    os.makedirs(workdir, exist_ok=True)
    for f in glob.glob(os.path.join(workdir, f"{name}.*")):
        try:
            os.remove(f)
        except OSError:
            pass
    spec = x13_spec_from_passport(y, dates, p, working_days_calendar, name=name)
    out, err = run_x13(spec, workdir, name)

    seats = p["spec"]["decompose"]["params"].get("decomposition") == "seats"
    sa_t = _read_x13_tab(os.path.join(workdir, f"{name}.{'s11' if seats else 'd11'}"), ["sa"])
    sf_t = _read_x13_tab(os.path.join(workdir, f"{name}.{'s10' if seats else 'd10'}"), ["sf"])
    tr_t = _read_x13_tab(os.path.join(workdir, f"{name}.{'s12' if seats else 'd12'}"), ["trend"])
    fc_t = _read_x13_tab(os.path.join(workdir, f"{name}.fct"), ["fc", "lo", "hi"])
    if sa_t is None:
        head = [l for l in (out or "").splitlines()
                if re.match(r"\s*(ERROR|\*\*ERROR)", l) or "ERROR:" in l][:6]
        raise RuntimeError(tr(
            "x13as не вернул сезонно скорректированный ряд. Строки отчёта с ошибками:\n  ",
            "x13as returned no seasonally adjusted series. Report lines with errors:\n  ")
            + ("\n  ".join(head) if head else tr("(ошибок в отчёте нет -- смотрите x13_out целиком)",
                                                 "(no errors in the report -- inspect x13_out in full)")))

    sa = sa_t["sa"].reindex(dates)
    trend = tr_t["trend"].reindex(dates) if tr_t is not None else None
    sf = sf_t["sf"] if sf_t is not None else None
    seasonal = (sf.reindex(dates) if sf is not None else None)

    mult = p["spec"]["transform"] == "log" or (sf is not None and 0.5 < float(np.nanmedian(sf)) < 2)
    forecast = saar_fc = None
    if fc_t is not None and sf is not None:
        fut = fc_t.index
        sff = sf.reindex(fut)
        if mult:
            fc_sa, lo_sa, hi_sa = fc_t["fc"] / sff, fc_t["lo"] / sff, fc_t["hi"] / sff
        else:
            fc_sa, lo_sa, hi_sa = fc_t["fc"] - sff, fc_t["lo"] - sff, fc_t["hi"] - sff
        forecast = {"forecast": fc_sa.values, "lo": lo_sa.values, "hi": hi_sa.values,
                    "dates": fut, "se": ((hi_sa - lo_sa) / (2 * 1.96)).values,
                    "scale": "level", "source": "x13as",
                    "ci": float(p["spec"]["output"].get("forecast_ci", 0.95)),
                    "forecast_original": fc_t["fc"].values,
                    "lo_original": fc_t["lo"].values, "hi_original": fc_t["hi"].values}
        if saar_n_sim and mult:
            rng = np.random.default_rng(seed)
            h = len(fut)
            sd = (np.log(hi_sa.values) - np.log(lo_sa.values)) / (2 * 1.96)
            w = np.cumsum(rng.standard_normal((saar_n_sim, h)), axis=1)
            w = w / np.sqrt(np.arange(1, h + 1))               # keep the printed marginal spread
            paths = fc_sa.values * np.exp(sd * w)              # сохраняем напечатанный разброс
            full = np.concatenate([np.tile(sa.values, (saar_n_sim, 1)), paths], axis=1)
            dfp = pd.DataFrame(full.T)
            mode = p["spec"]["output"].get("saar_mode") or "level_ma"
            ma3p = dfp.rolling(3).mean()
            if mode == "level_ma":
                sp = ((ma3p / ma3p.shift(3)) ** 4 - 1) * 100
            elif mode == "level_3m":
                sp = ((dfp / dfp.shift(3)) ** 4 - 1) * 100
            else:
                gp = dfp / dfp.shift(1) - 1
                sp = ((1 + gp.rolling(3).mean()) ** 12 - 1) * 100
            arr = sp.to_numpy()
            n_hist = len(sa)
            point = np.concatenate([np.full(n_hist, np.nan), np.nanmedian(arr[n_hist:], axis=1)])
            lo = np.full_like(point, np.nan); hi = np.full_like(point, np.nan)
            # The coverage of the SAAR band is a separate setting. Annualisation raises the
            # uncertainty of the level to the fourth power, so a 95% band on a growth rate is
            # honest but very wide; 68% (one sigma) is often more readable for publication --
            # provided the coverage is stated next to the chart.
            # Охват веера SAAR -- отдельная настройка. Аннуализация возводит неопределённость
            # уровня в четвёртую степень, поэтому 95% на темпе роста честны, но очень широки;
            # для публикации часто читабельнее 68% (одна сигма) -- при условии, что охват указан
            # рядом с графиком.
            ci = float(p["spec"]["output"].get("saar_ci")
                       or p["spec"]["output"].get("forecast_ci", 0.95))
            lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
            lo[n_hist:] = np.nanpercentile(arr[n_hist:], lo_q, axis=1)
            hi[n_hist:] = np.nanpercentile(arr[n_hist:], hi_q, axis=1)
            saar_fc = {"point": point, "lo": lo, "hi": hi, "ci": ci}

    mode_saar = p["spec"]["output"].get("saar_mode") or "level_ma"
    ma3, saar_hist = three_month_ma_saar(sa.values, mode=mode_saar)
    if saar_fc is not None:
        saar_fc["point"][:len(sa)] = saar_hist.values

    z = np.log(y) if p["spec"]["transform"] == "log" else y
    z_sa = np.log(sa.values) if p["spec"]["transform"] == "log" else sa.values
    quality = evaluate_seasonality(z, z_sa, dates, period=p["spec"]["decompose"]["period"],
                                   method_name="X13_REAL", show_chart=False)
    quality["interpretation"] = _interpret_quality(quality)

    rows = []
    for o in parse_outliers(out):
        try:
            d = pd.Timestamp(f"{o['date']}-01") + pd.offsets.MonthEnd(0)
            i = int(np.argmin(np.abs(dates - d)))
            rows.append((i, o["type"], o["t"]))
        except Exception:
            pass
    if p["spec"]["preprocess"]["outliers"]["mode"] == "manual":
        for o in p["spec"]["preprocess"]["outliers"]["list"]:
            d = pd.Timestamp(o["date"])
            near = dates[(dates.year == d.year) & (dates.month == d.month)]
            if len(near):
                rows.append((int(dates.get_loc(near[0])), o["type"], np.nan))

    return {"input_format": p["indicator"]["input_format"], "transform": p["spec"]["transform"],
            "y_level": y, "y_clean": y, "y_ready": y, "dates": dates,
            "calendar_names": [k for k in ("trading_day", "easter")
                               if p["spec"]["preprocess"]["calendar"][k]["use"]],
            "outliers": rows, "arima_used": parse_arima_model(out),
            "decomposition": {"trend": (trend.values if trend is not None else None),
                              "seasonal": (seasonal.values if seasonal is not None else None),
                              "sa": sa.values, "irregular": None},
            "quality": quality, "ma3": ma3, "saar_hist": saar_hist,
            "forecast": forecast, "saar_forecast": saar_fc, "backtest": None,
            "method": "X13_REAL", "x13_out": out, "x13_err": err, "x13_spec": spec}


def x13_model_summary(result, show=True):
    """What model x13as actually chose and estimated: the ARIMA order, the coefficients of the
    regression (outliers and calendar) with their t-values, the ARIMA parameters with standard
    errors, and the likelihood statistics.

    Takes the result of apply_passport()/x13_run_passport() or the text of the report itself.
    Returns a dict; with show=True also prints it.

    A practical note: AICC may be compared between models only within the same order of
    differencing -- the likelihood is computed for the differenced series, so a model with a
    different (d, D) explains a different dependent variable.

    [RU] Какую модель x13as на самом деле выбрал и оценил: порядок ARIMA, коэффициенты регрессии
    (выбросы и календарь) с t-статистиками, параметры ARIMA со стандартными ошибками и статистики
    правдоподобия.

    Принимает результат apply_passport()/x13_run_passport() или сам текст отчёта. Возвращает
    словарь; при show=True ещё и печатает его.

    Практическая оговорка: AICC сравним между моделями только при одинаковом порядке
    дифференцирования -- правдоподобие считается для продифференцированного ряда, поэтому модель с
    другим (d, D) объясняет другую зависимую переменную.
    """
    out = result if isinstance(result, str) else (result or {}).get("x13_out") or ""
    if not out:
        raise ValueError(tr("В результате нет отчёта x13as (движок был не x13_real?).",
                            "The result contains no x13as report (was the engine not x13_real?)."))
    res = {"order": None, "auto_choice": None, "regression": None, "arima": None, "stats": {}}

    m = re.search(r"Final automatic model choice\s*:\s*(\([\d\s]+\)\([\d\s]+\))", out)
    res["auto_choice"] = m.group(1).strip() if m else None
    res["order"] = parse_arima_model(out)

    rows = []
    mr = re.search(r"Regression Model\s*\n\s*-{5,}(.*?)-{5,}\s*\n\s*-{5,}", out, re.S)
    if mr is None:
        mr = re.search(r"Regression Model\s*\n\s*-{5,}(.*?)\n\s*\n", out, re.S)
    if mr:
        for line in mr.group(1).splitlines():
            mm = re.match(r"\s*(\S+)\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s*$", line)
            if mm:
                rows.append({"variable": mm.group(1), "estimate": float(mm.group(2)),
                             "se": float(mm.group(3)), "t": float(mm.group(4))})
    res["regression"] = pd.DataFrame(rows, columns=["variable", "estimate", "se", "t"])

    rows, block = [], None
    ma = re.search(r"ARIMA Model:.*?\n(.*?)(?:Variance|Likelihood Statistics)", out, re.S)
    if ma:
        group = ""
        for line in ma.group(1).splitlines():
            g = re.match(r"\s*((?:Non)?[Ss]easonal (?:AR|MA|Difference))", line)
            if g:
                group = g.group(1).strip()
            mm = re.match(r"\s*Lag\s+(\d+)\s+(-?[\d.]+)\s+([\d.]+)", line)
            if mm:
                est, se = float(mm.group(2)), float(mm.group(3))
                rows.append({"block": group, "lag": int(mm.group(1)), "estimate": est,
                             "se": se, "t": est / se if se else np.nan})
    res["arima"] = pd.DataFrame(rows, columns=["block", "lag", "estimate", "se", "t"])

    ml = out[out.rfind("Likelihood Statistics"):]
    for key, pat in [("nobs", r"Number of observations \(nobs\)\s+(\d+)"),
                     ("nefobs", r"Effective number of observations \(nefobs\)\s+(\d+)"),
                     ("np", r"Number of parameters estimated \(np\)\s+(\d+)"),
                     ("loglik", r"Log likelihood\s+(-?[\d.]+)"),
                     ("aic", r"AIC\s+(-?[\d.]+)"),
                     ("aicc", r"AICC[^\d\-]*(-?[\d.]+)"),
                     ("bic", r"BIC\s+(-?[\d.]+)")]:
        mm = re.search(pat, ml)
        if mm:
            res["stats"][key] = float(mm.group(1))

    if show:
        o = res["order"]
        print(tr("Модель ARIMA: ", "ARIMA model: ")
              + (f"({o[0][0]} {o[0][1]} {o[0][2]})({o[1][0]} {o[1][1]} {o[1][2]})" if o else "?")
              + (tr("   (подобрана автоматически)", "   (chosen automatically)")
                 if res["auto_choice"] else tr("   (задана вручную)", "   (set by hand)")))
        if len(res["arima"]):
            print(tr("\nПараметры ARIMA:", "\nARIMA parameters:"))
            print(res["arima"].round(4).to_string(index=False))
        if len(res["regression"]):
            print(tr("\nРегрессия (выбросы и календарь):", "\nRegression (outliers and calendar):"))
            print(res["regression"].round(4).to_string(index=False))
        if res["stats"]:
            print(tr("\nСтатистики: ", "\nStatistics: ")
                  + "  ".join(f"{k}={v:g}" for k, v in res["stats"].items()))
    return res


# ======================================================================
#  РАБОЧАЯ ПАПКА X-13
# ======================================================================
X13_WORKDIR = "x13data"
"""Куда x13as кладёт свои рабочие файлы (.spc, .out, .err, .d10, .d11, .fct и прочие).

X-13 -- программа на Фортране, она общается через файлы на диске, и по умолчанию это была
текущая папка: рядом с блокнотами копился десяток служебных файлов на каждый прогон.
Здесь они собираются в одну папку, которая создаётся сама. Папку можно сменить
(sx.X13_WORKDIR = "...") или вернуть прежнее поведение, поставив ".".

Файлы не удаляются после прогона намеренно: именно в .out лежит отчёт движка, на который
ссылаются x13_model_summary() и диагностика при сбое.
"""

def _x13_dir(workdir=None):
    """Куда писать и откуда читать файлы x13as. Текущая папка означает X13_WORKDIR.

    Важно, что подмену делают И запись, И чтение: иначе прогон запишет файлы в одно место,
    а парная функция чтения пойдёт искать их в другое и вернёт None. Ровно так сломался
    учебный блокнот, где идут подряд run_x13(spec, ".", name) и read_d11(".", name).

    [EN] Where the x13as files are written and read. The current folder means X13_WORKDIR.
    Both writing AND reading must be redirected, otherwise a run writes to one place while
    its paired reader looks in another and returns None.
    """
    return workdir if workdir not in (None, ".", "") else X13_WORKDIR


_run_x13_v10 = run_x13
_read_d11_v10 = read_d11
_read_x13_table_v10 = _read_x13_table


def run_x13(spec_text, workdir=None, name="x13run"):
    """Запуск x13as с рабочей папкой по умолчанию из X13_WORKDIR.

    Обёртка над версией 1.0: подменяется только папка, всё остальное без изменений.
    Вызовы, где папка задана явно, работают как раньше.

    [EN] Run x13as with the default working folder taken from X13_WORKDIR.
    """
    wd = _x13_dir(workdir)
    os.makedirs(wd, exist_ok=True)
    return _run_x13_v10(spec_text, wd, name)


def read_d11(workdir=None, name="x13run"):
    """Чтение <name>.d11 из той же папки, куда писал run_x13.

    [EN] Read <name>.d11 from the same folder run_x13 wrote to.
    """
    return _read_d11_v10(_x13_dir(workdir), name)


def _read_x13_table(workdir, name, suffix):
    """Чтение произвольной таблицы x13as из той же папки, куда писал run_x13.

    [EN] Read any x13as table from the same folder run_x13 wrote to.
    """
    return _read_x13_table_v10(_x13_dir(workdir), name, suffix)


# ======================================================================
#  БЫСТРАЯ КОРРЕКТИРОВКА ДЛЯ ОТЧЁТОВ
# ======================================================================
def quick_adjust(df, value_col="Value", model="auto", period=12, forecast=12,
                 engine=None, workdir=None, warn_on_fail=True):
    """Быстрая автоматическая сезонная корректировка -- для графика в отчёте.

    Отличие от паспортного пути принципиальное, и его стоит держать в голове. Здесь ВСЁ
    решает автоматика: преобразование, календарь, выбросы, модель ARIMA. Никаких решений
    аналитика не фиксируется, ничего не сохраняется, сравнить два прогона нельзя. Это
    нормально для иллюстрации в обзоре, где показателей десятки и каждый нужен «примерно».

    Для публикуемой цифры так делать не следует: там нужен паспорт (new_passport →
    apply_passport), где выбросы обоснованы, метод выбран по сравнению, а пересмотр
    зафиксирован и воспроизводим.

    model: "auto" (X-13 решает сам), "additive" (без логарифма), "multiplicative" (логарифм).
    engine: по умолчанию настоящий x13as, если он доступен, иначе наш x11.

    Возвращает DataFrame с тем же столбцом value_col -- сезонно скорректированный ряд.
    При сбое возвращает исходный ряд и предупреждает: в обзоре с десятком графиков падение
    одного ряда не должно рушить весь рендер.

    [EN] Quick automatic seasonal adjustment for a chart in a report: everything is decided
    automatically, nothing is recorded and two runs cannot be compared. For a published
    figure use the passport path instead. On failure returns the original series with a
    warning, so that one series cannot break the whole render.
    """
    if df is None or (hasattr(df, "empty") and df.empty):
        return pd.DataFrame(columns=[value_col])
    if isinstance(df, pd.Series):
        s = df.rename(value_col)
    else:
        if value_col not in df.columns:
            return pd.DataFrame(columns=[value_col])
        s = df[value_col]
    s = pd.Series(pd.to_numeric(s, errors="coerce").values,
                  index=pd.DatetimeIndex(pd.to_datetime(s.index))).sort_index()
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 3 * period:
        if warn_on_fail:
            warnings.warn(tr(f"Слишком короткий ряд для сезонной корректировки ({len(s)} наблюдений).",
                             f"The series is too short for seasonal adjustment ({len(s)} observations)."))
        return pd.DataFrame({value_col: s.values}, index=s.index)

    transform = {"auto": "auto", "additive": "none", "multiplicative": "log"}.get(model, "auto")
    eng = engine or ("x13_real" if x13_available() else "x11")
    p = new_passport("quick", "quick", indicator_type="flow_volume", input_format="level",
                     period=period, forecast_horizon=forecast, engine=eng)
    p["spec"]["transform"] = transform
    p["spec"]["decompose"]["engine"] = eng
    p["spec"]["preprocess"]["outliers"]["mode"] = "auto"
    p["spec"]["preprocess"]["calendar"]["trading_day"]["use"] = False
    p["spec"]["preprocess"]["calendar"]["easter"]["use"] = False
    p["run"]["run_folder"] = workdir or X13_WORKDIR

    try:
        res = (x13_run_passport(s, p, workdir=workdir, name="quick_adjust", saar_n_sim=0)
               if eng == "x13_real"
               else analyze_series(s, **{**passport_to_kwargs(p, None),
                                         "auto_outliers": True, "outliers": None},
                                   plot_mode=None, saar_n_sim=0))
    except Exception as ex:
        if warn_on_fail:
            warnings.warn(tr(f"Сезонная корректировка не удалась ({type(ex).__name__}: {ex}); "
                             f"возвращён исходный ряд.",
                             f"Seasonal adjustment failed ({type(ex).__name__}: {ex}); "
                             f"the original series is returned."))
        return pd.DataFrame({value_col: s.values}, index=s.index)

    sa = np.asarray(res["decomposition"]["sa"], dtype=float)
    out = pd.DataFrame({value_col: sa}, index=pd.DatetimeIndex(res["dates"])).dropna()
    out.attrs["engine"] = res.get("method")
    out.attrs["transform"] = res.get("transform")
    out.attrs["outliers"] = res.get("outliers")
    return out


def quick_adjust_mom(df, value_col="Value"):
    """Сезонно скорректированный темп м/м: сначала корректируется УРОВЕНЬ, затем считается
    прирост.

    Порядок важен: у ряда темпов сдвиг уровня неотличим от разового всплеска, и типизация
    выбросов ломается (подробно -- в разделе VI.2 учебного блокнота).

    [EN] Seasonally adjusted m/m growth: the LEVEL is adjusted first, growth is derived
    afterwards. In a series of rates a level shift is indistinguishable from a one-off spike.
    """
    level_sa = quick_adjust(df, value_col=value_col, model="multiplicative")
    if level_sa.empty or value_col not in level_sa.columns:
        return pd.DataFrame(columns=[value_col])
    return (level_sa.pct_change(periods=1) * 100).dropna()


# Совместимость с прежними именами из db_utils: в отчётах Quarto они вызываются десятками
# раз, и менять там нужно только строку импорта.
# Compatibility aliases for the old db_utils names used across the Quarto reports.
seasonally_adjust_monthly = quick_adjust
seasonally_adjust_mom_from_level = quick_adjust_mom



# ======================================================================
#  MODEL SPAN, STOCK TRADING DAY, SEASONAL OUTLIERS, BEFORE/AFTER CHARTS
#  ОБУЧАЮЩАЯ ВЫБОРКА, КАЛЕНДАРЬ ЗАПАСОВ, СЕЗОННЫЕ ВЫБРОСЫ, ГРАФИКИ ДО/ПОСЛЕ
# ======================================================================
PASSPORT_SCHEMA_VERSION = "1.1"


def _migrate_1_0_to_1_1(p):
    """Passport 1.0 -> 1.1: optional fields are added, nothing is renamed or lost.

    New: spec.sample.model_span (estimate the model on one interval, adjust the whole series),
    spec.preprocess.calendar.trading_day.kind ("flow" or "stock") and .stock_day, and the "SO"
    type becomes allowed in the outlier list.

    [RU] Паспорт 1.0 -> 1.1: добавляются необязательные поля, ничего не переименовано и не
    потеряно.

    Новое: spec.sample.model_span (оценивать модель на одном интервале, корректировать весь ряд),
    spec.preprocess.calendar.trading_day.kind ("flow" или "stock") и .stock_day, а в списке
    выбросов разрешается тип "SO".
    """
    p = copy.deepcopy(p)
    p.setdefault("spec", {}).setdefault("sample", {})
    p["spec"]["sample"].setdefault("model_span", {"start": None, "end": None,
                                                  "origin": None, "reason": ""})
    td = p["spec"].setdefault("preprocess", {}).setdefault("calendar", {}).setdefault(
        "trading_day", {})
    td.setdefault("kind", "flow")
    td.setdefault("stock_day", 31)
    p["schema_version"] = "1.1"
    return p


_PASSPORT_MIGRATIONS["1.0"] = _migrate_1_0_to_1_1


# ----------------------------------------------------------------------
#  Trading day: a flow and a stock are different regressors
#  Торговые дни: поток и запас -- разные регрессоры
# ----------------------------------------------------------------------
def tdstock_regressors(dates, stock_day=31, center=True):
    """Trading-day regressors for a STOCK series: six day-of-week contrasts of the date on which
    the stock is measured.

    The difference from the flow regressor is substantive rather than technical. A flow (output,
    retail turnover) accumulates over the month, so what matters is HOW MANY working days there
    were. A stock (money supply, balances, reserves) is measured ON ONE DATE, and the number of
    working days in the month changes nothing: what matters is WHICH DAY OF THE WEEK that date
    fell on. If the reporting date is a Saturday, the balance carries Friday's position; if it is a
    Monday, it already carries the weekend's payments.

    stock_day is the day of the month on which the stock is reported; 31 means the end of the
    month (this is exactly the X-13 convention, tdstock[31]).

    Returns a matrix n x 6: the indicators of Monday..Saturday against Sunday, centred.

    [RU] Регрессоры торговых дней для ряда-ЗАПАСА: шесть контрастов дня недели той даты, на которую
    фиксируется запас.

    Отличие от потокового регрессора содержательное, а не техническое. Поток (выпуск, оборот
    розницы) накапливается за месяц, поэтому важно, СКОЛЬКО было рабочих дней. Запас (денежная
    масса, остатки, резервы) измеряется НА ОДНУ ДАТУ, и число рабочих дней в месяце ничего не
    меняет: важно, на КАКОЙ ДЕНЬ НЕДЕЛИ эта дата пришлась. Если отчётная дата -- суббота, остаток
    несёт позицию пятницы; если понедельник -- в нём уже платежи выходных.

    stock_day -- день месяца, на который фиксируется запас; 31 означает конец месяца (это ровно
    конвенция X-13, tdstock[31]).

    Возвращает матрицу n x 6: индикаторы понедельника..субботы против воскресенья, центрированные.
    """
    idx = pd.DatetimeIndex(dates)
    days = []
    for d in idx:
        dim = pd.Period(d, freq="M").days_in_month
        day = min(stock_day, dim)
        days.append(pd.Timestamp(year=d.year, month=d.month, day=day).weekday())
    days = np.asarray(days)                       # 0 = Monday ... 6 = Sunday
    X = np.zeros((len(idx), 6))
    for k in range(6):
        X[:, k] = (days == k).astype(float) - (days == 6).astype(float)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return X


def calendar_significance_ext(y, dates, period=12, n_harm=2, working_days_calendar=None,
                              easter_tradition=None, easter_window=None, kind="flow",
                              stock_day=31):
    """Significance of the calendar regressors, with the right one chosen for the type of series.

    kind="flow"  -- one regressor: the deviation of the number of working days from the average.
    kind="stock" -- a block of six day-of-week contrasts of the reporting date, tested jointly by
                    an F-test, exactly as X-13 tests tdstock[w].

    Testing a flow regressor on a stock is the commonest way to conclude "the calendar is not
    needed" when in fact a different calendar is needed.

    [RU] Значимость календарных регрессоров с выбором правильного под тип ряда.

    kind="flow"  -- один регрессор: отклонение числа рабочих дней от среднего.
    kind="stock" -- блок из шести контрастов дня недели отчётной даты, проверяемый совместно
                    F-тестом, ровно так же, как X-13 проверяет tdstock[w].

    Проверить потоковый регрессор на запасе -- самый частый способ сделать вывод «календарь не
    нужен» там, где нужен другой календарь.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    dates = pd.DatetimeIndex(dates)
    base, _ = _harmonic_trend_cols(n, period, n_harm)
    tradition = easter_tradition or DEFAULT_EASTER_TRADITION
    window = easter_window or DEFAULT_EASTER_WINDOW

    def _fit(cols):
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        k = X.shape[1]
        s2 = float(r @ r) / n
        ll = -0.5 * n * (np.log(2 * np.pi * max(s2, 1e-300)) + 1)
        return beta, r, -2 * ll + 2 * k + 2 * k * (k + 1) / max(n - k - 1, 1), X

    _, r0, aicc0, X0 = _fit(base)
    rss0 = float(r0 @ r0)
    rows = []

    if kind == "stock":
        blk = tdstock_regressors(dates, stock_day)
        beta, r1, aicc1, X1 = _fit(base + [blk[:, k] for k in range(blk.shape[1])])
        rss1 = float(r1 @ r1)
        df1, df2 = blk.shape[1], n - X1.shape[1]
        F = ((rss0 - rss1) / df1) / (rss1 / df2) if df2 > 0 else np.nan
        rows.append({"regressor": f"tdstock[{stock_day}]", "coef": np.nan, "t_stat": np.nan,
                     "F": F, "p_value": float(stats.f.sf(F, df1, df2)) if df2 > 0 else np.nan,
                     "aicc_delta": aicc1 - aicc0,
                     "verdict": tr("включать", "include")
                     if (stats.f.sf(F, df1, df2) < 0.05 and aicc1 < aicc0)
                     else tr("не включать", "do not include")})
    else:
        col = trading_day_regressor(dates, working_days_calendar)
        beta, r1, aicc1, X1 = _fit(base + [col])
        dof = n - X1.shape[1]
        s2 = float(r1 @ r1) / max(dof, 1)
        se = np.sqrt(max(s2 * np.linalg.pinv(X1.T @ X1)[-1, -1], 1e-12))
        t = beta[-1] / se
        rows.append({"regressor": "trading_day", "coef": beta[-1], "t_stat": t, "F": np.nan,
                     "p_value": float(2 * stats.t.sf(abs(t), max(dof, 1))),
                     "aicc_delta": aicc1 - aicc0,
                     "verdict": tr("включать", "include") if (abs(t) >= 2 and aicc1 < aicc0)
                     else tr("не включать", "do not include")})

    col = easter_regressor(dates, L=window, tradition=tradition, center=True)
    beta, r1, aicc1, X1 = _fit(base + [col])
    dof = n - X1.shape[1]
    s2 = float(r1 @ r1) / max(dof, 1)
    se = np.sqrt(max(s2 * np.linalg.pinv(X1.T @ X1)[-1, -1], 1e-12))
    t = beta[-1] / se
    rows.append({"regressor": "easter", "coef": beta[-1], "t_stat": t, "F": np.nan,
                 "p_value": float(2 * stats.t.sf(abs(t), max(dof, 1))),
                 "aicc_delta": aicc1 - aicc0,
                 "verdict": tr("включать", "include") if (abs(t) >= 2 and aicc1 < aicc0)
                 else tr("не включать", "do not include")})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
#  Model span: estimate on one interval, adjust the whole series
#  Обучающая выборка: оцениваем на одном интервале, корректируем весь ряд
# ----------------------------------------------------------------------
def seasonal_profile_by_year(series, period=12, transform=None):
    """The seasonal profile year by year: the average SI ratio of each calendar month within each
    year. This is the object that tells whether the seasonality of the 2000s has anything to do
    with today's.

    [RU] Сезонный профиль по годам: среднее SI-отношение каждого календарного месяца внутри года.
    Именно по нему видно, имеет ли сезонность 2000-х отношение к сегодняшней.
    """
    s = pd.Series(series).sort_index()
    idx = pd.DatetimeIndex(s.index)
    y = np.asarray(s.values, dtype=float)
    used = transform or choose_transform(y, period)
    z = np.log(y) if (used == "log" and np.nanmin(y) > 0) else y
    si = _detrend_si(z, period)
    df = pd.DataFrame({"si": si, "year": idx.year, "m": idx.month}).dropna()
    return df.pivot_table(index="year", columns="m", values="si", aggfunc="mean")


def suggest_model_span(series, period=12, transform=None, ref_years=5, min_years=8, tol=None):
    """Suggest an interval on which to ESTIMATE the model, while the whole series is still
    adjusted.

    Why this is needed. The seasonal pattern of the 2000s may have nothing to do with today's, yet
    an estimate over the whole history lets it shape the current factors. Statistical offices solve
    this by restricting the estimation interval (modelspan in X-13) rather than by cutting the
    series: the old years remain in the output, they simply stop voting on what the seasonality
    looks like now.

    How the suggestion is computed. The average seasonal profile of the recent `ref_years` is taken
    as the reference -- twelve numbers, one per calendar month. Every year of the series is then
    compared with it, and the recommended start is the first year from which ALL subsequent years
    stay within the tolerance. The tolerance is derived from the data: it is the typical
    year-to-year scatter of the profile inside the reference window, so "different" means
    "different by more than the profile normally fluctuates".

    This is a hint, not a verdict. A structural break with a known date (a change of methodology, a
    redenomination, a change of the reporting perimeter) is a matter for expert judgement, and then
    the interval is set by hand with a reason -- see passport.spec.sample.model_span.origin.

    Returns a table by year, the recommended start and the tolerance used.

    [RU] Предлагает интервал, на котором ОЦЕНИВАТЬ модель, тогда как корректируется весь ряд.

    Зачем это нужно. Сезонный профиль 2000-х может не иметь отношения к сегодняшнему, а оценка по
    всей истории позволяет ему влиять на текущие факторы. Статслужбы решают это ограничением
    интервала оценивания (modelspan в X-13), а не обрезкой ряда: старые годы остаются в выводе, они
    просто перестают голосовать за то, как выглядит сезонность сейчас.

    Как считается подсказка. За эталон берётся средний сезонный профиль последних `ref_years` лет --
    двенадцать чисел, по одному на календарный месяц. Затем с ним сравнивается каждый год ряда, и
    рекомендуется первый год, начиная с которого ВСЕ последующие укладываются в допуск. Допуск
    берётся из самих данных: это типичный межгодовой разброс профиля внутри эталонного окна, то
    есть «отличается» означает «отличается сильнее, чем профиль колеблется обычно».

    Это подсказка, а не приговор. Структурный слом с известной датой (смена методики, деноминация,
    изменение периметра отчётности) -- предмет экспертного суждения, и тогда интервал задаётся
    руками с обоснованием, см. passport.spec.sample.model_span.origin.

    Возвращает таблицу по годам, рекомендуемое начало и использованный допуск.
    """
    prof = seasonal_profile_by_year(series, period, transform)
    prof = prof.dropna(thresh=max(period - 1, 1))
    if len(prof) < ref_years + 3:
        raise ValueError(tr("Слишком короткий ряд для подсказки по обучающей выборке.",
                            "The series is too short for a model-span suggestion."))
    ref = prof.iloc[-ref_years:]
    ref_mean = ref.mean(axis=0)
    inner = np.abs(ref.subtract(ref_mean, axis=1)).max(axis=1)
    if tol is None:
        tol = float(np.nanmedian(inner) * 2.0)

    rows = []
    for yr, row in prof.iterrows():
        d = float(np.nanmax(np.abs(row.values - ref_mean.values)))
        rows.append({"year": int(yr), "max_diff": d, "within_tol": bool(d <= tol)})
    tbl = pd.DataFrame(rows)

    idx = pd.DatetimeIndex(pd.Series(series).sort_index().index)
    last_year = int(idx[-1].year)
    best_year = None
    for k in range(len(tbl)):
        tail = tbl.iloc[k:]
        if tail["within_tol"].all() and (last_year - int(tail.iloc[0]["year"])) >= min_years:
            best_year = int(tail.iloc[0]["year"])
            break
    if best_year is None:
        best_year = int(tbl["year"].iloc[max(0, len(tbl) - min_years - 1)])
    start = idx[idx.year == best_year]
    best = str((start[0] if len(start) else idx[0]).date())
    tbl["recommended_start"] = tbl["year"] == best_year
    return tbl, best, float(tol)


def plot_model_span(series, passport, dark=False):
    """The series with the estimation interval highlighted: what the model is fitted on and what is
    only adjusted. The subtitle states where the interval came from -- expert judgement or the
    calculation.

    [RU] Ряд с выделенным интервалом оценивания: на чём подбирается модель и что только
    корректируется. В подзаголовке сказано, откуда взялся интервал -- из экспертного суждения или
    из расчёта.
    """
    pal_ = econ_style("dark" if dark else "light")
    p = migrate_passport(passport)
    ms = p["spec"]["sample"].get("model_span", {})
    s = pd.Series(series).sort_index()
    idx = pd.DatetimeIndex(s.index)
    a = pd.Timestamp(ms["start"]) if ms.get("start") else idx[0]
    b = pd.Timestamp(ms["end"]) if ms.get("end") else idx[-1]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.plot(idx, s.values, color=pal_["muted"], alpha=0.45,
            label=tr("вне обучающей выборки (только корректируется)",
                     "outside the model span (only adjusted)"))
    m = (idx >= a) & (idx <= b)
    ax.plot(idx[m], s.values[m], color=pal_["lazur"],
            label=tr("обучающая выборка (на ней оценивается модель)",
                     "model span (the model is estimated on it)"))
    ax.axvspan(a, b, color=pal_["lazur"], alpha=0.07)
    origin = ms.get("origin")
    sub = tr(f"интервал: {a.date()} .. {b.date()}", f"span: {a.date()} .. {b.date()}")
    if origin == "expert":
        sub += tr("   |   экспертное суждение: ", "   |   expert judgement: ") + (ms.get("reason") or "")
    elif origin == "computed":
        sub += tr("   |   расчётная подсказка: ", "   |   computed suggestion: ") + (ms.get("reason") or "")
    else:
        sub += tr("   |   не задан: модель оценивается по всему ряду",
                  "   |   not set: the model is estimated on the whole series")
    chart_frame(ax, tr("Обучающая выборка и корректируемый ряд",
                       "Model span and the adjusted series"), sub, pal=pal_)
    ax.legend(fontsize=8)
    plt.tight_layout()
    return fig, ax


# ----------------------------------------------------------------------
#  Before/after: ACF, periodogram, share of power
#  До/после: ACF, периодограмма, доля мощности
# ----------------------------------------------------------------------
def plot_seasonality_check(original, adjusted, dates, period=12, transform=None, dark=False,
                           title=None):
    """Four panels answering "has the seasonality gone": the ACF before and after, the periodogram
    before and after, and the share of seasonal power as a number.

    Everything is computed on SI ratios (the series minus a centred moving average), because on a
    strongly trending series the trend dominates both the autocorrelation and the spectrum, and
    both pictures stop showing the seasonality.

    What to look at. In the ACF the target is the bar at lag 12 (and 24): before adjustment it is
    high, after it should be near zero. The bar at lag 1 may stay high both before and after -- that
    is business-cycle inertia, not seasonality, and it must not be removed. In the periodogram the
    peaks at 1..6 cycles a year should disappear.

    [RU] Четыре панели, отвечающие на вопрос «ушла ли сезонность»: ACF до и после, периодограмма до
    и после и доля сезонной мощности числом.

    Всё считается по SI-отношениям (ряд минус центрированное скользящее среднее), потому что у
    сильно трендового ряда тренд доминирует и в автокорреляции, и в спектре, и обе картинки
    перестают показывать сезонность.

    На что смотреть. В ACF цель -- столбик на лаге 12 (и 24): до корректировки он высокий, после
    должен быть около нуля. Столбик на лаге 1 может остаться высоким и до, и после -- это инерция
    делового цикла, а не сезонность, и убирать её не нужно. На периодограмме пики на 1..6 циклах в
    год должны исчезнуть.
    """
    pal_ = econ_style("dark" if dark else "light")
    y0 = np.asarray(original, dtype=float)
    y1 = np.asarray(adjusted, dtype=float)
    if transform is None:
        transform = "log" if (np.nanmin(y0) > 0 and np.nanmax(y0) / max(np.nanmin(y0), 1e-9) > 10) else "none"
    z0 = np.log(y0) if transform == "log" else y0
    z1 = np.log(y1) if transform == "log" else y1
    si0, si1 = _detrend_si(z0, period), _detrend_si(z1, period)
    m0, m1 = np.isfinite(si0), np.isfinite(si1)

    nlags = 2 * period + 2
    a0, a1 = acf(si0[m0], nlags), acf(si1[m1], nlags)
    f0, p0 = periodogram(si0[m0])
    f1, p1 = periodogram(si1[m1])
    c0, c1 = f0 * period, f1 * period
    sh0 = seasonal_power_share(si0[m0], period)
    sh1 = seasonal_power_share(si1[m1], period)
    # The neutral level: the bands around the seasonal frequencies cover part of the spectrum by
    # their width alone, so even pure noise gives a non-zero share. Without this baseline "30%
    # after adjustment" looks alarming when it in fact means the seasonality is gone.
    # Нейтральный уровень: полосы вокруг сезонных частот занимают часть спектра уже своей шириной,
    # поэтому даже чистый шум даёт ненулевую долю. Без этого ориентира «30% после корректировки»
    # выглядит тревожно, хотя означает, что сезонности нет.
    rng = np.random.default_rng(0)
    base_share = float(np.mean([seasonal_power_share(rng.standard_normal(int(m1.sum())), period)
                                for _ in range(15)]))

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    for ax, a, lab in ((axes[0, 0], a0, tr("ДО корректировки", "BEFORE adjustment")),
                       (axes[0, 1], a1, tr("ПОСЛЕ корректировки", "AFTER adjustment"))):
        lags = np.arange(len(a))
        cols = [pal_["zoloto"] if (k % period == 0 and k > 0) else pal_["lazur"] for k in lags]
        ax.bar(lags, a, color=cols, width=0.75)
        ci = 1.96 / np.sqrt(max(m0.sum(), 1))
        ax.axhline(ci, color=pal_["muted"], linewidth=0.6, linestyle=":")
        ax.axhline(-ci, color=pal_["muted"], linewidth=0.6, linestyle=":")
        ax.axhline(0, color=pal_["muted"], linewidth=0.7)
        chart_frame(ax, tr(f"Автокорреляция: {lab}", f"Autocorrelation: {lab}"),
                    tr("золотом -- сезонные лаги 12 и 24", "gold -- seasonal lags 12 and 24"), pal=pal_)
    for ax, c, p, lab in ((axes[1, 0], c0, p0, tr("ДО", "BEFORE")),
                          (axes[1, 1], c1, p1, tr("ПОСЛЕ", "AFTER"))):
        ax.plot(c[1:], 10 * np.log10(np.maximum(p[1:], 1e-30)), color=pal_["lazur"], linewidth=1.1)
        for k in range(1, period // 2 + 1):
            ax.axvline(k, color=pal_["zoloto"], linewidth=0.7, alpha=0.7)
        chart_frame(ax, tr(f"Периодограмма: {lab}", f"Periodogram: {lab}"),
                    tr("частота, циклов в год; логарифмическая шкала мощности",
                       "frequency, cycles per year; logarithmic power scale"), pal=pal_)
    plt.tight_layout()
    drop = sh0 / max(sh1, 1e-12)
    excess0, excess1 = sh0 / base_share, sh1 / base_share
    print(tr(f"Доля сезонной мощности: до {sh0:.1%}, после {sh1:.1%}; нейтральный уровень (чистый "
             f"шум) {base_share:.1%}",
             f"Share of seasonal power: before {sh0:.1%}, after {sh1:.1%}; neutral level (pure "
             f"noise) {base_share:.1%}"))
    print(tr(f"Превышение над нейтральным: до x{excess0:.1f}, после x{excess1:.1f}"
             + (" -- сезонности не осталось" if excess1 < 1.2 else " -- сезонность ушла не полностью"),
             f"Excess over neutral: before x{excess0:.1f}, after x{excess1:.1f}"
             + (" -- no seasonality left" if excess1 < 1.2 else " -- seasonality not fully removed")))
    print(tr(f"ACF на лаге {period}: до {a0[period]:+.2f}, после {a1[period]:+.2f}   |   "
             f"на лаге 1 (инерция, убирать не нужно): до {a0[1]:+.2f}, после {a1[1]:+.2f}",
             f"ACF at lag {period}: before {a0[period]:+.2f}, after {a1[period]:+.2f}   |   "
             f"at lag 1 (inertia, not to be removed): before {a0[1]:+.2f}, after {a1[1]:+.2f}"))
    return {"acf_before": a0, "acf_after": a1, "power_before": sh0, "power_after": sh1,
            "power_neutral": base_share, "excess_before": excess0, "excess_after": excess1,
            "drop": drop, "fig": fig}



# ======================================================================
#  EVIDENCE OF SEASONALITY, SEASONAL OUTLIERS, OFFICIAL FORECAST
#  ДОКАЗАТЕЛЬСТВА СЕЗОННОСТИ, СЕЗОННЫЕ ВЫБРОСЫ, ОФИЦИАЛЬНЫЙ ПРОГНОЗ
# ======================================================================
def plot_seasonality_evidence(series, dates=None, period=12, transform=None, dark=False,
                              title=None, subtitle=None):
    """Two pictures BEFORE any adjustment: the autocorrelation and the periodogram. This is the
    place to see the seasonality with your own eyes rather than trust an F-test.

    What to look at. In the ACF the bar at lag `period` (and at 2*period) should stand out clearly
    above the confidence band -- that is the seasonality. A high bar at lag 1 is ordinary
    business-cycle inertia, it is NOT what is being removed. In the periodogram there should be
    sharp peaks exactly on the gold lines (1, 2, ... period/2 cycles a year).

    Everything is computed on SI ratios and, for a fast-growing series, on the logarithm: otherwise
    the trend dominates both the spectrum and the autocorrelations.

    [RU] Две картинки ДО всякой корректировки: автокорреляция и периодограмма. Это место, где
    сезонность видно глазами, а не со слов F-теста.

    На что смотреть. В ACF столбик на лаге `period` (и на 2*period) должен заметно выходить за
    доверительную полосу -- это и есть сезонность. Высокий столбик на лаге 1 -- обычная инерция
    делового цикла, её убирать НЕ нужно. На периодограмме должны быть острые пики ровно на золотых
    линиях (1, 2, ... period/2 цикла в год).

    Всё считается по SI-отношениям и, для быстро растущего ряда, по логарифму: иначе тренд забивает
    и спектр, и автокорреляции.
    """
    pal_ = econ_style("dark" if dark else "light")
    s = pd.Series(series).sort_index() if dates is None else pd.Series(np.asarray(series), index=pd.DatetimeIndex(dates))
    y = np.asarray(s.values, dtype=float)
    used = transform or ("log" if (np.nanmin(y) > 0 and choose_transform(y, period) == "log") else "none")
    z = np.log(y) if used == "log" else y
    si = _detrend_si(z, period)
    ok = np.isfinite(si)
    nlags = min(3 * period, max(len(z) // 3, period + 1))
    a = acf(si[ok], nlags)
    f, pw = periodogram(si[ok])
    cyc = f * period
    share = seasonal_power_share(si[ok], period)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    axes[0].bar(range(len(a)), a, color=pal_["lazur"], width=0.7)
    for k in range(period, len(a), period):
        axes[0].bar(k, a[k], color=pal_["korall"], width=0.7)
    ci = 1.96 / np.sqrt(max(ok.sum(), 1))
    for v in (ci, -ci):
        axes[0].axhline(v, color=pal_["muted"], linewidth=0.6, linestyle=":")
    axes[0].axhline(0, color=pal_["muted"], linewidth=0.7)
    chart_frame(axes[0], tr("Автокорреляция ДО корректировки", "Autocorrelation BEFORE adjustment"),
                tr(f"коралл -- сезонные лаги; ACF({period}) = {a[period]:+.2f}, ACF(1) = {a[1]:+.2f} (инерция)",
                   f"coral -- seasonal lags; ACF({period}) = {a[period]:+.2f}, ACF(1) = {a[1]:+.2f} (inertia)"),
                pal=pal_)
    axes[1].plot(cyc[1:], 10 * np.log10(np.maximum(pw[1:], 1e-18)), color=pal_["lazur"], linewidth=1.0)
    for k in range(1, period // 2 + 1):
        axes[1].axvline(k, color=pal_["zoloto"], linewidth=0.8, alpha=0.7)
    chart_frame(axes[1], tr("Периодограмма ДО корректировки", "Periodogram BEFORE adjustment"),
                tr(f"золото -- сезонные частоты; доля сезонной мощности {share:.1%}",
                   f"gold -- seasonal frequencies; share of seasonal power {share:.1%}"), pal=pal_)
    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout()
    return {"acf": a, "acf_seasonal": float(a[period]), "acf_lag1": float(a[1]),
            "power_share": float(share), "transform_used": used, "fig": fig}


def so_candidates(series, period=12, transform=None, cv=3.0, min_side=3):
    """Search for SEASONAL outliers -- our own, because X-13 cannot do it.

    X-13 estimates an SO if you name it, but it does not look for one: the automatic search accepts
    only AO, LS, TC. So the candidates have to be found separately.

    How it works. The series is detrended into SI ratios and split by calendar month. Within each
    month the values form a series across years; a break in its MEAN is exactly a change in the
    seasonal profile of that month. For every admissible break year a two-sample t-statistic is
    computed, and the largest one per month is kept. `min_side` is the minimum number of years on
    each side -- a break cannot be identified reliably at the very edge.

    Returns a table: date, type SO, t-statistic, the shift of the month's factor in percentage
    points, and the number of years on each side. Like any automatic hint, it is a starting point
    for the analyst rather than a verdict: a real SO usually has a story behind it (a change in the
    payment calendar, in methodology, in behaviour).

    [RU] Поиск СЕЗОННЫХ выбросов -- свой, потому что X-13 этого не умеет.

    X-13 оценит SO, если его назвать, но искать не будет: автопоиск принимает только AO, LS, TC.
    Поэтому кандидатов приходится находить отдельно.

    Как устроено. Ряд детрендируется в SI-отношения и разрезается по календарным месяцам. Внутри
    месяца значения образуют ряд по годам; слом его СРЕДНЕГО -- это и есть изменение сезонного
    профиля этого месяца. Для каждого допустимого года слома считается двухвыборочная
    t-статистика, и по каждому месяцу берётся наибольшая. `min_side` -- минимум лет с каждой
    стороны: у самого края слом надёжно не определить.

    Возвращает таблицу: дата, тип SO, t-статистика, сдвиг фактора месяца в процентных пунктах и
    число лет по сторонам. Как и любая автоподсказка, это отправная точка для аналитика, а не
    приговор: у настоящего SO обычно есть содержательная причина (смена платёжного календаря,
    методики, поведения).
    """
    s = pd.Series(series).sort_index()
    idx = pd.DatetimeIndex(s.index)
    y = np.asarray(s.values, dtype=float)
    used = transform or ("log" if (np.nanmin(y) > 0 and choose_transform(y, period) == "log") else "none")
    z = np.log(y) if used == "log" else y
    si = pd.Series(_detrend_si(z, period), index=idx).dropna()

    rows = []
    for m, g in si.groupby(si.index.month):
        v = g.values
        years = g.index.year.values
        if len(v) < 2 * min_side + 1:
            continue
        best = None
        for k in range(min_side, len(v) - min_side):
            a, b = v[:k], v[k:]
            n1, n2 = len(a), len(b)
            sp = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
            if sp <= 0:
                continue
            t = (b.mean() - a.mean()) / (sp * np.sqrt(1 / n1 + 1 / n2))
            if best is None or abs(t) > abs(best[1]):
                best = (k, t, b.mean() - a.mean(), n1, n2)
        if best is None or abs(best[1]) < cv:
            continue
        k, t, shift, n1, n2 = best
        d = g.index[k]
        rows.append({"date": str(d.date()), "type": "SO", "month": int(m), "t_stat": float(t),
                     "shift_pp": float(100 * (np.exp(shift) - 1) if used == "log" else shift),
                     "years_before": int(n1), "years_after": int(n2),
                     "origin": "auto", "decision": "", "reason": ""})
    df = pd.DataFrame(rows, columns=["date", "type", "month", "t_stat", "shift_pp",
                                     "years_before", "years_after", "origin", "decision", "reason"])
    df.attrs["transform_used"] = used
    return df.sort_values("t_stat", key=np.abs, ascending=False).reset_index(drop=True) if len(df) else df


def plot_so_candidates(series, candidates=None, passport=None, period=12, transform=None, dark=False):
    """Month sub-series with marks: where the automatic search suspects a seasonal outlier, and
    where the analyst has set one by hand in the passport.

    Each line is one calendar month across years. A seasonal outlier looks like a line that steps
    to a new level and stays there -- unlike an ordinary outlier, which is one point out of line.

    [RU] Подряды по месяцам с отметками: где сезонный выброс подозревает автопоиск и где аналитик
    поставил его руками в паспорте.

    Каждая линия -- один календарный месяц по годам. Сезонный выброс выглядит как линия, которая
    шагнула на новый уровень и там осталась, -- в отличие от обычного выброса, который выбивается
    одной точкой.
    """
    pal_ = econ_style("dark" if dark else "light")
    s = pd.Series(series).sort_index()
    idx = pd.DatetimeIndex(s.index)
    y = np.asarray(s.values, dtype=float)
    used = transform or ("log" if (np.nanmin(y) > 0 and choose_transform(y, period) == "log") else "none")
    z = np.log(y) if used == "log" else y
    si = pd.Series(_detrend_si(z, period), index=idx).dropna()
    if candidates is None:
        candidates = so_candidates(s, period=period, transform=used)
    manual = []
    if passport is not None:
        p = migrate_passport(passport)
        manual = [o for o in p["spec"]["preprocess"]["outliers"]["list"] if o.get("type") == "SO"]

    cols = cycle_colors(period)
    fig, ax = plt.subplots(figsize=(12.5, 5))
    for k, (m, g) in enumerate(si.groupby(si.index.month)):
        ax.plot(g.index.year, g.values, color=cols[k % len(cols)], linewidth=1.2, alpha=0.9)
        ax.annotate(str(m), (g.index.year[-1], g.values[-1]), fontsize=7,
                    color=cols[k % len(cols)], xytext=(3, 0), textcoords="offset points")
    seen = {}
    for _, r in (candidates.iterrows() if len(candidates) else []):
        d = pd.Timestamp(r["date"])
        ax.plot([d.year], [si.get(d, np.nan)], "o", ms=9, mfc="none",
                mec=pal_["zoloto"], mew=1.8)
        # Spread the labels apart: several candidates in one year used to overlap into a blob.
        # Разводим подписи: несколько кандидатов одного года слипались в нечитаемое пятно.
        k = seen.get(d.year, 0)
        seen[d.year] = k + 1
        ax.annotate(f"SO {d.year}-{d.month:02d}, t={r['t_stat']:+.1f}", (d.year, si.get(d, np.nan)),
                    fontsize=7, color=pal_["zoloto"],
                    xytext=(6, 8 + 11 * k), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=pal_["zoloto"], lw=0.5, alpha=0.6))
    for o in manual:
        d = pd.Timestamp(o["date"])
        ax.plot([d.year], [si.get(d, np.nan)], "s", ms=9, mfc="none", mec=pal_["korall"], mew=1.8)
        ax.annotate(tr("в паспорте", "in passport"), (d.year, si.get(d, np.nan)), fontsize=7,
                    color=pal_["korall"], xytext=(4, -10), textcoords="offset points")
    ax.axhline(0, color=pal_["muted"], linewidth=0.7)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.set_xlabel(tr("год", "year"), fontsize=8, color=pal_["muted"])
    ax.set_ylabel(tr("сезонное отклонение месяца, SI", "the month's seasonal deviation, SI"),
                  fontsize=8, color=pal_["muted"])
    chart_frame(ax, tr("Сезонный профиль по месяцам и кандидаты в сезонные выбросы (SO)",
                       "Seasonal profile by month and candidate seasonal outliers (SO)"),
                tr("по горизонтали -- годы; каждая линия = один календарный месяц, её высота = "
                   "насколько этот месяц выше или ниже тренда. Золото -- нашёл автопоиск, коралл -- в паспорте",
                   "horizontal -- years; each line = one calendar month, its height = how far above or below "
                   "the trend that month sits. Gold -- found automatically, coral -- set in the passport"),
                pal=pal_)
    plt.tight_layout()
    return fig, candidates


def load_forecast_band(indicator, tag, data_dir=".", suffix_lo="L", suffix_hi="H"):
    """Load the official forecast band: two tables named like the indicator plus a suffix --
    <indicator>_L_<tag> for the lower bound and <indicator>_H_<tag> for the upper one.

    For example, for RU_M2_M_CB with tag "CBRBASE_202604" the tables RU_M2_M_CB_L_CBRBASE_202604
    and RU_M2_M_CB_H_CBRBASE_202604 are read. Each holds Date and Value, one value per year end.

    Returns a DataFrame indexed by date with columns lo and hi. If a table is missing, returns None
    -- the absence of an official forecast must not break the run.

    [RU] Загружает официальный прогнозный коридор: две таблицы, названные как показатель плюс
    суффикс, -- <показатель>_L_<tag> для нижней границы и <показатель>_H_<tag> для верхней.

    Например, для RU_M2_M_CB с тегом "CBRBASE_202604" читаются таблицы RU_M2_M_CB_L_CBRBASE_202604
    и RU_M2_M_CB_H_CBRBASE_202604. В каждой -- Date и Value, по одному значению на конец года.

    Возвращает DataFrame с датой в индексе и колонками lo и hi. Если таблицы нет, возвращает None:
    отсутствие официального прогноза не должно ломать прогон.
    """
    out = {}
    for key, suf in (("lo", suffix_lo), ("hi", suffix_hi)):
        name = f"{indicator}_{suf}_{tag}"
        try:
            out[key] = load_indicator(name,
                                      path=os.path.join(data_dir, f"{name}.xlsx"))
        except Exception:
            try:
                out[key] = load_indicator(name,
                                          path=os.path.join(data_dir, f"{name}.csv"))
            except Exception:
                return None
    df = pd.DataFrame(out).sort_index()
    df.attrs["tag"] = tag
    df.attrs["indicator"] = indicator
    return df




def load_forecast(ticker, tag, kind="band", data_dir=".",
                  suffix_lo="L", suffix_hi="H"):
    """Официальный прогноз: коридор из двух границ или одна точка.

    Прогнозы публикуются по-разному, и это не мелочь оформления. Центральный банк даёт
    диапазон («рост M2 на 10-14%»), потому что честно показывает неопределённость. Другие
    источники дают одно число. Рисовать их нужно по-разному, поэтому вид указывается явно:

      kind="band"  -> читаются <тикер>_<L>_<тег> и <тикер>_<H>_<тег>, возвращаются lo и hi;
      kind="point" -> читается <тикер>_<тег>, возвращается value.

    Тикер прогноза может не совпадать с тикером показателя: например, корректируется уровень
    денежной массы, а прогнозируется её годовой прирост.

    Возвращает DataFrame с датой в индексе; в attrs["kind"] -- вид прогноза. Если таблиц
    нет, возвращает None: отсутствие прогноза не должно ломать страницу.

    [EN] An official forecast: a band of two bounds or a single point. Central banks publish
    a range, other sources a single number, and the two are drawn differently, so the kind is
    stated explicitly. The forecast ticker need not match the indicator's.
    """
    def _read(name):
        try:
            return load_indicator(name,
                                  path=os.path.join(data_dir, f"{name}.xlsx"))
        except Exception:
            try:
                return load_indicator(name,
                                      path=os.path.join(data_dir, f"{name}.csv"))
            except Exception:
                return None

    if kind == "point":
        s = _read(f"{ticker}_{tag}")
        if s is None:
            return None
        df = pd.DataFrame({"value": s}).sort_index()
    else:
        lo, hi = _read(f"{ticker}_{suffix_lo}_{tag}"), _read(f"{ticker}_{suffix_hi}_{tag}")
        if lo is None or hi is None:
            return None
        df = pd.DataFrame({"lo": lo, "hi": hi}).sort_index()
    df.attrs["kind"] = kind
    df.attrs["tag"] = tag
    df.attrs["ticker"] = ticker
    return df


# Export everything defined in the module, including the internal single-underscore
# names (_harmonic_trend_cols, _outlier_col, _wrap_data and so on): the notebook uses
# them directly, and "from seasonal_toolbox import *" would not pick them up otherwise.
# Экспортируем всё, что определено в модуле, включая служебные имена с одним
# подчёркиванием (_harmonic_trend_cols, _outlier_col, _wrap_data и т.п.): блокнот
# пользуется ими напрямую, а "from seasonal_toolbox import *" по умолчанию их не берёт.
_NOT_EXPORTED = {"os", "re", "stat", "datetime", "sqlite3", "subprocess", "warnings",
                 "np", "pd", "mpl", "plt", "fm", "stats", "minimize",
                 "Lasso", "LinearRegression", "StandardScaler",
                 # LANG is set only through set_lang(): exporting it would let
                 # "from seasonal_toolbox import *" overwrite the notebook's own LANG.
                 # LANG задаётся только через set_lang(): при экспорте
                 # "from seasonal_toolbox import *" затирал бы LANG блокнота.
                 "LANG"}
__all__ = sorted(n for n in globals()
                 if not n.startswith("__") and n not in _NOT_EXPORTED
                 and n not in {"_NOT_EXPORTED"})
