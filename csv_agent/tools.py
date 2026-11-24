"""Tools for the CSV/Excel analysis agent (step 3 of the build).

All the logic lives in the DataStore class so it can be tested without an LLM.
build_tools() wraps each method as a LlamaIndex FunctionTool. The docstrings of
those wrappers are what the LLM reads, so they are written for the model.

Note: this file must not use `from __future__ import annotations`. LlamaIndex
builds each tool's schema from the real type hints.
"""

import ast
import builtins
import contextlib
import io
import re
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
from llama_index.core.tools import FunctionTool
from matplotlib.figure import Figure

SUPPORTED_FILES = {".csv", ".tsv", ".xlsx", ".xlsm"}

CLEAN_ACTIONS = (
    "drop_duplicates",
    "drop_null_rows",
    "fill_null_mean",
    "fill_null_median",
    "fill_null_mode",
    "fill_null_value",
    "strip_whitespace",
    "to_numeric",
    "to_datetime",
)

PLOT_KINDS = ("bar", "line", "hist", "scatter")
AGGREGATIONS = ("sum", "mean", "median", "count", "min", "max")
TIME_GRAINS = ("D", "W", "M", "Q", "Y")

MAX_OUTPUT_CHARS = 4000
MAX_BARS = 20

# --- Guardrails for run_pandas ------------------------------------------------
# These checks stop common mistakes and obvious misuse. They are NOT a real
# sandbox. Do not expose this agent to people you do not trust.

SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min", "print",
        "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
        "zip",
    )
}

BLOCKED_NAMES = {
    "open", "exec", "eval", "compile", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help",
    "dir", "type",
}

BLOCKED_ATTRS = {
    "eval", "to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql",
    "to_hdf", "to_feather", "to_stata", "to_clipboard", "to_orc", "to_xml",
    "to_json", "to_html", "to_latex",
}


def _check_code(code):
    """Parse the code and reject imports, file access and private attributes."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError(
                "Imports are not allowed. pd (pandas), np (numpy) and df are already available."
            )
        if isinstance(node, ast.Name):
            if node.id in BLOCKED_NAMES or node.id.startswith("__"):
                raise ValueError(f"'{node.id}' is not allowed.")
        if isinstance(node, ast.Attribute):
            attr = node.attr
            if attr.startswith("_") or attr.startswith("read_") or attr in BLOCKED_ATTRS:
                raise ValueError(f"The attribute '.{attr}' is not allowed.")
    return tree


def _safe(method):
    """Return errors as text so the agent can read them and fix its call."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            return f"Error: {exc}"

    return wrapper


def _is_text(series):
    return pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)


def _truncate(text):
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + "\n... (output cut off, ask for a smaller result)"


class DataStore:
    """Holds the loaded DataFrame and implements every tool."""

    def __init__(self, output_dir="outputs"):
        self.df = None
        self.source = None
        self.output_dir = Path(output_dir)

    # -- helpers ---------------------------------------------------------------

    def _require(self):
        if self.df is None:
            raise ValueError("No file is loaded yet. Call load_file first.")
        return self.df

    def _check_column(self, column):
        df = self._require()
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found. Available columns: {list(df.columns)}")
        return column

    # -- tool 1: load_file -----------------------------------------------------

    @_safe
    def load_file(self, path, sheet=""):
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"File not found: {path}")
        ext = p.suffix.lower()
        if ext not in SUPPORTED_FILES:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_FILES)}"
            )

        sheet_note = ""
        if ext in {".csv", ".tsv"}:
            sep = "\t" if ext == ".tsv" else ","
            try:
                df = pd.read_csv(p, sep=sep)
            except UnicodeDecodeError:
                df = pd.read_csv(p, sep=sep, encoding="latin-1")
        else:
            workbook = pd.ExcelFile(p)
            names = workbook.sheet_names
            chosen = sheet or names[0]
            if chosen not in names:
                raise ValueError(f"Sheet '{chosen}' not found. Available sheets: {names}")
            df = workbook.parse(chosen)
            sheet_note = f"Sheet: {chosen} (all sheets: {names})\n"

        self.df = df
        self.source = p.name
        return (
            f"Loaded {p.name}\n{sheet_note}"
            f"Rows: {len(df)}, Columns: {len(df.columns)}\n"
            f"Columns: {list(df.columns)}\n"
            f"First 5 rows:\n{df.head().to_string()}"
        )

    # -- tool 2: describe_data -------------------------------------------------

    @_safe
    def describe_data(self):
        df = self._require()
        rows = max(len(df), 1)
        lines = [
            f"File: {self.source}",
            f"Shape: {len(df)} rows x {len(df.columns)} columns",
            f"Duplicate rows: {int(df.duplicated().sum())}",
            "",
            "Columns:",
        ]
        for col in df.columns:
            s = df[col]
            missing = int(s.isna().sum())
            lines.append(
                f"- {col}: type={s.dtype}, missing={missing} ({missing / rows:.1%}), "
                f"unique={s.nunique()}"
            )
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            lines += ["", "Numeric summary:", numeric.describe().round(2).to_string()]
        return _truncate("\n".join(lines))

    # -- tool 3: clean_data ----------------------------------------------------

    @_safe
    def clean_data(self, action, column="", value=""):
        self._require()
        action = action.strip().lower()
        if action not in CLEAN_ACTIONS:
            raise ValueError(f"Unknown action '{action}'. Choose one of: {', '.join(CLEAN_ACTIONS)}")
        if column:
            self._check_column(column)

        needs_column = {
            "fill_null_mean", "fill_null_median", "fill_null_mode",
            "fill_null_value", "to_numeric", "to_datetime",
        }
        if action in needs_column and not column:
            raise ValueError(f"The action '{action}' needs a column name.")

        df = self.df.copy()
        before = len(df)

        if action == "drop_duplicates":
            df = df.drop_duplicates().reset_index(drop=True)
            msg = f"Removed {before - len(df)} duplicate rows."

        elif action == "drop_null_rows":
            df = (df.dropna(subset=[column]) if column else df.dropna()).reset_index(drop=True)
            where = f" in '{column}'" if column else ""
            msg = f"Dropped {before - len(df)} rows with missing values{where}."

        elif action in {"fill_null_mean", "fill_null_median"}:
            if not pd.api.types.is_numeric_dtype(df[column]):
                raise ValueError(f"'{column}' is not numeric, so mean/median cannot be used.")
            fill = df[column].mean() if action == "fill_null_mean" else df[column].median()
            count = int(df[column].isna().sum())
            df[column] = df[column].fillna(fill)
            msg = f"Filled {count} missing values in '{column}' with {fill:.4g}."

        elif action == "fill_null_mode":
            modes = df[column].mode()
            if modes.empty:
                raise ValueError(f"'{column}' has no values to take a mode from.")
            count = int(df[column].isna().sum())
            df[column] = df[column].fillna(modes.iloc[0])
            msg = f"Filled {count} missing values in '{column}' with the most common value ({modes.iloc[0]})."

        elif action == "fill_null_value":
            if value == "":
                raise ValueError("fill_null_value needs the 'value' argument.")
            fill = pd.to_numeric(value) if pd.api.types.is_numeric_dtype(df[column]) else value
            count = int(df[column].isna().sum())
            df[column] = df[column].fillna(fill)
            msg = f"Filled {count} missing values in '{column}' with {fill!r}."

        elif action == "strip_whitespace":
            targets = [column] if column else [c for c in df.columns if _is_text(df[c])]
            for c in targets:
                if not _is_text(df[c]):
                    raise ValueError(f"'{c}' is not a text column.")
                df[c] = df[c].str.strip()
            msg = f"Removed extra spaces in {len(targets)} text column(s)."

        elif action == "to_numeric":
            nulls_before = int(df[column].isna().sum())
            df[column] = pd.to_numeric(df[column], errors="coerce")
            new_nulls = int(df[column].isna().sum()) - nulls_before
            msg = f"Converted '{column}' to numbers. {new_nulls} values could not be converted and are now missing."

        else:  # to_datetime
            nulls_before = int(df[column].isna().sum())
            df[column] = pd.to_datetime(df[column], errors="coerce")
            new_nulls = int(df[column].isna().sum()) - nulls_before
            msg = f"Converted '{column}' to dates. {new_nulls} values could not be converted and are now missing."

        self.df = df
        return f"{msg} The data now has {len(df)} rows."

    # -- tool 4: run_pandas ----------------------------------------------------

    @_safe
    def run_pandas(self, code):
        df = self._require()
        tree = _check_code(code)

        namespace = {
            "df": df.copy(),  # work on a copy, the loaded data is never changed here
            "pd": pd,
            "np": np,
            "__builtins__": SAFE_BUILTINS,
        }
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(tree, "<agent>", "exec"), namespace)

        parts = []
        if printed.getvalue().strip():
            parts.append(printed.getvalue().strip())
        if "result" in namespace:
            result = namespace["result"]
            if isinstance(result, (pd.DataFrame, pd.Series)):
                parts.append(result.to_string(max_rows=40))
            else:
                parts.append(str(result))
        if not parts:
            return "The code ran but produced no output. Put the answer in a variable called result, or use print()."
        return _truncate("\n".join(parts))

    # -- tool 5: plot ----------------------------------------------------------

    @_safe
    def plot(self, kind, x, y="", agg="sum", freq="", title=""):
        df = self._require()
        kind = kind.strip().lower()
        if kind not in PLOT_KINDS:
            raise ValueError(f"Unknown chart type '{kind}'. Choose one of: {', '.join(PLOT_KINDS)}")
        self._check_column(x)
        if y:
            self._check_column(y)
        agg = (agg or "sum").strip().lower()
        if agg not in AGGREGATIONS:
            raise ValueError(f"Unknown agg '{agg}'. Choose one of: {', '.join(AGGREGATIONS)}")

        fig = Figure(figsize=(9, 5.5))
        ax = fig.subplots()

        if kind == "hist":
            data = pd.to_numeric(df[x], errors="coerce").dropna()
            if data.empty:
                raise ValueError(f"Column '{x}' has no numeric values to plot.")
            ax.hist(data, bins=20, edgecolor="white")
            ax.set_xlabel(x)
            ax.set_ylabel("Count")
            auto_title = f"Distribution of {x}"

        elif kind == "scatter":
            if not y:
                raise ValueError("A scatter plot needs both x and y.")
            points = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
            if points.empty:
                raise ValueError(f"'{x}' and '{y}' need numeric values for a scatter plot.")
            ax.scatter(points[x], points[y], alpha=0.6)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            auto_title = f"{y} vs {x}"

        else:  # bar or line
            keys = df[x]
            if freq:
                freq = freq.strip().upper()
                if freq not in TIME_GRAINS:
                    raise ValueError(f"Unknown freq '{freq}'. Choose one of: {', '.join(TIME_GRAINS)}")
                keys = pd.to_datetime(keys, errors="coerce").dt.to_period(freq).astype(str)

            if y:
                values = df[y] if agg == "count" else pd.to_numeric(df[y], errors="coerce")
                series = pd.DataFrame({"key": keys, "val": values}).groupby("key")["val"].agg(agg)
                ylabel = f"{agg} of {y}"
                auto_title = f"{agg.capitalize()} of {y} by {x}"
            else:
                series = keys.value_counts()
                ylabel = "Count"
                auto_title = f"Count by {x}"

            if kind == "line" or freq:
                series = series.sort_index()  # keep time / key order
            else:
                series = series.sort_values(ascending=False)
            if kind == "bar":
                series = series.head(MAX_BARS)
            if series.empty:
                raise ValueError("There is nothing to plot after grouping.")

            labels = [str(k) for k in series.index]
            if kind == "bar":
                ax.bar(labels, series.values)
            else:
                ax.plot(labels, series.values, marker="o")
            ax.set_xlabel(x)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45)
            for label in ax.get_xticklabels():
                label.set_ha("right")

        ax.set_title(title or auto_title)
        ax.grid(alpha=0.3)
        fig.tight_layout()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{kind}_{x}_{y}").strip("_")
        path = self.output_dir / f"{slug}_{datetime.now():%Y%m%d_%H%M%S}.png"
        fig.savefig(path, dpi=150)
        return f"Chart saved to {path}"


def build_tools(store):
    """Wrap the DataStore methods as LlamaIndex tools. The docstrings are for the LLM."""

    def load_file(path: str, sheet: str = "") -> str:
        """Load a CSV, TSV or Excel (.xlsx) file so it can be analysed.
        Call this first when the user gives a file path. For Excel files only one
        sheet is loaded: the first one, or the one named in `sheet`.
        Returns the shape, the column names and the first 5 rows."""
        return store.load_file(path, sheet)

    def describe_data() -> str:
        """Describe the loaded data: number of rows and columns, each column's type,
        missing values and unique counts, duplicate rows, and basic statistics for
        numeric columns. Call this before answering questions so you know the real
        column names."""
        return store.describe_data()

    def clean_data(action: str, column: str = "", value: str = "") -> str:
        """Clean the loaded data. This CHANGES the data, so use it only when asked or needed.
        action must be one of:
        - drop_duplicates: remove duplicate rows
        - drop_null_rows: drop rows with missing values (only in `column` if given)
        - fill_null_mean / fill_null_median / fill_null_mode: fill missing values in `column`
        - fill_null_value: fill missing values in `column` with `value`
        - strip_whitespace: remove extra spaces in text (one `column`, or all text columns)
        - to_numeric / to_datetime: convert `column` to numbers or dates
        Returns what was changed and the new row count."""
        return store.clean_data(action, column, value)

    def run_pandas(code: str) -> str:
        """Run pandas code on the loaded data to answer a question.
        The data is available as the DataFrame `df`. `pd` (pandas) and `np` (numpy)
        are available. Do not import anything and do not read or write files.
        Put the final answer in a variable called `result` (or use print()).
        The code runs on a copy, so it never changes the loaded data.
        Example: result = df.groupby('region')['revenue'].sum().sort_values(ascending=False)"""
        return store.run_pandas(code)

    def plot(kind: str, x: str, y: str = "", agg: str = "sum", freq: str = "", title: str = "") -> str:
        """Create a chart and save it as a PNG. Returns the file path.
        kind: bar, line, hist or scatter.
        x: the column for the x-axis (for hist, the column to plot).
        y: the numeric column for the y-axis. Leave empty for a bar chart of counts.
        agg: how to combine y for each x value in bar/line charts:
             sum, mean, median, count, min or max (default sum).
        freq: for date columns in bar/line charts, group by D (day), W (week),
              M (month), Q (quarter) or Y (year).
        title: optional chart title.
        Example: plot(kind='line', x='order_date', y='revenue', freq='M')"""
        return store.plot(kind, x, y, agg, freq, title)

    functions = (load_file, describe_data, clean_data, run_pandas, plot)
    return [FunctionTool.from_defaults(fn=fn) for fn in functions]
