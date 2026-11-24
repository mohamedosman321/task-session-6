"""Tests for the tools. They need no API key and make no network calls."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from csv_agent.tools import DataStore, build_tools  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "sample_sales.csv"


@pytest.fixture()
def store(tmp_path):
    s = DataStore(output_dir=tmp_path / "charts")
    s.load_file(str(SAMPLE))
    return s


def test_build_tools_names():
    names = [t.metadata.name for t in build_tools(DataStore())]
    assert names == ["load_file", "describe_data", "clean_data", "run_pandas", "plot"]


def test_tool_schema_has_arguments():
    tools = {t.metadata.name: t for t in build_tools(DataStore())}
    schema = tools["plot"].metadata.get_parameters_dict()
    assert {"kind", "x", "y", "agg", "freq", "title"} <= set(schema["properties"])
    assert "kind" in schema["required"]


def test_load_csv(tmp_path):
    s = DataStore()
    msg = s.load_file(str(SAMPLE))
    assert msg.startswith("Loaded sample_sales.csv")
    assert len(s.df) == 308


def test_load_excel_with_sheets(tmp_path):
    path = tmp_path / "book.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame({"a": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
        pd.DataFrame({"b": [3, 4, 5]}).to_excel(writer, sheet_name="second", index=False)
    s = DataStore()
    s.load_file(str(path))
    assert list(s.df.columns) == ["a"]
    s.load_file(str(path), sheet="second")
    assert list(s.df.columns) == ["b"]
    assert s.load_file(str(path), sheet="nope").startswith("Error:")


def test_load_errors():
    s = DataStore()
    assert s.load_file("missing.csv").startswith("Error: File not found")
    assert s.load_file(__file__).startswith("Error: Unsupported file type")


def test_tools_need_a_file_first():
    s = DataStore()
    assert "No file is loaded" in s.describe_data()
    assert "No file is loaded" in s.run_pandas("result = 1")


def test_describe_data(store):
    text = store.describe_data()
    assert "Shape: 308 rows x 10 columns" in text
    assert "Duplicate rows: 8" in text
    assert "customer_age" in text and "missing=15" in text


def test_clean_drop_duplicates(store):
    msg = store.clean_data("drop_duplicates")
    assert "Removed 8 duplicate rows" in msg
    assert len(store.df) == 300


def test_clean_fill_and_strip(store):
    store.clean_data("fill_null_median", "customer_age")
    assert store.df["customer_age"].isna().sum() == 0
    store.clean_data("fill_null_value", "discount", "0")
    assert store.df["discount"].isna().sum() == 0
    store.clean_data("strip_whitespace")
    assert not store.df["region"].str.endswith(" ").any()
    assert set(store.df["region"]) == {"Cairo", "Alexandria", "Giza", "Aswan"}


def test_clean_convert_types(store):
    store.clean_data("to_datetime", "order_date")
    assert pd.api.types.is_datetime64_any_dtype(store.df["order_date"])
    msg = store.clean_data("to_numeric", "product")
    assert "could not be converted" in msg


def test_clean_errors(store):
    assert store.clean_data("explode").startswith("Error: Unknown action")
    assert store.clean_data("fill_null_mean").startswith("Error: The action")
    assert store.clean_data("fill_null_mean", "nope").startswith("Error: Column 'nope' not found")
    assert store.clean_data("fill_null_mean", "region").startswith("Error: 'region' is not numeric")


def test_run_pandas_result_and_print(store):
    out = store.run_pandas("result = df['revenue'].sum().round(2)")
    assert out == str(round(store.df["revenue"].sum(), 2))
    assert "hello" in store.run_pandas("print('hello')")


def test_run_pandas_returns_tables(store):
    out = store.run_pandas("result = df.groupby('region')['revenue'].sum().sort_values(ascending=False)")
    assert "Cairo" in out


def test_run_pandas_does_not_change_data(store):
    rows = len(store.df)
    store.run_pandas("df.drop(df.index, inplace=True)\nresult = len(df)")
    assert len(store.df) == rows


def test_run_pandas_no_output_hint(store):
    assert "result" in store.run_pandas("x = 1")


@pytest.mark.parametrize(
    "code",
    [
        "import os\nresult = os.listdir('.')",
        "from os import path",
        "result = open('/etc/passwd').read()",
        "result = eval('1+1')",
        "result = df.__class__",
        "result = pd.read_csv('data/sample_sales.csv')",
        "df.to_csv('stolen.csv')",
        "result = getattr(df, 'shape')",
    ],
)
def test_run_pandas_blocks_unsafe_code(store, code):
    assert store.run_pandas(code).startswith("Error:")


def test_run_pandas_reports_code_errors(store):
    assert store.run_pandas("result = df['nope'].sum()").startswith("Error:")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "bar", "x": "region", "y": "revenue"},
        {"kind": "bar", "x": "category"},
        {"kind": "line", "x": "order_date", "y": "revenue", "freq": "M"},
        {"kind": "hist", "x": "customer_age"},
        {"kind": "scatter", "x": "quantity", "y": "revenue"},
        {"kind": "bar", "x": "product", "y": "revenue", "agg": "mean", "title": "Average"},
    ],
)
def test_plot_creates_png(store, kwargs):
    msg = store.plot(**kwargs)
    assert msg.startswith("Chart saved to ")
    path = Path(msg.removeprefix("Chart saved to "))
    assert path.exists() and path.stat().st_size > 1000


def test_plot_errors(store):
    assert store.plot("pie", "region").startswith("Error: Unknown chart type")
    assert store.plot("scatter", "quantity").startswith("Error: A scatter plot needs")
    assert store.plot("bar", "nope").startswith("Error: Column 'nope' not found")
    assert store.plot("bar", "region", "revenue", agg="mode").startswith("Error: Unknown agg")
    assert store.plot("line", "order_date", "revenue", freq="X").startswith("Error: Unknown freq")
    assert store.plot("hist", "region").startswith("Error: Column 'region' has no numeric")


def test_tool_call_through_llamaindex(store):
    tools = {t.metadata.name: t for t in build_tools(store)}
    out = tools["run_pandas"].call(code="result = len(df)")
    assert out.content == "308"
