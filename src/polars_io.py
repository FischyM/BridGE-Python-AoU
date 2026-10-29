"""Polars I/O helpers shared by the pipeline modules."""
import io
import math

import numpy as np
import polars as pl
from openpyxl import Workbook


def read_whitespace(path, comment_prefix=None, **kwargs):
    """Read a whitespace-delimited text file, like pandas' sep=r'\\s+'.

    polars only takes a single-byte separator, so each line's runs of spaces and
    tabs are collapsed to one tab first. Blank lines are skipped and, if
    comment_prefix is given, everything from it to the end of the line is dropped
    (pandas' comment=). Types are inferred from every row unless
    infer_schema_length is passed; other kwargs go to pl.read_csv.
    """
    lines = []
    with open(path) as f:
        for line in f:
            if comment_prefix is not None:
                line = line.split(comment_prefix, 1)[0]
            fields = line.split()
            if fields:
                lines.append('\t'.join(fields))
    kwargs.setdefault('infer_schema_length', None)
    return pl.read_csv(io.StringIO('\n'.join(lines)), separator='\t', **kwargs)


def list_column(series):
    """One int64 array per row of a list column, as an object array.

    The SNP index columns of pathway_indices-*.parquet (ind, ind1, ind2) are list
    columns. The arrays are views into one flat buffer.
    """
    lengths = series.list.len().to_numpy()
    out = np.empty(lengths.size, dtype=object)
    if lengths.size == 0:
        return out
    # explode turns an empty list into a null row; dropping those leaves the plain concatenation
    flat = series.explode().drop_nulls().to_numpy()
    for i, part in enumerate(np.split(flat, np.cumsum(lengths)[:-1])):
        out[i] = part
    return out


def _cell(value):
    # pandas' to_excel writes NaN/None as an empty string and +/-inf as the strings 'inf'/'-inf'
    if value is None:
        return ''
    if isinstance(value, float):
        if math.isnan(value):
            return ''
        if math.isinf(value):
            return 'inf' if value > 0 else '-inf'
    return value


def to_workbook(sheets):
    """openpyxl Workbook with one sheet per DataFrame, in the layout pandas' to_excel wrote.

    sheets maps sheet name -> polars DataFrame. Row 1 holds the column names; a
    column named '' gets no header cell, which is how pandas wrote an unnamed
    index. Null/NaN become empty strings and cells are unstyled, as with pandas.
    """
    wb = Workbook()
    wb.remove(wb.active)
    for name, df in sheets.items():
        ws = wb.create_sheet(name)
        ws.append([c if c != '' else None for c in df.columns])
        for row in df.iter_rows():
            ws.append([_cell(v) for v in row])
    return wb
