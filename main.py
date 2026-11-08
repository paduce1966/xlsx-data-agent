import base64
import io
from typing import Annotated, Any, Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

_MAX_ROWS_RETURNED = 50
_MAX_GROUPS_RETURNED = 200
_MAX_EXAMPLES = 5

_DATASETS: dict[str, dict[str, Any]] = {}


def _session_store(ctx: agents.ToolCallbackContext) -> dict[str, Any]:
    return _DATASETS.setdefault(ctx.session.id, {})


def _sanitize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _sanitize(value.item())
        except (ValueError, AttributeError):
            return str(value)
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.where(pd.notna(df), None)
    return [{k: _sanitize(v) for k, v in row.items()} for row in clean.to_dict(orient="records")]


def _get_df(ctx: agents.ToolCallbackContext, sheet: Optional[str]) -> tuple[dict[str, Any], pd.DataFrame, str]:
    store = _session_store(ctx)
    if not store.get("sheets"):
        raise agents.ToolError(
            "Aucun fichier n'est chargé dans cette session. Demande d'abord à l'utilisateur "
            "le lien Google Drive du fichier XLSX, puis charge-le avec load_dataset."
        )
    sheets: dict[str, pd.DataFrame] = store["sheets"]
    name = sheet or store.get("active") or next(iter(sheets))
    if name not in sheets:
        raise agents.ToolError(
            f"Feuille '{name}' introuvable. Feuilles disponibles: {sorted(sheets)}."
        )
    return store, sheets[name], name


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise agents.ToolError(
            f"Colonne '{column}' introuvable. Colonnes disponibles: {list(df.columns)}."
        )


class RowFilter(BaseModel):
    column: str = Field(description="Nom exact de la colonne à filtrer.")
    operator: Literal[
        "eq", "ne", "gt", "gte", "lt", "lte",
        "contains", "not_contains", "starts_with",
        "in", "not_in", "is_null", "not_null",
    ] = Field(description="Opérateur de comparaison.")
    value: Any = Field(
        default=None,
        description="Valeur de comparaison. Liste pour 'in'/'not_in'. Inutilisée pour is_null/not_null.",
    )


class Aggregation(BaseModel):
    column: Optional[str] = Field(
        default=None,
        description="Colonne à agréger. Obligatoire sauf pour la fonction 'count' (nombre de lignes).",
    )
    function: Literal["sum", "mean", "median", "min", "max", "count", "nunique", "std"] = Field(
        description="Fonction d'agrégation."
    )


@agents.tool(model_access="direct", namespace="data")
def load_dataset(
    ctx: agents.ToolCallbackContext,
    /,
    file_id: Annotated[str, Field(description="ID Google Drive du fichier (extrait du lien fourni par l'utilisateur).")],
    file_name: Annotated[str, Field(description="Nom du fichier, ex. 'ventes.xlsx'.")],
    content_base64: Annotated[str, Field(description="Contenu du fichier XLSX encodé en base64, retourné par le téléchargement Drive.")],
    sheet: Annotated[Optional[str], Field(description="Feuille à activer. Par défaut la première feuille.")] = None,
) -> dict[str, Any]:
    """Charge un fichier XLSX téléchargé depuis Google Drive dans la session courante.

    Toutes les feuilles sont lues ; la feuille active devient `sheet` ou la première.
    Appelle ce outil une fois par fichier, puis utilise describe_columns, filter_rows
    et aggregate pour répondre. Ne rappelle load_dataset que si aucun fichier n'est
    chargé (par exemple après un redémarrage du worker).
    """
    try:
        raw = base64.b64decode(content_base64, validate=True)
    except Exception:
        raise agents.ToolError("content_base64 n'est pas du base64 valide.")
    try:
        sheets = pd.read_excel(io.BytesIO(raw), sheet_name=None, engine="openpyxl")
    except Exception as exc:
        raise agents.ToolError(f"Impossible de lire le fichier comme XLSX: {exc}")
    if not sheets:
        raise agents.ToolError("Le classeur ne contient aucune feuille.")
    store = _session_store(ctx)
    store["file_id"] = file_id
    store["file_name"] = file_name
    store["sheets"] = sheets
    active = sheet if sheet in sheets else next(iter(sheets))
    if sheet and sheet not in sheets:
        raise agents.ToolError(f"Feuille '{sheet}' introuvable. Feuilles disponibles: {sorted(sheets)}.")
    store["active"] = active
    return {
        "file_name": file_name,
        "active_sheet": active,
        "sheets": {
            name: {"rows": int(df.shape[0]), "columns": [str(c) for c in df.columns]}
            for name, df in sheets.items()
        },
    }


@agents.tool(model_access="direct", namespace="data")
def dataset_info(
    ctx: agents.ToolCallbackContext,
    /,
) -> dict[str, Any]:
    """Décrit le fichier actuellement chargé dans la session: nom, feuilles, dimensions et feuille active."""
    store = _session_store(ctx)
    if not store.get("sheets"):
        raise agents.ToolError("Aucun fichier chargé dans cette session.")
    sheets: dict[str, pd.DataFrame] = store["sheets"]
    return {
        "file_name": store.get("file_name"),
        "file_id": store.get("file_id"),
        "active_sheet": store.get("active"),
        "sheets": {
            name: {"rows": int(df.shape[0]), "columns": [str(c) for c in df.columns]}
            for name, df in sheets.items()
        },
    }


@agents.tool(model_access="direct", namespace="data")
def describe_columns(
    ctx: agents.ToolCallbackContext,
    /,
    sheet: Annotated[Optional[str], Field(description="Feuille à décrire. Par défaut la feuille active.")] = None,
    max_examples: Annotated[int, Field(description="Nombre de valeurs fréquentes renvoyées par colonne texte.")] = _MAX_EXAMPLES,
) -> list[dict[str, Any]]:
    """Décrit chaque colonne de la feuille active: type, valeurs nulles, cardinalité, min/max ou valeurs fréquentes."""
    _, df, name = _get_df(ctx, sheet)
    max_examples = max(1, min(max_examples, 20))
    out: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        entry: dict[str, Any] = {
            "sheet": name,
            "column": str(col),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            entry["min"] = _sanitize(series.min())
            entry["max"] = _sanitize(series.max())
            entry["mean"] = _sanitize(series.mean())
        elif pd.api.types.is_datetime64_any_dtype(series):
            entry["min"] = _sanitize(series.min())
            entry["max"] = _sanitize(series.max())
        else:
            top = series.value_counts(dropna=True).head(max_examples)
            entry["top_values"] = [
                {"value": _sanitize(v), "count": int(c)} for v, c in top.items()
            ]
        out.append(entry)
    return out


@agents.tool(model_access="direct", namespace="data")
def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    filters: Annotated[list[RowFilter], Field(description="Liste de filtres combinés par ET.")],
    limit: Annotated[int, Field(description="Nombre maximum de lignes retournées.")] = _MAX_ROWS_RETURNED,
    sheet: Annotated[Optional[str], Field(description="Feuille à filtrer. Par défaut la feuille active.")] = None,
) -> dict[str, Any]:
    """Filtre les lignes de la feuille active selon une liste de conditions (combinées par ET).

    Retourne le nombre total de lignes correspondantes et au plus `limit` lignes.
    """
    _, df, name = _get_df(ctx, sheet)
    limit = max(1, min(limit, _MAX_ROWS_RETURNED))
    mask = pd.Series(True, index=df.index)
    for f in filters:
        _require_column(df, f.column)
        col = df[f.column]
        op = f.operator
        if op == "is_null":
            m = col.isna()
        elif op == "not_null":
            m = col.notna()
        elif op in ("eq", "ne", "gt", "gte", "lt", "lte"):
            try:
                m = {"eq": col == f.value, "ne": col != f.value, "gt": col > f.value,
                     "gte": col >= f.value, "lt": col < f.value, "lte": col <= f.value}[op]
            except TypeError:
                numeric = pd.to_numeric(col, errors="coerce")
                m = {"eq": numeric == f.value, "ne": numeric != f.value, "gt": numeric > f.value,
                     "gte": numeric >= f.value, "lt": numeric < f.value, "lte": numeric <= f.value}[op]
        elif op == "in":
            if not isinstance(f.value, list):
                raise agents.ToolError(f"L'opérateur 'in' exige une liste de valeurs pour '{f.column}'.")
            m = col.isin(f.value)
        elif op == "not_in":
            if not isinstance(f.value, list):
                raise agents.ToolError(f"L'opérateur 'not_in' exige une liste de valeurs pour '{f.column}'.")
            m = ~col.isin(f.value)
        elif op in ("contains", "not_contains", "starts_with"):
            text = col.astype(str).str.lower()
            needle = str(f.value).lower()
            m = {"contains": text.str.contains(needle, regex=False),
                 "not_contains": ~text.str.contains(needle, regex=False),
                 "starts_with": text.str.startswith(needle)}[op]
        else:
            raise agents.ToolError(f"Opérateur inconnu: {op}")
        mask &= m.fillna(False)
    matched = df[mask]
    return {
        "sheet": name,
        "matched": int(len(matched)),
        "returned": int(min(len(matched), limit)),
        "rows": _records(matched.head(limit)),
    }


@agents.tool(model_access="direct", namespace="data")
def aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    aggregations: Annotated[list[Aggregation], Field(description="Agrégations à calculer, ex. [{'column': 'montant', 'function': 'sum'}].")],
    group_by: Annotated[Optional[list[str]], Field(description="Colonnes de regroupement. Vide ou omis pour agréger toute la feuille.")] = None,
    sheet: Annotated[Optional[str], Field(description="Feuille à agréger. Par défaut la feuille active.")] = None,
    max_groups: Annotated[int, Field(description="Nombre maximum de groupes retournés.")] = _MAX_GROUPS_RETURNED,
) -> dict[str, Any]:
    """Calcule des agrégats (sum, mean, median, min, max, count, nunique, std) par groupe.

    Retourne les groupes tels quels, au plus `max_groups` lignes, plus le nombre total de groupes.
    """
    _, df, name = _get_df(ctx, sheet)
    max_groups = max(1, min(max_groups, _MAX_GROUPS_RETURNED))
    if not aggregations:
        raise agents.ToolError("Au moins une agrégation est requise.")
    for agg in aggregations:
        if agg.column is None and agg.function != "count":
            raise agents.ToolError(
                f"La fonction '{agg.function}' exige une colonne ; seule 'count' peut s'en passer."
            )
        if agg.column is not None:
            _require_column(df, agg.column)
    group_cols = list(group_by or [])
    for col in group_cols:
        _require_column(df, col)
    named: dict[str, tuple[str, str]] = {}
    for i, agg in enumerate(aggregations):
        if agg.column is None:
            named[f"count_{i}"] = (str(df.columns[0]), "size")
        else:
            named[f"{agg.column}__{agg.function}"] = (agg.column, agg.function)
    if group_cols:
        result = df.groupby(group_cols, dropna=False).agg(**named).reset_index()
    else:
        agg_result = df.agg(**named)
        if isinstance(agg_result, pd.Series):
            agg_result = agg_result.to_frame().T
        result = agg_result.head(1).reset_index(drop=True)
    total = int(len(result))
    return {
        "sheet": name,
        "groups": total,
        "returned": int(min(total, max_groups)),
        "rows": _records(result.head(max_groups)),
    }


INSTRUCTIONS = """Tu es un agent d'analyse de données. L'utilisateur te donne un lien vers un fichier XLSX
stocké dans son Google Drive et te pose des questions en langage naturel sur son contenu.

Règles de fonctionnement:
1. Quand l'utilisateur fournit un lien Drive, extrais l'ID du fichier du lien
   (formes usuelles: https://drive.google.com/file/d/<ID>/view, .../open?id=<ID>, .../folders/<ID>).
   Si le lien ne contient pas d'ID, utilise google_drive_mcp.search_files avec le nom du fichier.
2. Vérifie le fichier avec google_drive_mcp.get_file_metadata, puis télécharge-le avec
   google_drive_mcp.download_file_content (le contenu revient en base64).
3. Charge le classeur avec l'outil data.load_dataset en passant ce contenu base64.
   Le fichier chargé est conservé pour toute la session: ne le retélécharge pas à chaque question.
4. Réponds aux questions avec les outils fixes data.dataset_info, data.describe_columns,
   data.filter_rows et data.aggregate. Tu n'as AUCUN autre moyen de calculer: n'invente jamais
   un chiffre, cite uniquement les résultats de ces outils.
5. La session conserve tout l'historique de la conversation: tu te souviens du fichier chargé
   et des questions précédentes sans que l'utilisateur se répète. Pour une question de suivi,
   réutilise directement les outils sur le fichier déjà chargé.
6. Si plusieurs feuilles existent et que la question ne précise pas laquelle, utilise la feuille
   active et dis-le; demande à l'utilisateur quelle feuille il veut si c'est ambigu.
7. Si un outil signale qu'aucun fichier n'est chargé (par exemple après un redémarrage),
   retélécharge le fichier depuis le lien déjà connu de la conversation, puis réponds.
8. Réponds en français, de façon concise, avec les chiffres réels issus des données,
   et précise l'unité ou la feuille concernée quand c'est utile.
"""

agent = agents.Agent(
    model="mistral-large-4",
    name="xlsx-data-agent",
    description=(
        "Agent d'analyse de fichiers XLSX depuis Google Drive: chargement, description des "
        "colonnes, filtrage de lignes et agrégats par groupe, en langage naturel, sans exécution de code."
    ),
    instructions=INSTRUCTIONS,
    tools=[load_dataset, dataset_info, describe_columns, filter_rows, aggregate],
    connectors=[agents.connector("google_drive_mcp")],
    builtins=agents.Builtins(programmatic_tool_calling=False, subagents=False),
)
