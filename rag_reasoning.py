import pandas as pd
import numpy as np
import re
from datascience import * # type: ignore
from typing import Literal, Callable

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

class ReasoningDataQuery:
    def __init__(self, selected_columns: list[str], dnf: list[list[tuple[str, str, str]]]):
        self.selected_columns = selected_columns
        self.dnf = dnf

    
    def interpret(self, entity_name: str) -> str:
        dnf_interp: list[str] = []
        for cnf in self.dnf:
            if not cnf:
                continue
            cnf_interp: list[str] = []
            for expression in cnf:
                operator, lhs, rhs = expression
                if operator == "REACHES":
                    if rhs == "MIN":
                        cnf_interp.append(f"{lhs} đạt giá trị nhỏ nhất")
                    elif rhs == "MAX":
                        cnf_interp.append(f"{lhs} đạt giá trị lớn nhất")
                    else:
                        raise ValueError(f"Invalid REACHES value: {rhs}")
                else:
                    o = operator if operator != "CONTAINS" else "chứa cụm từ hoặc số "
                    cnf_interp.append(f"{lhs} {o} \"{rhs}\"")
            dnf_interp.append(" và ".join(cnf_interp))

        if len(dnf_interp) == 0:
            return f"Tất cả các {entity_name}"
        elif len(dnf_interp) == 1:
            return f"Tất cả các {entity_name} thỏa điều kiện: " + dnf_interp[0]
        else:
            return f"Tất cả các {entity_name} thỏa một trong các điều kiện sau:\n\n" + "\n\n".join(
                (
                    f"Điều kiện {i}: {cnf_interp}"
                    for i, cnf_interp in enumerate(dnf_interp, start=1) if cnf_interp
                )
            )

# https://regex101.com/r/8q3J89/1
re_select_and_where = re.compile(r'^\s*SELECT\s+(?P<selected_columns>(\"([^\"])+\")(\s*,\s*(\"([^\"])+\"))*)\s*WHERE\s+(?P<where_clause>.*)$', re.MULTILINE)

re_simple_expression = re.compile(r'(?P<lhs>\"([^\"])+\")\s+((REACHES\s+(?P<reaches>MIN|MAX))|((?P<infixOperator>=|<=|>=|>|<|\!=|CONTAINS)\s+(?P<rhs>\"([^\"])+\")))', re.MULTILINE)

class ReasoningDataQueryEngine:
    def __init__(self, df: pd.DataFrame, embedder: Callable[[str], np.ndarray]):
        # Convert possibly numeric columns to numeric types
        for col in df.columns:
            col_non_empty = df[col].replace("", pd.NA).dropna()
            # Try converting to numeric
            converted = pd.to_numeric(col_non_empty, errors="coerce")

            # If most of the non-empty values are numeric, convert the whole column
            if converted.notna().sum() >= len(col_non_empty) * 0.9:
                df[col] = pd.to_numeric(df[col], errors="coerce")  # convert with NaNs preserved

        self.table = Table().from_df(df)
        self.embedder = embedder
    
    @staticmethod
    def compile(query: str):
        result = re_select_and_where.match(query)
        if not result:
            raise ValueError("Invalid query format. Expected format: SELECT <columns> WHERE <condition>")
        selected_columns = result.group('selected_columns').split(',')
        selected_columns = [col.strip().strip('"') for col in selected_columns]
        where_clause = result.group('where_clause')

        remaining_where_clause = where_clause.strip()

        # A list of Conjunctive Normal Form (CNF) expressions
        # So this basically is a Disjunctive Normal Form (DNF) expression as a whole!
        dnf: list[list[tuple[str, str, str]]] = []
        cnf: list[tuple[str, str, str]] = []
        first_expression = True
        while remaining_where_clause:
            simple_expression_match = re.search(re_simple_expression, remaining_where_clause)
            if not simple_expression_match:
                raise ValueError("Invalid WHERE clause format. Expected format: <column> <operator> <value>")

            start, end = simple_expression_match.span()
            previous = remaining_where_clause[:start].strip()
            if previous:
                if previous == "OR":
                    dnf.append(cnf)
                    cnf = []
                elif previous == "AND":
                    pass
                else:
                    if first_expression:
                        first_expression = False
                    else:
                        raise ValueError("Invalid WHERE clause format. Expecting AND or OR in this position")

            lhs = simple_expression_match.group('lhs')
            lhs = lhs.strip('"')
            infixOperator = simple_expression_match.group('infixOperator')
            if infixOperator:
                rhs = simple_expression_match.group('rhs')
                rhs = rhs.strip('"')
                infixOperator = infixOperator.strip()
                cnf.append((infixOperator, lhs, rhs))
            else:
                rhs = simple_expression_match.group('reaches')
                if rhs:
                    infixOperator = "REACHES"
                    rhs = rhs.strip()
                    cnf.append((infixOperator, lhs, rhs))

            remaining_where_clause = remaining_where_clause[end:].strip()
        
        if cnf:
            dnf.append(cnf)
            cnf = []

        return ReasoningDataQuery(selected_columns, dnf)
    
    def apply(self, query: ReasoningDataQuery):
        try:
            original_table = self.table.with_columns(
                "_P_index", range(self.table.num_rows)
            )
            filtered_rows_index_to_order_map: dict[int, int] = {}

            for cnf in query.dnf:
                if not cnf:
                    continue
                table = original_table.copy()
                cnf.sort(key=lambda expression: 1 if expression[0] == "CONTAINS" else 10)
                for expression in cnf:
                    table = self._filter_by_expression(table, expression)
                for i in table.column("_P_index"):
                    if i not in filtered_rows_index_to_order_map:
                        filtered_rows_index_to_order_map[i] = len(filtered_rows_index_to_order_map)
            
            original_table = original_table.with_columns(
                "_P_order", original_table.apply(
                    lambda pIndex: filtered_rows_index_to_order_map.get(pIndex, -1), "_P_index"
                )
            )
            
            filtered_table: Table = original_table.where("_P_order", are.not_equal_to(-1)).sort("_P_order")#.drop("_P_index", "_P_order")

            selected_columns = filter(
                lambda x: x in filtered_table.labels,
                [*query.selected_columns, "Chi tiết"],
            )

            result_table = filtered_table.select(selected_columns)
            result_df = result_table.to_df()
            result_df = result_df.dropna(axis=1, how='all')
            result_table = Table().from_df(result_df)
            return result_table
        except Exception as e:
            print(f"Error applying query: {e}")
            return Table()
    
    def _filter_by_expression(self, table: Table, expression: tuple[str, str, str]):
        operator, lhs, _rhs = expression

        if lhs not in table.labels:
            print(f"NOTE: Column '{lhs}' not found in the table, whose columns are {table.labels}")
            return table

        if operator == "REACHES":
            if _rhs == "MIN":
                minValue = table.column(lhs).min()
                if (table.num_rows > 0):
                    return table.sort(lhs).take(np.arange(0, min(table.num_rows, 5)))
                else:
                    return table
            elif _rhs == "MAX":
                maxValue = table.column(lhs).max()
                if (table.num_rows > 0):
                    return table.sort(lhs, descending=True).take(np.arange(0, min(table.num_rows, 5)))
                else:
                    return table
            else:
                raise ValueError(f"Invalid REACHES value: {_rhs}")
        else:
            rhs = self._normalize_rhs(_rhs, table.column(lhs))
            if operator == "=":
                return table.where(lhs, are.equal_to(rhs))
            elif operator == "<=":
                return table.where(lhs, are.below_or_equal_to(rhs))
            elif operator == ">=":
                return table.where(lhs, are.above_or_equal_to(rhs))
            elif operator == "<":
                return table.where(lhs, are.below(rhs))
            elif operator == ">":
                return table.where(lhs, are.above(rhs))
            elif operator == "!=":
                return table.where(lhs, are.not_equal_to(rhs))
            elif operator == "CONTAINS":
                t = table.with_column(
                    "_P_relevance", self._compute_similarity_vector(str(rhs), [str(x) for x in table.column(lhs)])
                ).sort("_P_relevance", descending=True).drop("_P_relevance").take(np.arange(0, min(table.num_rows, 5)))
                return t
            else:
                raise ValueError(f"Invalid operator: {operator}")
    
    @classmethod
    def _normalize_rhs(cls, rhs: str, column: np.ndarray):
        if np.issubdtype(column.dtype, np.number):
            try:
                return float(rhs)
            except ValueError:
                return rhs

        return rhs
    
    def _compute_similarity_vector(self, query: str, candidates: list[str]) -> np.ndarray:
        query_embedding = np.array(self.embedder(query))
        candidate_embeddings = [np.array(self.embedder(c)) for c in candidates]
        similarities = [cosine_similarity(query_embedding, emb) for emb in candidate_embeddings]
        return np.array(similarities)