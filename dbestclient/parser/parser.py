import re

import sqlparse
from sqlparse.sql import Function, Identifier, IdentifierList
from sqlparse.tokens import DDL, DML, Keyword


class DBEstParser:
    """
    parse a single SQL query, of the following form:

    - **DDL**
        >>> CREATE TABLE t_m(y real, x real)
        >>> FROM tbl
        >>> [GROUP BY z]
        >>> [SIZE 0.01]
        >>> [METHOD UNIFROM|HASH]
        >>> [SCALE FILE|DATA]
        >>> [ENCODING ONEHOT|BINARY]

    - **DML**
        >>> SELECT AF(y)
        >>> FROM t_m
        >>> [WHERE x BETWEEN a AND b]
        >>> [GROUP BY z]

    .. note::
        - model name should be ended with **_m** to indicate that it is a model, not a table.
        - AF, or aggregate function, could be COUNT, SUM, AVG, VARIANCE, PERCENTILE, etc.
    """

    def __init__(self):
        self.query = ""
        self.parsed = None

    def parse(self, query):
        """
        parse a single SQL query, of the following form:

        - **DDL**
            >>> CREATE TABLE t_m(y real, x_1 real, ... x_n categorical)
            >>> FROM tbl
            >>> [GROUP BY z]
            >>> [SIZE 0.01]
            >>> [METHOD UNIFROM|HASH]
            >>> [SCALE FILE|DATA]

        - **DML**
            >>> SELECT AF(y)
            >>> FROM t_m
            >>> [WHERE x BETWEEN a AND b]
            >>> [GROUP BY z]

        - **parameters**
        :param query: a SQL query
        """

        self.query = re.sub(' +', ' ', query).replace(" (", "(")  # query
        self.parsed = sqlparse.parse(self.query)[0]

    def if_nested_query(self):
        idx = 0
        if not self.parsed.is_group:
            return False
        for item in self.parsed.tokens:
            if item.ttype is DML and item.value.lower() == 'select':
                idx += 1
        if idx > 1:
            return True
        return False

    def get_aggregate_function_and_variable(self):
        for item in self.parsed.tokens:
            if item.ttype is DML and item.value.lower() == 'select':
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].tokens[0].value, \
                    self.parsed.tokens[idx].tokens[1].value.replace(
                        "(", "").replace(")", "")

    def if_where_exists(self):
        for item in self.parsed.tokens:
            if 'where' in item.value.lower():
                return True
        return False

    def get_where_name_and_range(self):
        for item in self.parsed.tokens:
            if 'where' in item.value.lower():
                whereclause = item.value.split()
                # print(whereclause)
                return whereclause[1], whereclause[3], whereclause[5]

    def if_contain_groupby(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                return True
        return False

    def if_contain_scaling_factor(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "scale":
                return True
        return False

    def get_scaling_method(self):
        if not self.if_contain_scaling_factor():
            return "data"
        else:
            for item in self.parsed.tokens:
                if item.ttype is Keyword and item.value.lower() == "scale":
                    idx = self.parsed.token_index(item, 0) + 2
                    if self.parsed.tokens[idx].value.lower() not in ["data", "file"]:
                        raise ValueError(
                            "Scaling method is not set properly, wrong argument provided.")
                    else:
                        file = self.parsed.tokens[idx+2].value.lower()
                        method = self.parsed.tokens[idx].value.lower()
                        if method == "file":
                            return method, file
                        else:
                            return method, None

    def get_groupby_value(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                idx = self.parsed.token_index(item, 0) + 2
                groups = self.parsed.tokens[idx].value
                return groups.replace(" ", "").split(",")

    def if_ddl(self):
        for item in self.parsed.tokens:
            if item.ttype is DDL and item.value.lower() == "create":
                return True
        return False

    def get_ddl_model_name(self):
        for item in self.parsed.tokens:
            if item.ttype is None and "(" in item.value.lower():
                return item.tokens[0].value

    def get_y(self):
        for item in self.parsed.tokens:
            if item.ttype is None and "(" in item.value.lower():
                return item.tokens[1].tokens[1].value, item.tokens[1].tokens[3].value

    def get_x(self):
        for item in self.parsed.tokens:
            if item.ttype is None and "(" in item.value.lower():
                return item.tokens[1].tokens[6].value, item.tokens[1].tokens[8].value

    def get_from_name(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "from":
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].value

    def get_sampling_ratio(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "size":
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].value
        return 0.01  # if sampling ratio is not passed, the whole dataset will be used to train the model

    def get_sampling_method(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "method":
                idx = self.parsed.token_index(item, 0) + 2
                return self.parsed.tokens[idx].value
        return "uniform"


if __name__ == "__main__":
    parser = DBEstParser()
    # parser.parse("create table mdl(y real, x real) from tbl group by z method uniform size 0.1 ")
    #
    # if parser.if_contain_groupby():
    #     print("yes, group by")
    #     print(parser.get_groupby_value())
    # else:
    #     print("no group by")
    #
    # if parser.if_ddl():
    #     print("ddl")
    #     print(parser.get_ddl_model_name())
    #     print(parser.get_y())
    #     print(parser.get_x())
    #     print(parser.get_from_name())
    #     print(parser.get_sampling_method())
    #     print(parser.get_sampling_ratio())

    parser.parse(
        "select count(y) from t_m where x BETWEEN  1 and 2 GROUP BY z1, z2 ,z3 method uniform")  # scale file
    print(parser.if_contain_scaling_factor())
    parser.parse(
        "select count(y) from t_m where x BETWEEN  1 and 2 GROUP BY z1, z2 ,z3 method uniform scale file   haha/num.csv  size 23")
    print(parser.if_contain_scaling_factor())
    if parser.if_contain_groupby():
        print("yes, group by")
        print(parser.get_groupby_value())
    else:
        print("no group by")
    if not parser.if_ddl():
        print("DML")
        print(parser.get_aggregate_function_and_variable())

    if parser.if_where_exists():
        print("where exists!")
        print(parser.get_where_name_and_range())

    print("method, ", parser.get_sampling_method())

    print("scaling factor ", parser.get_scaling_method())
