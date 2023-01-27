import os
import re
from dataclasses import replace

import sqlparse
from sqlparse.sql import Function, Identifier, IdentifierList
from sqlparse.tokens import DDL, DML, Keyword

from dbestclient.tools.date import to_timestamp


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

    def get_dml_aggregate_function_and_variable(self):
        values = self.parsed.tokens[2].normalized
        if "," in values:
            splits = values.split(",")
            # print(splits)
            y_splits = splits[1].replace(
                "(", " ").replace(")", " ")  # .split(" ")
            # print(y_splits)
            if "distinct" in y_splits.lower():
                y_splits = y_splits.split()
                # print(y_splits)
                return splits[0], [y_splits[i] for i in [0, 2, 1]]
            else:
                y_splits = y_splits.split()
                y_splits.append(None)
                return splits[0], y_splits
        else:
            y_splits = values.replace(
                "(", " ").replace(")", " ")
            if "distinct" in y_splits.lower():
                y_splits = y_splits.split()
                # print(y_splits)
                return None, [y_splits[i] for i in [0, 2, 1]]
            else:
                y_splits = y_splits.split()
                y_splits.append(None)
                return None, y_splits
        # for item in self.parsed.tokens:
        #     print(self.parsed.tokens[2].normalized)
        #     if item.ttype is DML and item.value.lower() == 'select':
        #         print(self.parsed.token_index)
        #         idx = self.parsed.token_index(item, 0) + 2
        #         return self.parsed.tokens[idx].tokens[0].value, \
        #             self.parsed.tokens[idx].tokens[1].value.replace(
        #                 "(", "").replace(")", "")

    def if_where_exists(self):
        for item in self.parsed.tokens:
            if 'where' in item.value.lower():
                return True
        return False

    def get_where_x_and_range(self):
        for item in self.parsed.tokens:
            # print(item)
            if 'where' in item.value.lower():
                whereclause = item.value.lower().split()
                # print(whereclause)
                idx = whereclause.index("between")
                # print(idx)
                return whereclause[idx-1], whereclause[idx+1], whereclause[idx+3]
                # return whereclause[1], whereclause[3], whereclause[5]

    def get_where_categorical_equal(self):
        for item in self.parsed.tokens:
            clause_lower = item.value.lower()
            clause = item.value
            if 'where' in clause_lower:

                # indexes = [m.start() for m in re.finditer('=', clause)]
                splits = clause.replace("=", " = ").split()
                splits_lower = clause_lower.replace("=", " = ").split()
                # print(clause)
                # print(clause.count("="))
                xs = []
                values = []
                while True:
                    if "=" not in splits:
                        break
                    idx = splits.index("=")
                    xs.append(splits_lower[idx-1])
                    if splits[idx+1] != "''":
                        values.append(splits[idx+1].replace("'", ""))
                    else:
                        values.append("")
                    splits = splits[idx+3:]
                    splits_lower = splits_lower[idx+3:]
                #     print(splits)
                # print(xs, values)
                return xs, values

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

                        method = self.parsed.tokens[idx].value.lower()
                        if method == "file":
                            file = self.parsed.tokens[idx+2].value.lower()
                            return method, file
                        else:
                            return method, None

    def get_groupby_value(self):
        groups = []
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "group by":
                idx = self.parsed.token_index(item, 0) + 2
                groups = self.parsed.tokens[idx].value
                groups = groups.replace(" ", "").split(",")
        return groups

    def if_ddl(self):
        for item in self.parsed.tokens:
            if item.ttype is DDL and item.value.lower() == "create":
                return True
        return False

    def get_ddl_model_name(self):
        # for item in self.parsed.tokens:
        #     if item.ttype is None and "(" in item.value.lower():
        #         return item.tokens[0].value
        return self.parsed.tokens[4].value

    def get_y(self):
        item = self.parsed.tokens[5].value  # original code had wrong index
        index_comma = item.index(",")
        item = item[:index_comma]
        y_list = item.lower().replace(
            "(", " ").replace(")", " ").replace(",", " ").split()
        # print("y_list", y_list)
        if y_list[1] not in ["real", "categorical"]:  # original code had wrong index
            raise TypeError("Unsupported type for " + y_list[0] + " -> " + y_list[1])
        # if item.ttype is None and "(" in item.value.lower():
        #     y_list = item.tokens[1].value.lower().replace(
        #         "(", "").replace(")", "").replace(",", " ").split()
        #     if y_list[1] not in ["real", "categorical"]:
        #         raise TypeError("Unsupported type for " +
        #                         y_list[0] + " -> " + y_list[1])
        if len(y_list) == 4:
            return [y_list[0], y_list[1], y_list[2]]
        else:
            return [y_list[0], y_list[1], None]

        # return item.tokens[1].tokens[1].value, item.tokens[1].tokens[3].value

    def get_x(self):
        item = self.parsed.tokens[5].value  # list of table columns and types
        # NOTE As far as I understand it, this filters out the first column because it
        # is assumed to be the aggregation column is the first column.
        # NOTE There may only be one (continuous) predicate column (i.e. independent
        # variable).
        index_comma = item.index(",")
        item = item[index_comma+1:]
        x_list = item.lower().replace(
            "(", "").replace(")", "").replace(",", " ").split()
        # print(x_list)
        continous = []
        categorical = []
        for idx in range(1, len(x_list), 2):
            if x_list[idx] == "real":
                continous.append(x_list[idx-1])
            elif x_list[idx] == "categorical":
                categorical.append(x_list[idx-1])

        if len(continous) > 1:
            raise SyntaxError(
                "Only one continous independent variable is supported at "
                "this moment, please modify your SQL query accordingly.")
        # print("continous,", continous)
        # print("categorical,", categorical)
        return continous, categorical

        # for item in self.parsed.tokens:
        #     print(item)
        #     if item.ttype is None and "(" in item.value.lower():
        #         x_list = item.tokens[1].value.lower().replace(
        #             "(", "").replace(")", "").replace(",", " ").split()
        #         continous = []
        #         categorical = []
        #         for idx in range(3, len(x_list), 2):
        #             if x_list[idx] == "real":
        #                 continous.append(x_list[idx-1])
        #             if x_list[idx] == "categorical":
        #                 categorical.append(x_list[idx-1])

        #         if len(continous) > 1:
        #             raise SyntaxError(
        #                 "Only one continous independent variable is supported at "
        #                 "this moment, please modify your SQL query accordingly.")
        #         # print("continous,", continous)
        #         # print("categorical,", categorical)
        #         return continous, categorical
        #         # return item.tokens[1].tokens[6].value, item.tokens[1].tokens[8].value

    def get_from_name(self):
        for item in self.parsed.tokens:
            if item.ttype is Keyword and item.value.lower() == "from":
                idx = self.parsed.token_index(item, 0) + 2
                from_value = self.parsed.tokens[idx].value.replace("'", "")
                return from_value

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

    def if_model_need_filter(self):
        x = self.get_x()
        gbs = self.get_groupby_value()
        # print("x", x)
        # NOTE This fails in the original code because gbs could be None. I changed this
        # in self.get_goupby_value.
        if x[0][0] in gbs:
            return True
        else:
            return False

    def get_filter(self):
        x_between_and = self.get_where_x_and_range()
        gbs = self.get_groupby_value()

        # print("x_between_and", x_between_and)
        if x_between_and[0] not in gbs:
            return None
        else:
            try:
                return [float(item) for item in x_between_and[1:]]
            except ValueError:
                # check if timestamp exists
                if "to_timestamp" in x_between_and[1]:
                    # print([to_timestamp(item.replace("to_timestamp(", "").replace(")", "").replace("'", "").replace('"', '')) for item in x_between_and[1:]])
                    return [to_timestamp(item.replace("to_timestamp(", "").replace(")", "").replace("'", "").replace('"', '')) for item in x_between_and[1:]]
                else:
                    raise ValueError("Error parse SQL.")


if __name__ == "__main__":
    parser = DBEstParser()
    # parser.parse(
    #     "create table mdl ( y categorical distinct, x0 real, x2 categorical, x3 categorical) from tbl group by z,x0 method uniform size 0.1 ")

    # if parser.if_contain_groupby():
    #     print("yes, group by")
    #     print(parser.get_groupby_value())
    # else:
    #     print("no group by")

    # if parser.if_ddl():
    #     print("ddl")
    #     print(parser.get_ddl_model_name())
    #     print(parser.get_y())
    #     print(parser.get_x())
    #     print(parser.get_from_name())
    #     print(parser.get_sampling_method())
    #     print(parser.get_sampling_ratio())
    #     print(parser.if_model_need_filter())

    # parser.parse(
    #     "select count(y) from t_m where x BETWEEN  1 and 2 GROUP BY z1, z2 ,z3 method uniform")  # scale file
    # print(parser.if_contain_scaling_factor())
    parser.parse(
        "select z, count ( y ) from t_m where x BETWEEN  to_timestamp('2019-02-28T16:00:00.000Z') and to_timestamp('2019-03-28T16:00:00.000Z') and X1 = grp and x2 = 'HaHaHa' and x3='' GROUP BY z1, z2 ,x method uniform scale data   haha/num.csv  size 23")
    print(parser.if_contain_scaling_factor())
    if parser.if_contain_groupby():
        print("yes, group by")
        print(parser.get_groupby_value())
    else:
        print("no group by")
    if not parser.if_ddl():
        print("DML")
        print(parser.get_dml_aggregate_function_and_variable())

    if parser.if_where_exists():
        print("where exists!")
        print(parser.get_where_x_and_range())
        parser.get_where_categorical_equal()

    print("method, ", parser.get_sampling_method())

    print("scaling factor ", parser.get_scaling_method())

    print(parser.get_where_x_and_range())
    print(parser.get_where_categorical_equal())

    print(parser.get_filter())
