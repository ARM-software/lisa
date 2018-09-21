#! /usr/bin/env python3

from exekall.engine import NoValue, get_name

class AdaptorBase:
    def __init__(self, args=None):
        if args is None:
            args = dict()
        self.args = args

    def get_db_loader(self):
        return None

    def filter_callable_pool(self, callable_pool):
        return callable_pool

    def filter_cls_map(self, cls_map):
        return cls_map

    def filter_op_map(self, op_map):
        return op_map

    def get_prebuilt_list(self):
        return []

    def get_hidden_callable_set(self, op_map):
        return set()

    @staticmethod
    def register_cli_param(parser):
        pass

    def resolve_cls_name(self, goal):
        return engine.get_class_from_name(goal, sys.modules)

    def load_db(self, db_path):
        return engine.StorageDB.from_path(db_path)

    def finalize_expr(self, expr):
        pass

    def result_str(self, result):
        val = result.value
        if val is NoValue or val is None:
            failed_parents = result.get_failed_values()
            for failed_parent in failed_parents:
                excep = failed_parent.excep
                return '{type}: {msg}'.format(
                    type = get_name(type(excep)),
                    msg = excep
                )
            return 'No result computed'
        else:
            return str(val)

    def process_results(self, result_map):
        hidden_callable_set = getattr(self, 'hidden_callable_set', set())
        for expr, result_list in result_map.items():
            for result in result_list:
                msg = self.result_str(result)
                msg = msg + '\n' if '\n' in msg else msg
                print('{id}: {result}'.format(
                    id=result.get_id(
                        hidden_callable_set=hidden_callable_set,
                        full_qual=False,
                    ),
                    result=msg,
                ))

    @classmethod
    def get_adaptor_cls(cls, name=None):
        for subcls in cls.__subclasses__():
            if name:
                if subcls.name == name:
                    return subcls
            else:
                return subcls
        return None

