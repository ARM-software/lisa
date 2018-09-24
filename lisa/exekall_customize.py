#! /usr/bin/env python3

import contextlib
import pathlib
import itertools
import functools
import logging
import sys

from lisa.env import TargetConfig, ArtifactPath
from lisa.utils import HideExekallID, Loggable

from exekall import utils, engine
from exekall.engine import reusable, ExprData, Consumer, PrebuiltOperator, NoValue, get_name
from exekall.customization import AdaptorBase

@reusable(False)
class ArtifactStorage(ArtifactPath, Loggable, HideExekallID):
    __slots__ = ('artifact_root',)

    def __new__(cls, root, *args, **kwargs):
        root = pathlib.Path(root).resolve()
        # We treat relative paths as relative to the results root
        path = super().__new__(cls, root, *args, **kwargs).resolve()
        path.artifact_root = root
        return path

    def __reduce__(self):
        # Serialize the path relatively to the root, so it can be relocated
        # easily
        relative = self.relative_to(self.artifact_root)
        return (type(self), (self.artifact_root, *relative.parts))

    def with_artifact_root(self, artifact_root):
        # Get the path relative to the old root
        old_artifact_root = self.artifact_root
        relative = self.relative_to(old_artifact_root)

        # Swap-in the new artifact_root
        return type(self)(artifact_root, relative)

    @classmethod
    def from_expr_data(cls, data:ExprData, consumer:Consumer) -> 'ArtifactStorage':
        """
        Factory used when running under `exekall`
        """
        artifact_root = pathlib.Path(data['artifact_root']).resolve()
        root = data['testcase_artifact_root']
        consumer_name = get_name(consumer)
        cls.get_logger().info('Creating storage for: ' + consumer_name)

        # Find a non-used directory
        for i in itertools.count(1):
            artifact_dir = pathlib.Path(root, consumer_name, str(i))
            if not artifact_dir.exists():
                break

        # Get canonical absolute paths
        artifact_dir = artifact_dir.resolve()

        artifact_dir.mkdir(parents=True)
        relative = artifact_dir.relative_to(artifact_root)
        return cls(artifact_root, relative)

class LISAAdaptor(AdaptorBase):

    def __init__(self, args):
        super().__init__(args)

    def get_prebuilt_list(self):
        if not self.args.target_conf:
            return []
        return [
            PrebuiltOperator(TargetConfig, [
                TargetConfig.from_path(conf)
                for conf in self.args.target_conf
            ])
        ]

    def get_hidden_callable_set(self, op_map):
        hidden_callable_set = set()
        for produced, op_set in op_map.items():
            if issubclass(produced, HideExekallID):
                hidden_callable_set.update(op.callable_ for op in op_set)

        self.hidden_callable_set = hidden_callable_set
        return hidden_callable_set

    @staticmethod
    def register_cli_param(parser):
        parser.add_argument('--target-conf', action='append',
            help="Target config files")

    def get_db_loader(self):
        return self.load_db

    @classmethod
    def load_db(cls, db_path, *args, **kwargs):
        # This will relocate ArtifactStorage instances to the new absolute path
        # of the results folder, in case it has been moved to another place
        artifact_root = pathlib.Path(db_path).parent.resolve()
        db = engine.StorageDB.from_path(db_path, *args, **kwargs)

        # Relocate ArtifactStorage embeded in objects so they will always
        # contain an absolute path that adapts to the local filesystem
        for serial in db.obj_store.get_all():
            val = serial.value
            try:
                dct = val.__dict__
            except AttributeError:
                continue
            for attr, attr_val in dct.items():
                if isinstance(attr_val, ArtifactStorage):
                    setattr(val, attr,
                        attr_val.with_artifact_root(artifact_root)
                    )

        return db

    def finalize_expr(self, expr):
        testcase_artifact_root = expr.data['testcase_artifact_root']
        artifact_root = expr.data['artifact_root']
        for expr_val in expr.get_all_values():
            self._finalize_expr_val(expr_val, artifact_root, testcase_artifact_root)

    def _finalize_expr_val(self, expr_val, artifact_root, testcase_artifact_root):
        val = expr_val.value

        # Add symlinks to artifact folders for ExprValue that were used in the
        # ExprValue graph, but were initially computed for another Expression
        if isinstance(val, ArtifactStorage):
            try:
                # If the folder is already a subfolder of our artifacts, we
                # don't need to do anything
                val.relative_to(testcase_artifact_root)
            # Otherwise, that means that such folder is reachable from our
            # parent ExprValue and we want to get a symlink to them
            except ValueError:
                # We get the name of the callable
                callable_folder = val.parts[-2]
                folder = testcase_artifact_root/callable_folder

                # We build a relative path back in the hierarchy to the root of
                # all artifacts
                relative_artifact_root = pathlib.Path(*(
                    '..' for part in
                    folder.relative_to(artifact_root).parts
                ))

                # The target needs to be a relative symlink, so we replace the
                # absolute artifact_root by a relative version of it
                target = relative_artifact_root/val.relative_to(artifact_root)

                with contextlib.suppress(FileExistsError):
                    folder.mkdir(parents=True)

                for i in itertools.count(1):
                    symlink = pathlib.Path(folder, str(i))
                    if not symlink.exists():
                        break

                symlink.symlink_to(target, target_is_directory=True)

        for param, param_expr_val in expr_val.param_value_map.items():
            self._finalize_expr_val(param_expr_val, artifact_root, testcase_artifact_root)


