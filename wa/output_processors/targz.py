import shutil
import tarfile

from wa import OutputProcessor, Parameter
from wa.framework import signal


class TargzProcessor(OutputProcessor):

    name = 'targz'

    description = '''
    Create a tarball of the output directory.

    This will create a gzip-compressed tarball of the output directory. By
    default, it will be created at the same level and will have the same name
    as the output directory but with a .tar.gz extensions.
    '''

    parameters = [
        Parameter('outfile',
                  description='''
                  The name  of the output file to be used. If this is not an
                  absolute path, the file will be created realtive to the
                  directory in which WA was invoked. If this contains
                  subdirectories, they must already exist.

                  The name may contain named format specifiers. Any of the
                  ``RunInfo`` fields can be named, resulting in the value of
                  that filed (e.g. ``'start_time'``) being formatted into the
                  tarball name.

                  By default, the output file will be created at the same
                  level, share the name of the WA output directory (but with
                  .tar.gz extension).
                  '''),
        Parameter('delete-output', kind=bool, default=False,
                  description='''
                  if set to ``True``, WA output directory will be deleted after
                  the tarball is created.
                  '''),
    ]

    def initialize(self):
        if self.delete_output:
            self.logger.debug('Registering RUN_FINALIZED handler.')
            signal.connect(self.delete_output_directory, signal.RUN_FINALIZED, priority=-100)

    def export_run_output(self, run_output, target_info):
        if self.outfile:
            outfile_path = self.outfile.format(**run_output.info.to_pod())
        else:
            outfile_path = run_output.basepath.rstrip('/') + '.tar.gz'

        self.logger.debug('Creating {}'.format(outfile_path))
        with tarfile.open(outfile_path, 'w:gz') as tar:
            tar.add(run_output.basepath)

    def delete_output_directory(self, context):
        self.logger.debug('Deleting output directory')
        shutil.rmtree(context.run_output.basepath)


