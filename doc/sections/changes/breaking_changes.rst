****************
Breaking changes
****************

Here is a list of commits introducing breaking changes in LISA:

.. exec::

    from lisa.utils import LISA_HOME
    from lisa._git import find_commits, log

    pattern = 'BREAK'

    repo = LISA_HOME
    commits = find_commits(repo, grep=pattern)
    ignored_sha1s = {
        '30d75656c7ff8a159dd52164269e69eed6dfccad',
    }
    for sha1 in commits:
        if sha1 in ignored_sha1s:
            continue
        commit_log = log(repo, ref=sha1, format='%cd%n%H%n%B')
        entry = '.. code-block:: text\n\n  {}\n'.format(commit_log.replace('\n', '\n  '))
        print(entry)

