Contributing Code
=================

We welcome code contributions via GitHub pull requests. To help with
maintainability of the code line we ask that the code uses a coding style
consistent with the rest of WA code. Briefly, it is

- `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ with line length and block
  comment rules relaxed (the wrapper for PEP8 checker inside ``dev_scripts``
  will run it with appropriate configuration).
- Four-space indentation (*no tabs!*).
- Title-case for class names, underscore-delimited lower case for functions,
  methods, and variables.
- Use descriptive variable names. Delimit words with ``'_'`` for readability.
  Avoid shortening words, skipping vowels, etc (common abbreviations such as
  "stats" for "statistics", "config" for "configuration", etc are OK). Do
  *not* use Hungarian notation (so prefer ``birth_date`` over ``dtBirth``).

New extensions should also follow implementation guidelines specified in the
:ref:`writing-plugins` section of the documentation.

We ask that the following checks are performed on the modified code prior to
submitting a pull request:

.. note:: You will need pylint and pep8 static checkers installed::

                pip install pep8
                pip install pylint

           It is recommended that you install via pip rather than through your
           distribution's package manager because the latter is likely to
           contain out-of-date version of these tools.

- ``./dev_scripts/pylint`` should be run without arguments and should produce no
  output (any output should be addressed by making appropriate changes in the
  code or adding a pylint ignore directive, if there is a good reason for
  keeping the code as is).
- ``./dev_scripts/pep8`` should be run without arguments and should produce no
  output (any output should be addressed by making appropriate changes in the
  code).
- If the modifications touch core framework (anything under ``wa/framework``), unit
  tests should be run using ``nosetests``, and they should all pass.

          - If significant additions have been made to the framework, unit
            tests should be added to cover the new functionality.

- If modifications have been made to documentation (this includes description
  attributes for Parameters and Extensions), documentation should be built to
  make sure no errors or warning during build process, and a visual inspection
  of new/updated sections in resulting HTML should be performed to ensure
  everything renders as expected.

Once you have your contribution is ready, please follow instructions in `GitHub
documentation <https://help.github.com/articles/creating-a-pull-request/>`_ to
create a pull request.
