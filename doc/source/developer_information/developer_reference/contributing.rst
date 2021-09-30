Contributing
============

Code
----

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

- If modifications have been made to the UI Automation source of a workload, the
  corresponding APK should be rebuilt and submitted as part of the same pull
  request. This can be done via the ``build.sh`` script in the relevant
  ``uiauto`` subdirectory.
- If modifications have been made to documentation (this includes description
  attributes for Parameters and Extensions), documentation should be built to
  make sure no errors or warning during build process, and a visual inspection
  of new/updated sections in resulting HTML should be performed to ensure
  everything renders as expected.

Once you have your contribution is ready, please follow instructions in `GitHub
documentation <https://help.github.com/articles/creating-a-pull-request/>`_ to
create a pull request.

--------------------------------------------------------------------------------

Documentation
-------------

Headings
~~~~~~~~

To allow for consistent headings to be used through out the document the
following character sequences should be used when creating headings

::

        =========
        Heading 1
        =========

        Only used for top level headings which should also have an entry in the
        navigational side bar.

        *********
        Heading 2
        *********

        Main page heading used for page title, should not have a top level entry in the
        side bar.

        Heading 3
        ==========

        Regular section heading.

        Heading 4
        ---------

        Sub-heading.

        Heading 5
        ~~~~~~~~~

        Heading 6
        ^^^^^^^^^

        Heading 7
        """""""""


--------------------------------------------------------------------------------

Configuration Listings
~~~~~~~~~~~~~~~~~~~~~~

To keep a consistent style for presenting configuration options, the preferred
style is to use a `Field List`.

(See: http://docutils.sourceforge.net/docs/user/rst/quickref.html#field-lists)

Example::

        :parameter: My Description

Will render as:

        :parameter: My Description


--------------------------------------------------------------------------------

API Style
~~~~~~~~~

When documenting an API the currently preferred style is to provide a short
description of the class, followed by the attributes of the class in a
`Definition List` followed by the methods using the `method` directive.

(See: http://docutils.sourceforge.net/docs/user/rst/quickref.html#definition-lists)


Example::

        API
        ===

        :class:`MyClass`
        ----------------

        :class:`MyClass` is an example class to demonstrate API documentation.

        ``attribute1``
            The first attribute of the example class.

        ``attribute2``
            Another attribute example.

        methods
        """""""

        .. method:: MyClass.retrieve_output(name)

            Retrieve the output for ``name``.

            :param name:  The output that should be returned.
            :return: An :class:`Output` object for ``name``.
            :raises NotFoundError: If no output can be found.


Will render as:

:class:`MyClass` is an example class to demonstrate API documentation.

``attribute1``
    The first attribute of the example class.

``attribute2``
    Another attribute example.

methods
^^^^^^^

.. method:: MyClass.retrieve_output(name)

    Retrieve the output for ``name``.

    :param name:  The output that should be returned.
    :return: An :class:`Output` object for ``name``.
    :raises NotFoundError: If no output can be found.
