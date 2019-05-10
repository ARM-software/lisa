Man page
========

Description
+++++++++++

``exekall`` is a python-based test runner. The expressions it executes are
discovered from Python PEP 484 parameter and return value annotations.

Options
+++++++

exekall
-------

.. run-command::
   :ignore-error:
   :literal:

   exekall --help

exekall run
-----------

.. run-command::
   :ignore-error:
   :literal:

   # Give the python sources to exekall to get the LISA options in addition to
   # the generic ones.
   exekall run "$LISA_HOME/lisa/tests/" --help

exekall compare
---------------

.. run-command::
   :ignore-error:
   :literal:

   exekall compare --help

exekall show
------------

.. run-command::
   :ignore-error:
   :literal:

   exekall show --help

exekall merge
-------------

.. run-command::
   :ignore-error:
   :literal:

   exekall merge --help


Executing expressions
+++++++++++++++++++++

Expressions are built by scanning the python source code passed to ``exekall
run``. Selecting which expression to execute using ``exekall run`` can be
achieved in several ways:

   * ``--select``/``-s`` with a pattern matching an expression ID. Pattern
     prefixed with **!** can be used to exclude some expressions.
   * Pointing ``exekall run`` at a subset of python source files. Only files
     (directly or indirectly) imported from these python modules will be
     scanned for callables.

Once the expressions are selected, multiple iterations of it can be executed
using ``-n``. ``--share TYPE_PATTERN`` can be used to share part of the expression
graph between all iterations, to avoid re-executing some parts of the
expression. Be aware that all parameters of what is shared will also be shared
implicitly to keep consistent expressions.

The adaptor found in the customization module of the python sources you are
using can add extra options to ``exekall run``, which are shown in ``--help``
only when these sources are specified as well. 

Expression engine
+++++++++++++++++

At the core of ``exekall`` is the expression engine. It is in charge of
building sensible sequences of calls out of python-level annotations (see PEP
484), and then executing them. An expression is a graph where each node has
named *parameters* that point to other nodes. 

Expression ID
-------------

Each expression has an associated ID that is derived from its structure. The rules are:

   1. The ID of the first parameter of a given node is prepended to the ID of
      the node, separated with **:**.  The code :code:`f(g())` has the ID
      ``g:f``.
   2. The ID of the node is composed of the name of the operator of that node
      (name of a Python callable), followed by a
      parenthesis-enclosed list of parameters ID, excluding the first
      parameter. The code :code:`f(p1=g(), p2=h(k()))` has the ID
      ``g:f(p2=k:h)``. 
   3. Expression values can have named tags attached to them. When displaying
      the ID of such a value, the tag would be inserted right after the
      operator name, inside brackets. The value returned by ``g`` tagged with a
      tag named ``mytag`` with value ``42`` would give:
      ``g[mytag=42]:f(p2=k:h)``. Note that tags are only relevant when using
      expression values, since the tags are attached to values, not operators.

The first rule allows seamless composition of simple pipeline stages and is
especially suited to object oriented programming, since the first parameter of
methods will be ``self``.

Tags can be used to add attach some important metadata to the return value of
an operator, so it can be easily distinguished when taken out of context.

Sharing subexpressions
----------------------

When multiple expressions are to be executed, ``exekall`` will eliminate common
subexpressions. That will apply both inside an expression and between different
expressions. That avoids re-executing the same operator multiple times if it
can be reused and if it would have been called with the same parameters. That
also ensures that referring to a given type for a parameter will give back the
same object within any given expression. Executing the IDs ``g:f(p2=g)`` and
``g:h`` will translate to an expression graph equivalent to::

   x = g()
   res1 = f(x, p2=x)
   res2 = h(x)

The expression execution engine logs when a given value is computed or reused.

Execution
---------

Executing an expression means evaluating each node if it has not already been
evaluated. If an operator is not reusable, it will always be called when a
value is requested from it, even if some existing values computed with the same
parameters exist. By default, all operators are reusable, but some types can be
flagged as non-reusable by the customization module (see :ref:`customize`).

Operators are allowed to be generator functions as well. In that case, the
engine will iterate over the generator, and will execute the downstream
expressions for each value it provides. Multiple generator functions can be
chained, leading to a cascade of values for the same expression.

Once an expression has been executed, all its values will get a UUID that can
be used to uniquely refer to it, and track where it was used in the logs.

Exploiting artifacts
++++++++++++++++++++

``exekall run`` produces an artifact folder. The location can be set using
``--artifact-dir`` and other options.

Folder hierarchy
----------------

The artifact folder contains the following files:

   * **INFO.log** and **DEBUG.log** contain logs for info and debug levels of the
     ``logging`` standard module. Note that standard output is not included in
     this log, as it does not go through the ``logging`` module
   * **ValueDB.pickle.xz** contains a serialized objects graph for each
     expression that was executed. The value of each subexpression is included
     if the object was serializable.
   * **BY_UUID** contains symlinks named after UUIDs, and pointing to a
     relevant subfolder in the artifacts. That allows quick lookup of the
     artifacts of a given expression if one has its UUID.
   * A folder for each expression.
   * Optionally, an **ORIGIN** folder if the artifact folder is the result of
     **exekall merge**, or **exekall run --load-db**. It contains the hierarchy
     of each original artifact folder by using folders and symlinks pointing
     inside the artifact folder.

Inside each expression's folder, there is a folder with the UUID of the
expression itself. Having that level allows merging artifact folders together
and avoids conflict in case two different expressions share the same ID.

Inside that folder, the following files can be found:
   
   * **STRUCTURE** which contains the structure of the expression. Each
     operator is described by its callable name, its return type, and its
     parameters. Parameters are recursively defined the same way. An **svg** or
     **.dot** (graphviz) variant may exist as well.
   * **EXPRESSION.py** and **TEMPLATE_EXPRESSION.py** files are executable
     Python script that are equivalent to what was executed by ``exekall run``.
     The template one is created before execution and contains some
     placeholders for the sparks. The other one is updated after execution to
     add commented code that reloads any given value from the database. That
     gives the option to the user to not re-execute some part of the code, but
     load a serialized value instead.
   * Artifact folders allocated by some operators.
   
exekall compare
---------------

**ValueDB.pickle.xz** can be compared using ``exekall compare``. This will call the
comparison method of the adaptor that was used when ``exekall run`` was
executed. That function is expected to compare the expression values found in
the databases, by matching values that have the same ID on both databases.

Adding new expressions
++++++++++++++++++++++

Since ``exekall run`` will discover expressions based on type annotations of
callable parameters and return value, all that is needed to extend an existing
package is to write new callables with such annotations. It is possible to use
a base class in an annotation, in which case the engine will be free to pick
all the subclasses it can, and produce an expression with each. A dummy example
would be::

   import abc
   class BaseConf(abc.ABC):
      @abc.abstractmethod
      def get_conf(self):
         pass

   class Conf(BaseConf):
      # By default, callables with an empty parameter list are ignored. They
      # can be explicitly be used with "exekall run --allow '*Conf'"
      def __init__(self):
         self.x = 42

      def get_conf(self):
         return x

   class Stage1:
      # exekall recognizes classes as a special case: the parameter annotations
      # are taken from __init__ and the return type is the class
      def __init__(self, conf:BaseConf):
         print("building stage1")
         self.conf = conf

      # first parameter of methods is automatically annotated with the right
      # class.
      # "forward-references are possible by using a string to annotate.
      def process_method(self) -> 'Stage2':
         return Stage2(x.conf.x == 42)

   class Stage2:
      def __init__(self, passed):
         self.passed = passed

   def process1(x:Stage1) -> Stage2:
      return Stage2(x.conf.x == 42)

   def process2(x:Stage1, conf:BaseConf, has_default_val=33) -> Stage2:
      return Stage2(x.conf.x == 0)

From that, ``exekall run --allow '*Conf' --goal '*Stage2'`` would infer the
expressions ``Conf:Stage1:process_method``, ``Conf:Stage1:process1`` and
``Conf:Stage1:process2(conf=Conf)``. The common subexpression ``Conf:Stage1`` would be
shared between these two by default.

If a parameter has a default value, its annotation can be omitted. If a
parameter has both a default value and an annotation, ``exekall`` will try to
provide a value for it, or use the default value if no subexpression has the right
type.

When an expression is not detected correctly, ``--verbose``/``-v`` can be used and
repeated twice to get more information on what callables are being ignored and
why. Most common issues are:

   * Partial annotations: all parameters and return values need to be either
     annotated or have a default value.
   * Abstract Base Classes (see :class:`abc.ABC`) with missing implementation
     of some attributes.
   * Cycles in the expression graphs. Considering types as pipeline stages
     helps avoiding cycles in expression graphs when architecturing a module.
     Not all classes need to be considered as such, only the ones that will be
     used in annotations.
   * Missing "spark", i.e. operator that can provide values without any
     parameter. The adaptor in the customization module usually takes care of
     doing that based on domain-specific command line options, but some ignored
     callables may be forcefully selected using ``--allow`` if needed.
   * Missing ``import`` chain from the sources given to ``exekall run`` to the
     module that defines the callable that is expected to be used. That can be
     solved by adding more ``import`` statements, or simply giving that source
     file directly to ``exekall run``.
   * Wrong goal selected using ``--goal``.

.. _customize:

Customizing exekall
+++++++++++++++++++

The behavior of ``exekall`` can be customized by subclassing
:class:`exekall.customization.AdaptorBase` in a module that must be called
``exekall_customization.py`` and located in one of the parent packages of the
modules that are explicitly passed to ``exekall run``.  This allows adding
extra options to ``exekall run`` and ``compare``, tag values in IDs, change the
set of callables that will be hidden from the ID and define what type is
considered to provide reusable values by the engine among other things.

