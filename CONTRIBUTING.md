# Contributing to LISA

First of all, if you're reading this, thanks for thinking about contributing!
This project is maintained by us Arm folks, but we welcome contributions from
anyone.

## How to reach us

If you're hitting an error/bug and need help, it's best to raise an issue on github.

## Coding style

As a rule of thumb, the code you write should follow the
[PEP-8](https://www.python.org/dev/peps/pep-0008/).

We strongly recommend using a code checker such as [pylint](https://www.pylint.org/),
as it tracks unused imports/variables, informs you when you can simplify a
statement using Python features, and overall just helps you write better code.

## Documentation

Docstring documentation should follow the ReST/Sphinx style.
Classes, class attributes and public methods must be documented. If deemed
necessary, private methods can be documented as well.

All in all, it should look like this:

```python
def foo(a, b):
	"""
	A one liner description

	:param a: A description for param a
	:type a: int

	:param b: A description for param b
	:type b: str

	Whatever extra description you might over as many lines as you need
	(but be reasonable)
	"""
	pass
```

## Tests

You should strive to validate as much of your code as possible through self-tests.
It’s a nice way to showcase that your code works, and also how it works. On top
of that, it makes sure that later changes won’t break it.

Have a look at [the doc](https://lisa-linux-integrated-system-analysis.readthedocs.io/en/master/contributors_guide.html#validating-your-changes) for more info on LISA self-tests.

## Commits

As for the shape of the commit, nothing out of the ordinary: just follow the
good old 50/72 rule (it's okay if you bite off a few extra chars).

The header should highlight the impacted files/classes. The 'lisa' prefix can be omitted - for instance,
if you're modifying the `lisa/wlgen/rta.py` file, we'd expect a header of the shape `wlgen/rta: ...`.

When that path gets a bit verbose, it's alright to shorten it as long as there
is no confusion as to what you're referencing. In that case, if modifying the
`lisa/tests/kernel/scheduler/load_tracking.py` file, we'd expect a header of
the shape `tests: load_tracking: ...`.

When in doubt, have a look at the git log.
