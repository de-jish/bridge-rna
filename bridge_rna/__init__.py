"""Bridge RNA: retrieval of Earth transcriptomic analogs for a NASA OSDR sample.

The retrieval half of the app. The map half is the sibling `manifold` package;
`app.py` mounts both behind one shell.

This was a single 2,470-line module until the two halves were merged. Splitting
it was not tidiness for its own sake: the layout and callbacks had to become
functions the router could mount, and once they were, the data layer underneath
had no reason to stay welded to them.
"""
