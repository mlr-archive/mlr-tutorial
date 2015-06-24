# Tutorial
URL: http://mlr-org.github.io/mlr-tutorial/tutorial/current/html/

## Howto
* Install dependencies:
  `pip install --user mkdocs` or `easy_install --user mkdocs`.
  Install the [math extension for Python-Markdown](https://github.com/mitya57/python-markdown-math):
  After download type `python setup.py build` and `python setup.py install`.
  Install R dependencies as required.
* Only edit R markdown files in subfolder `src/`.
* To add a new section to the tutorial or change their irder you need to edit the pages configuration in `mkdocs.yml`.
* Markdown basics:
  * Basic Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
  * RStudio Support: https://support.rstudio.com/hc/en-us/articles/200552086-Using-R-Markdown
  * Knitr options: http://yihui.name/knitr/options
* Link to mlr manual: `[&function]` and `[name](&function)`.
* Link to other manuals: `[&pkg::function]` and `[name](&pkg::function)`.
* Links will only work properly if they point to the *base name* of the help file:
  For example, in order to link to mlr function `foo` documented on help page `bar` write
  `[foo](&bar)` instead of `[&foo]`.
* Link to packages: `[%pkg]` and `[name](%pkg)`.
* Put additional images in `../images`.
* Run `./build` to generate new static HTML.
* If everything works:
  Commit and push **only** your changes in `src/` (and `mkdocs.yml`) to update the tutorial.
  After your commit the HTML pages are automatically built and pushed by Travis.
  

## More
* "mkdocs serve" starts a http server listening on http://localhost:8000
  and updates the docs on change.
* Sometimes function names collide. These packages must be loaded _first_
  in "build". That way mlr overwrites these functions again, e.g. caret::train.
* The build caches the output of running the R commands in the cache/ directory.
  If your R setup has changed (e.g. new version of mlr), you should delete
  everything in the cache directory to make sure that the tutorial is
  regenerated with the new code.
