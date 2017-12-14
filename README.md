# mlr-tutorial

The mlr package online tutorial

[![Build Status](https://travis-ci.org/mlr-org/mlr-tutorial.svg?branch=gh-pages)](https://travis-ci.org/mlr-org/mlr-tutorial)

To view the tutorial online see:
 * http://mlr-org.github.io/mlr-tutorial/devel/html/ - targets the github-version of **mlr**
 * http://mlr-org.github.io/mlr-tutorial/release/html/ - targets the cran-version of **mlr**.

## Building the tutorial locally

### First steps
Install dependencies:
* `pip install --user mkdocs` or `easy_install --user mkdocs`.
* Install the [math extension for Python-Markdown](https://github.com/mitya57/python-markdown-math):
  After download `chmod a+x setup.py`, edit the first line in the file if you use `python2`, type `python setup.py build` and `python setup.py install`.
* Install R dependencies as required.

### Howto

#### Edit a tutorial section
* Only edit R markdown files in subfolder `src/`.
* Markdown basics:
  * Basic Markdown: https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet
  * RStudio Support: https://support.rstudio.com/hc/en-us/articles/200552086-Using-R-Markdown
  * Knitr options: http://yihui.name/knitr/options
* Link to another tutorial page in file `tutorial_page.md`: `[clever text](tutorial_page.md)`.
* Link to a certain section on a tutorial page:
  If you want to link e.g. to section `### Confusion matrix` on the prediction page in file
  `predict.md` write: `[clever text](predict.md#confusion-matrix)`
  (i.e. use only one `#`, section title all lower case and words separated by hyphens).
  Use this sparingly because links break if the section title changes!
* Link to mlr manual: `[&function]` and `[name](&function)`.
* Link to other manuals: `[&pkg::function]` and `[name](&pkg::function)`.
* Links will only work properly if they point to the *base name* of the help file:
  For example, in order to link to mlr function `foo` documented on help page `bar` write
  `[foo](&bar)` instead of `[&foo]`.
* Link to packages: `[%pkg]` and `[name](%pkg)`.

#### Add a new tutorial section
* Create a new R markdown file in subfolder `src/`.
* Add the new section to the pages configuration in `mkdocs.yml`.

#### Include images
If you want to include an additional image in file `pic.png`:  
* Put this file in subfolder `img/`.
* Add a symlink in directory `custom_theme/img/`: `pic.png -> ../../img/pic.png`.
* When including the image in the R markdown link to `img/pic.png`:  
  `![alt text](img/pic.png "Image Title")`

#### Commit your changes
* Run `./build` to generate new static HTML.
* If everything works:
  Commit and push your changes **except those in** `devel/` to update the tutorial.
  After your commit the HTML pages are automatically built and pushed by Travis.

### More
* "mkdocs serve" starts a http server listening on http://localhost:8000
  and updates the docs on change.
* Sometimes function names collide. These packages must be loaded _first_
  in `build`. That way mlr overwrites these functions again, e.g. `caret::train`.
* The build caches the output of running the R commands in the `cache/` directory.
  If your R setup has changed (e.g. new version of mlr), you should delete
  everything in the cache directory to make sure that the tutorial is
  regenerated with the new code.

  This particularily means: If you encounter an error in building the tutorial
  due to your R setup (e.g. outdated/missing packages) you **have to** delete
  the cache (after updating said packages) to get rid of the error.
