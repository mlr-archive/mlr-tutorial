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

#### Building the Files
* The tutorial is created according to the `Makefile` using `make`.
* `./build` is a wrapper script that calls `make` with multithreading enabled to generate new static HTML.
* The newly generated files are visible in the "local" directory
* Using `make` (or `make -j <NUMBER OF PROCESSORS>`) to build is also supported.
* To clean up caches and intermediate files, run `make clean`
* To build only pdf / html files, run `make pdf` or `make html`. This saves very little time, if at all, and is usually not worth it.
* `make all` will build the same things that `make` builds, plus the `.zip` file of the html pages.
* To copy files from the "local" to the "devel" directory use `make release`. **Only Travis should push changes in the devel directory**, so you usually shouldn't do this.

#### Commit Your Changes
* If everything works:
  Commit and push your changes to update the tutorial. The files in the "local" directory must **not** be committed (and are usually ignored by git).
  After your commit the HTML pages are automatically re-built and pushed by Travis.
  Travis builds its own version of the tutorial and puts it into the "devel" directory.
* If errors occur, debugging may be easier when using `make` instead of `./build`. The difference is that `./build`
  runs tasks in parallel and may make it difficult to inspect errors.

### More
* "mkdocs serve" starts a http server listening on http://localhost:8000
  and updates the docs on change.
* Sometimes function names collide. Lesser used packages must be loaded _first_
  in `./build`. That way mlr overwrites these functions again, e.g. `caret::train`
  is superseded by `mlr::train`.
* The build caches the output of running the R commands in the `cache/` directory,
  and furthermore only runs knitr on those `.Rmd` files that have changed.
  If your R setup has changed (e.g. new version of mlr), you should delete
  everything in the local directories by running `make clean`.

  This particularily means: If you encounter an error in building the tutorial
  due to your R setup (e.g. outdated/missing packages) you **have to** clean intermediate
  results (after updating said packages) to get rid of the error.
