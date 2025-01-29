# TODO

List of things to do

* Arrange files - ok
* Rename folders - ok
* Initial refactoring
    - rename file names to lower snake case - ok
    - add the gifs and images to own folders - ok
    - refactor the use of ui file - ok
    - refactor import * - ok
    - clarify UI variable names - ok
    - clarify the usage of translate matrix - IMPORTANT!!!
    - organize orphaned functions - static or class method?
    - refactor functions that are outside of classes
    - check parts that are still using fixed pipeline - can they be removed?
    
* Add tests
* Add documentation

## Checking if Function is to be a Private, Static, or Standalone function

* Does it the `self` variable?
  * Yes - private
  * No - is the function used only inside one class?
    * Yes - Static
    * No - Standalone, possible that a Utils class can be made
  
## Dependency Graph

Two ways to build dependency graph

* Using `pydeps`
```
$ pip install pydeps
$ pydeps tsugite -v -T png --noshow --rmprefix tsugite --rankdir BT --cluster --show-cycles

```

* Using `tach`

https://www.gauge.sh/blog/how-to-visualize-your-python-projects-dependency-graph

```
$ pip install tach
$ cd tsugite
$ tach mod    # follow the instructions
$ tach sync
$ tach show   # will generate dot file
$ dot -Tpng tach_module_graph.dot -o ../img/tsugite_graph.png

```

## Notes

https://nrotella.github.io/journal/first-steps-python-qt-opengl.html
https://metamost.com/post/tech/opengl-with-python/02-opengl-with-python-pt2/
https://johannesugb.github.io/gpu-programming/why-do-opengl-proj-matrices-fail-in-vulkan/