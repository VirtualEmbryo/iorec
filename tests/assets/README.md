In this folder put the necessary assets to run your tests.

- `binary.rec`, `text.rec`, `binary_2d.rec`, `text_2d.rec`: .rec meshes in both encodings.
  `binary.rec` holds 16 closed cells in a medium (label 0); in `text.rec` labels 0 to 3
  meet along a free boundary, so their surfaces are not closed.
- `two_cubes.fe`: a Surface Evolver datafile written for the tests: two unit cubes sharing the
  face x = 1, given by quadrilateral facets, with volume targets 1.
- `two_cubes_evolved.dmp`: the dump written by Surface Evolver 2.70a after loading
  `two_cubes.fe`, `r` (one refinement) and `g 12`. Evolver then printed
  `total_area` 9.26947364644248 and both body volumes 1.00000000083127.
