-- chooseList should accept arbitrary terms in the branches
(program 0.0.0 [[[(force (force (builtin chooseList))) (con (list integer) [ 0, 1, 2 ])] (con integer 0)] (con integer 1)])