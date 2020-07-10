#!/bin/sh
trap "exit" INT

for f in all_output/*/*/; do montage $f/disp*err.png $f/position-3d.png $f/rotation-err.png $f/displacement.png -geometry +2+2 png:- | feh -ZF - ; done