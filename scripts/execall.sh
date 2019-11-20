in=$1 # input files dir
out=$2 # time files dir

./../scripts/exec.sh bb_segsort/bbsegsort.exe $in 	> $out/bbsegsort.time
./../scripts/exec.sh mergeseg.exe $in 			> $out/mergeseg.time
./../scripts/exec.sh radixseg.exe $in 			> $out/radixseg.time
./../scripts/exec.sh fixcub.exe $in 			> $out/fixcub.time
./../scripts/exec.sh fixthrust.exe $in 			> $out/fixthrust.time
./../scripts/exec.sh cpugputhrust32.exe $in 		> $out/cpugputhrust.time
./../scripts/exec.sh newthrustsemaforo32.exe $in 	> $out/newthrustsemaforo.time
./../scripts/exec.sh newthrust32schedthr.exe $in 	> $out/newthrustschedthr.time
./../scripts/exec.sh newthrust32thr.exe $in 		> $out/newthrustthr.time

#./../scripts/exec.sh fixpassseq.exe $in > $out/fixpassseq.time
#./../scripts/exec.sh fixseq.exe $in > $out/fixseq.time
#./../scripts/exec.sh mseq.exe $in > $out/mseq.time
#./../scripts/exec.sh fixpasscub.exe $in > $out/fixpasscub.time
#./../scripts/exec.sh fixpassthrust.exe $in > $out/fixpassthrust.time
#./../scripts/execnthrust.sh nthrust.exe $in > $out/nthrust.time

