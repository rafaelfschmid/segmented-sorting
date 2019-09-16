in=$1 # input files dir
out=$2 # time files dir

./../scripts/execnthrust.sh newthrust32.exe $in > $out/newthrust32.time
./../scripts/execnthrust.sh newthrust32sched.exe $in > $out/newthrust32sched.time
./../scripts/execnthrust.sh newthrust32thr.exe $in > $out/newthrust32thr.time
./../scripts/execnthrust.sh newthrust32schedthr.exe $in > $out/newthrust32schedthr.time
./../scripts/execnthrust.sh mergeseg.exe $in > $out/mergeseg.time
./../scripts/execnthrust.sh radixseg.exe $in > $out/radixseg.time
./../scripts/execnthrust.sh fixcub.exe $in > $out/fixcub.time
./../scripts/execnthrust.sh fixthrust.exe $in > $out/fixthrust.time

