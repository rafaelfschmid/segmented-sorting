#!/bin/bash
dir=$1 # output files dir
c=59049
while [ $c -le 43046721 ]
do
	d=1
	while [[ $d -lt $c && $d -le 531441 ]]
	do
		((x=$c/$d))
		./equal.exe $d $x > $dir/$d"_"$c.in
		#echo $d"_"$c".in"
		((d=$d*3))
		sleep 1.0
	done
	((c=$c*3))
done
