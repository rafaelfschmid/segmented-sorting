#!/bin/bash
prog=$1
dir=$2
c_aux=0
for filename in `ls -tr $dir`; do
	file=$filename
	file=$(echo $file| cut -d'/' -f 3)
	file=$(echo $file| cut -d'.' -f 1)
	c=$(echo $file| cut -d'_' -f 1)
	d=$(echo $file| cut -d'_' -f 2)
	if [ $c_aux != $c ]
	then
		echo " "
		echo $c
		echo $d
		c_aux=$c
	fi
#  for b in `seq 1 10`; do
	./$prog < $dir/$filename
#	done
#	echo " "
done

