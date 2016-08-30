#!/bin/bash
prog1=$1 #program to test
dir1=$2 #test files dir
dir2=$3 #result files dir
dir3=$4 #errors files dir

for filename in `ls -tr $dir1`; do
	file=$filename
	file=$(echo $file| cut -d'/' -f 3)
	file=$(echo $file| cut -d'.' -f 1)
	c=$(echo $file| cut -d'_' -f 1)
	d=$(echo $file| cut -d'_' -f 2)
	echo $c"_"$d".in"

	./$prog1 < $dir1/$filename > $dir2/"test.out"

	if ! cmp -s $dir2/"test.out" $dir2/$c"_"$d".out"; then
		mkdir -p $dir3
		cat $dir2/"test.out" > $dir3/$c"_"$d".out"
		echo "There are something wrong."
		#break;
	else
		echo "Everthing ok."		
	fi
done

#echo "Do you want to remove the files at '"$dir3"'?"
#read -p "[Yes][No]: " yn
#if [[ $yn == "Yes" ]]
#then 
#	rm $dir3/*
#	echo "yes"
#else
#	echo "no"
#	exit 0
#fi

