prog=$1 #program to run
input=$2 #input files dir

n=32768
while [[ $n -le 134217728 ]]
do
	for s in 1 2 4 8 16 32 64 128 256 512 1024 ; do
        	echo " "
		echo ${s}
		echo ${n}          
     
		./$prog < $input/${s}_${n}.in
        done
	((n=$n*2))
done

