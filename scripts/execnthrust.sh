prog=$1 #program to run
input=$2 #input files dir

n=32768
while [[ $n -le 134217728 ]]
do
	for ((s=1; s<=1048576; s*=2))
	do
		if [ $s == $n ]
		then
			break;
		fi
	
        	echo " "
		echo ${s}
		echo ${n}          
     
		./$prog < $input/${s}_${n}_1.in
        done
	((n=$n*2))
done

