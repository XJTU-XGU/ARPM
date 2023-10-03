seed=2023
gpu_id=0

## office-home
#for s in $(seq 0 3)
#do
#    for t in $(seq 0 3)
#    do
#        if [ "$s" -ne "$t" ]; then
#            python main.py --dset office-home --s $s --t $t --seed $seed --gpu_id $gpu_id
#        fi
#    done
#done
#
##visda
#python main.py --dset visda-2017 --s 0 --t 1 --seed $seed --gpu_id $gpu_id

#imagenet-caltech
for s in $(seq 0 1)
do
    for t in $(seq 0 1)
    do
        if [ "$s" -ne "$t" ]; then
            python main.py --dset imagenet_caltech --s $s --t $t --seed $seed --gpu_id $gpu_id
        fi
    done
done

#office
for s in $(seq 0 2)
do
    for t in $(seq 0 2)
    do
        if [ "$s" -ne "$t" ]; then
            python main.py --dset office --s $s --t $t --seed $seed --gpu_id $gpu_id
        fi
    done
done

#domainnet
for s in $(seq 0 3)
do
    for t in $(seq 0 3)
    do
        if [ "$s" -ne "$t" ]; then
            python main.py --dset domainnet --s $s --t $t --seed $seed --gpu_id $gpu_id
        fi
    done
done
