seed=2023
gpu_id=2

# python main.py --dset office_home --s 1 --t 3 --seed $seed --gpu_id $gpu_id

for s in $(seq 0 3)
do
   for t in $(seq 0 3)
   do
       if [ "$s" -ne "$t" ]; then
           python main.py --dset office_home --s $s --t $t --seed $seed --gpu_id $gpu_id
       fi
   done
done

seed=2021
gpu_id=2

for s in $(seq 0 3)
do
   for t in $(seq 0 3)
   do
       if [ "$s" -ne "$t" ]; then
           python main.py --dset office_home --s $s --t $t --seed $seed --gpu_id $gpu_id
       fi
   done
done

seed=2019
gpu_id=2

for s in $(seq 0 3)
do
   for t in $(seq 0 3)
   do
       if [ "$s" -ne "$t" ]; then
           python main.py --dset office_home --s $s --t $t --seed $seed --gpu_id $gpu_id
       fi
   done
done
