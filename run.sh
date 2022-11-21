DATADIR='/mnt/gpu_data1/kubapok/bigdata2022challenge-encryption/bigdatachallange/BigDataCup2022'

FOLDS=10
SHUFFLE=30

rm OUTPUTS/*

python test-prepare.py $DATADIR
python prepare.py $DATADIR $FOLDS
for (( i=0 ; i<$FOLDS ; i++ )) ; do python train.py S1 $SHUFFLE $i ; done
for (( i=0 ; i<$FOLDS ; i++ )) ; do python train.py S2 $SHUFFLE $i ; done
python merge.py S1 $SHUFFLE $FOLDS
python merge.py S2 $SHUFFLE $FOLDS
