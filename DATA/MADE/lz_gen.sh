t=0
while [ $t -lt 5 ]
do
    python generate.py
    $t=$t+1
done

./move
