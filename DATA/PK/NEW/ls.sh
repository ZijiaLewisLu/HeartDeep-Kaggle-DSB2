list=file_list.txt

echo "[" > $list

for f in \[N4,T5\]*
do
    echo $f
    echo " '$f', " >> $list
done

echo "]" >> $list
