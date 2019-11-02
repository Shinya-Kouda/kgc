echo { >> trivial-v1.0.json
echo "    \"data\": [" >> trivial-v1.0.json
counter=0
while read subject beverb complement
do
    counter=`expr $counter + 1`
    id=$(printf "%09d\n" "${counter}")
    echo "        {" >> trivial-v1.0.json
    echo "            \"id\": ${id}," >> trivial-v1.0.json
    echo "            \"nlr\": \"$subject $beverb $complement\"," >> trivial-v1.0.json
    echo "            \"kgr\": \"\$Scene\$ \$Situation\$ \$Subject\$ $subject %hasPredicate% !equalTo! %what% $complement\"" >> trivial-v1.0.json
    echo "        }," >> trivial-v1.0.json
done < 2_structure.list

while read subject predicate wh object
do
    counter=`expr $counter + 1`
    id=$(printf "%09d\n" "${counter}")
    echo "        {" >> trivial-v1.0.json
    echo "            \"id\": ${id}," >> trivial-v1.0.json
    echo "            \"nlr\": \"$subject $predicate $object\"," >> trivial-v1.0.json
    echo "            \"kgr\": \"\$Scene\$ \$Situation\$ \$Subject\$ $subject %hasPredicate% $predicate %$wh% $object\"" >> trivial-v1.0.json
    echo "        }," >> trivial-v1.0.json
done < 3_structure.list

while read subject predicate who what
do
    counter=`expr $counter + 1`
    id=$(printf "%09d\n" "${counter}")
    echo "        {" >> trivial-v1.0.json
    echo "            \"id\": ${id}," >> trivial-v1.0.json
    echo "            \"nlr\": \"$subject $predicate $who $what\"," >> trivial-v1.0.json
    echo "            \"kgr\": \"\$Scene\$ \$Situation\$ \$Subject\$ $subject %hasPredicate% $predicate $who $what\"" >> trivial-v1.0.json
    echo "        }," >> trivial-v1.0.json
done < 4_structure.list

sed -i -e "$ s/.$//" trivial-v1.0.json
sed -i -e "s/_/ /g" trivial-v1.0.json
echo "    ]" >> trivial-v1.0.json
echo "}" >> trivial-v1.0.json