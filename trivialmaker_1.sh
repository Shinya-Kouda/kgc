echo { >> test.json
echo "    \"data\": [" >> test.json
counter=0
while read subject beverb complement
do
    counter=`expr $counter + 1`
    id=$(printf "%09d\n" "${counter}")
    echo "        {" >> test.json
    echo "            \"kg_id\": ${id}," >> test.json
    echo "            \"nlr\": \"$subject $beverb $complement\"," >> test.json
    echo "            \"kgr\": \"Scene Situation Subject $subject hasPredicate equalTo what $complement\"" >> test.json
    echo "        }," >> test.json
done < equalTo.list
sed -i -e "$ s/.$//" test.json
echo "    ]" >> test.json
echo "}" >> test.json