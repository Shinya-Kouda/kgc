predict = ['[Subject]','X','and','Y','[eq]','genius','[asd]','asd','fgh','[722]','asd','asd','[xxx]','[Z]']
predict_sorted = predict
for i in range(len(predict)):
    p_start = i
    if predict_sorted[i][0] == '[' and predict_sorted[i][-1] == ']':
        for j in range(i+1,len(predict_sorted)):
            if predict_sorted[j][0] == '[' and predict_sorted[j][-1] == ']':
                p_end = j
                next_special_token_start = j
                next_special_token = predict[j]
                if next_special_token.lower() < predict_sorted[i].lower():
                    for k in range(j+1,len(predict_sorted)):
                        if predict_sorted[k][0] == '[' and predict_sorted[k][-1] == ']':
                            next_special_token_end = k
                            break
                        else:
                            next_special_token_end = len(predict_sorted)
                    if p_start == 0:
                        tmp1 = []
                    else:
                        tmp1 = predict_sorted[:p_start]
                    tmp2 = predict_sorted[p_start:p_end]
                    tmp3 = predict_sorted[next_special_token_start:next_special_token_end]
                    if next_special_token_end == len(predict_sorted):
                        tmp4 = []
                    else:
                        tmp4 = predict_sorted[next_special_token_end:]
                    predict_sorted = tmp1 + tmp3 + tmp2 + tmp4
                    print('tmp1:',tmp1,' tmp2:',tmp2,' tmp3:',tmp3,' tmp4:',tmp4)
                break
    
print(predict_sorted)

def is_special_id(id):#全部書き直す必要がある
    f = open('../01_raw/cased_L-12_H-768_A-12/vocab.txt')
    lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
    f.close()
    # lines2: リスト。要素は1行の文字列データ
    special_ids = []
    for line in lines2:
        if line[0] == '[' and line[-2] == ']':
            special_ids.append(line[:-1])
    if id in special_ids:
        return True
    else:
        return False

def add_segment_ids(ids):
    kg_segment_ids = [0]
    for i in range(len(ids)):
        if ids[i][0] == '[' and ids[i][-1] == ']':
            kg_segment_ids.append(kg_segment_ids[-1]+1)
        else:
            kg_segment_ids.append(kg_segment_ids[-1])
    return (ids, kg_segment_ids[1:])

print(add_segment_ids(predict))