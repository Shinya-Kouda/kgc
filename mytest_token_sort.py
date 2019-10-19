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