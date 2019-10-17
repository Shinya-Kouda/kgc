predict = ['[CLS]','[Subject]','X','and','Y','[eq]','genius','[END]']
predict_sorted = predict
for i in range(len(predict)):
    p_start = i
    if predict[i][0] == '[' and predict[i][-1] == ']':
        for j in range(i+1,len(predict)):
            if predict[j][0] == '[' and predict[j][-1] == ']':
                p_end = j
                next_special_token_start = j
                next_special_token = predict[j]
                if next_special_token.lower() < predict[i].lower():
                    for k in range(j+1,len(predict)):
                        if predict[k][0] == '[' and predict[k][-1] == ']':
                            next_special_token_end = k
                            break
                        else:
                            next_special_token_end = len(predict)
                    if p_start == 0:
                        tmp1 = []
                    else:
                        tmp1 = predict_sorted[:p_start]
                    tmp2 = predict_sorted[p_start:p_end]
                    tmp3 = predict_sorted[next_special_token_start:next_special_token_end]
                    if next_special_token_end == len(predict):
                        tmp4 = []
                    else:
                        tmp4 = predict_sorted[next_special_token_end:]
                    predict_sorted = tmp1 + tmp3 + tmp2 + tmp4
                    print('tmp1:',tmp1,' tmp2:',tmp2,' tmp3:',tmp3,' tmp4:',tmp4)
                break
    
print(predict_sorted)