f = open('../01_raw/cased_L-12_H-768_A-12/vocab.txt')
lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
f.close()
# lines2: リスト。要素は1行の文字列データ
counter = 1
for line in lines2:
  if line[0] == '[' and line[-2] == ']':
    print(line[:-1])
    
